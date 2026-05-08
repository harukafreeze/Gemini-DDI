# Main Gemini-DDI architecture
# prism/model.py (Final High-Performance Version)
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from .layers import PrismBlock, GINConv, ModalitySwitch

class SpotlightPool(Layer):
    def __init__(self, d_model, **kwargs):
        super(SpotlightPool, self).__init__(**kwargs)
        self.q = self.add_weight(shape=(1, 1, d_model), name="query", initializer="glorot_uniform")
        self.attn = Dense(1, name="at")
    def call(self, f, m):
        query = tf.tile(tf.cast(self.q, f.dtype), [tf.shape(f)[0], tf.shape(f)[1], 1])
        s = tf.squeeze(self.attn(tf.nn.tanh(f + query)), -1)
        # 评估致盲
        s = tf.cast(s, tf.float32) + (1.0 - tf.cast(m, tf.float32)) * -1e9
        w = tf.expand_dims(tf.cast(tf.nn.softmax(s), f.dtype), -1)
        return tf.reduce_sum(f * w, axis=1)

def PRISM_DDI(config, drop_rate=0.2):
    # 1. Inputs
    ins = {k: Input(shape=(None, 77) if 'at' in k and 'feat' in k else (None, None) if 'adj' in k else (None, 210) if 'global' in k else (None,), name=k) for k in ["atomic_features_a", "atomic_adj_a", "atomic_features_b", "atomic_adj_b", "motif_ids_a", "motif_adj_a", "motif_ids_b", "motif_adj_b", "global_features_a", "global_features_b"]}
    
    # 2. Shared Layers
    atom_emb = Dense(320, name="shared_atom_emb")
    motif_emb = Embedding(config.MOTIF_VOCAB_SIZE, 320, name="shared_motif_embed")
    gin = GINConv(320, name="shared_gin")
    gnorm = LayerNormalization(name="shared_gin_norm")
    
    macro_bn = BatchNormalization(name="macro_bn_base")
    se_d1 = Dense(52, activation='relu', name="macro_se_d1")
    se_d2 = Dense(210, activation='sigmoid', name="macro_se_d2")
    macro_d1 = Dense(256, activation='gelu', name="macro_dense_1")
    macro_d2 = Dense(320, activation='gelu', name="macro_dense_2")

    at_blks = [PrismBlock(320, 8, 1280, name=f"at_blk_{i}") for i in range(6)]
    mo_blks = [PrismBlock(320, 8, 1280, name=f"mo_blk_{i}") for i in range(6)]

    sp_at = SpotlightPool(320, name="sp_atom_pool")
    sp_mo = SpotlightPool(320, name="sp_motif_pool")
    mi_proj = Dense(320, activation='gelu', name="micro_proj_layer")
    
    fusion_gate = Dense(320, activation='sigmoid', name="fusion_gate_layer")
    fusion_norm = LayerNormalization(name="fusion_norm_layer")
    
    switch_a = ModalitySwitch(drop_rate=drop_rate, name="switch_a")
    switch_b = ModalitySwitch(drop_rate=drop_rate, name="switch_b")

    # 3. Macro-Encoding Logic (Bulletproof Version)
    def macro_enc(gf):
        # 1. 强制转为 float32 防止在这一步就溢出
        gf_f32 = tf.cast(gf, tf.float32)
        # 2. 绝对安全范围的 Log 压缩与 Clip
        safe_gf = tf.math.log1p(tf.clip_by_value(tf.math.abs(gf_f32), 0.0, 1e6))
        # 3. 转回混合精度的自适应类型
        safe_gf = tf.cast(safe_gf, macro_bn.compute_dtype)
        x = macro_bn(tf.reshape(safe_gf, [-1, config.GLOBAL_FEATURE_DIM]))
        se = se_d2(se_d1(x))
        return macro_d2(Dropout(config.DROPOUT_RATE)(macro_d1(x * se)))

    def get_repr(p):
        xa = gnorm(Add()([atom_emb(ins[f"atomic_features_{p}"]), gin([atom_emb(ins[f"atomic_features_{p}"]), tf.cast(ins[f"atomic_adj_{p}"]>0, tf.float32)])]))
        ma = gnorm(Add()([motif_emb(ins[f"motif_ids_{p}"]), gin([motif_emb(ins[f"motif_ids_{p}"]), ins[f"motif_adj_{p}"]])]))
        mask_a = tf.cast(tf.reduce_sum(tf.abs(ins[f"atomic_features_{p}"]),-1)>0, tf.float32)
        mask_m = tf.cast(ins[f"motif_ids_{p}"]>0, tf.float32)
        return xa, ma, mask_a, mask_m

    xa, ma, maa, mma = get_repr('a'); xb, mb, mab, mmb = get_repr('b')

    for blk in at_blks:
        xa, xb = blk([xa, maa, xb, mab, ins["atomic_adj_a"]]), blk([xb, mab, xa, maa, ins["atomic_adj_b"]])
    for blk in mo_blks:
        ma, mb = blk([ma, mma, mb, mmb, tf.cast(ins["motif_adj_a"], tf.int32)]), blk([mb, mmb, ma, mma, tf.cast(ins["motif_adj_b"], tf.int32)])

    def full_fuse(x_at, x_mo, m_at, m_mo, gf, sw):
        v_mi = sw(mi_proj(Concatenate()([sp_at(x_at, m_at), sp_mo(x_mo, m_mo)])))
        v_ma = macro_enc(gf)
        g = fusion_gate(Concatenate()([v_mi, v_ma]))
        return fusion_norm(g * v_mi + (1.0 - g) * v_ma)

    va_f, vb_f = full_fuse(xa, ma, maa, mma, ins["global_features_a"], switch_a), full_fuse(xb, mb, mab, mmb, ins["global_features_b"], switch_b)

    # 4. Final Head (Heavy Residual Armor - 绝杀版本)
    x = LayerNormalization(name="inter_head_norm")(Concatenate()([va_f, vb_f, va_f*vb_f, tf.math.square(va_f-vb_f)]))
    
    x = Dense(1024, activation='gelu', name="head_d1")(x)
    x = Dropout(config.DROPOUT_RATE)(x)
    
    # 核心残差块
    res = Dense(1024, name="head_res_proj")(x)
    x = Dense(1024, activation='gelu', name="head_d2")(x)
    x = Dense(1024, activation='gelu', name="head_d3")(x)
    x = Add()([x, res]) 
    x = LayerNormalization()(x)
    
    x = Dense(512, activation='gelu', name="head_final_dense")(x)
    logits = Dense(config.NUM_CLASSES, name="final_logits", dtype='float32')(x)

    return Model(inputs=ins, outputs=logits)