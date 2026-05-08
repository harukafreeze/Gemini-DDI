# Graph convolutions and Attention layers
# prism/layers.py (Final PR-Grade Version)
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Dropout, LayerNormalization, BatchNormalization, Activation, Embedding

class ModalitySwitch(Layer):
    def __init__(self, drop_rate=0.2, **kwargs):
        super(ModalitySwitch, self).__init__(**kwargs)
        self.drop_rate = drop_rate
    def call(self, inputs, training=None):
        if not training: return inputs
        random_val = tf.random.uniform(shape=[], minval=0., maxval=1.)
        return tf.cond(random_val < self.drop_rate, lambda: tf.zeros_like(inputs), lambda: inputs)

class GINConv(Layer):
    def __init__(self, units, **kwargs):
        super(GINConv, self).__init__(**kwargs)
        self.units = units
        self.epsilon = tf.Variable(0.0, trainable=True, name="epsilon")
        self.d1 = Dense(units, use_bias=False, name="dense")
        self.bn1 = BatchNormalization(name="batch_normalization")
        self.d2 = Dense(units, use_bias=False, name="dense_1")
        self.bn2 = BatchNormalization(name="batch_normalization_1")

    def call(self, inputs):
        x, adj = inputs
        # 【混合精度修复】：确保所有输入转为一致的浮点类型
        x = tf.cast(x, self.compute_dtype)
        adj = tf.cast(adj, self.compute_dtype)
        
        neighbor_sum = tf.matmul(adj, x)
        agg = (1.0 + tf.cast(self.epsilon, x.dtype)) * x + neighbor_sum
        h = tf.nn.relu(self.bn1(self.d1(agg)))
        return tf.nn.relu(self.bn2(self.d2(h)))

class GraphBiasedAttention(Layer):
    def __init__(self, d_model, num_heads, has_bias=True, **kwargs):
        super(GraphBiasedAttention, self).__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = d_model // num_heads
        self.has_bias = has_bias
        self.wq = Dense(d_model, name="wq")
        self.wk = Dense(d_model, name="wk")
        self.wv = Dense(d_model, name="wv")
        self.dense = Dense(d_model, name="out_dense")
        if self.has_bias:
            self.bond_bias = Embedding(6, num_heads, name="bond_bias")

    def call(self, q, k, v, mask, adj=None):
        bs = tf.shape(q)[0]
        def transform(x):
            x = tf.reshape(x, (bs, -1, self.num_heads, self.depth))
            return tf.transpose(x, [0, 2, 1, 3])

        qh, kh, vh = transform(self.wq(q)), transform(self.wk(k)), transform(self.wv(v))
        
        # 【混合精度修复】：加入 epsilon 防止除以 0，确保数值稳定
        scale = tf.math.sqrt(tf.cast(self.depth, qh.dtype)) + tf.keras.backend.epsilon()
        score = tf.matmul(qh, kh, transpose_b=True) / scale
        
        if self.has_bias and adj is not None:
            bias = self.bond_bias(tf.cast(adj, tf.int32))
            score += tf.cast(tf.transpose(bias, [0, 3, 1, 2]), score.dtype)
            
        if mask is not None:
            # 【核心修复】：float16下-1e9会溢出，必须改用-1e4
            mask_val = (1.0 - tf.cast(mask[:, tf.newaxis, tf.newaxis, :], score.dtype)) * -1e4 
            score += mask_val
            
        attn = tf.nn.softmax(score, axis=-1)
        out = tf.transpose(tf.matmul(tf.cast(attn, vh.dtype), vh), [0, 2, 1, 3])
        return self.dense(tf.reshape(out, (bs, -1, self.d_model)))

class PrismBlock(Layer):
    def __init__(self, d_model, num_heads, d_ff, **kwargs):
        super(PrismBlock, self).__init__(**kwargs)
        self.alpha_self = tf.Variable(0.0, trainable=True, name="alpha_self")
        self.alpha_cross = tf.Variable(0.0, trainable=True, name="alpha_cross")
        self.alpha_ffn = tf.Variable(0.0, trainable=True, name="alpha_ffn")
        self.sa = GraphBiasedAttention(d_model, num_heads, has_bias=True, name="sa")
        self.ca = GraphBiasedAttention(d_model, num_heads, has_bias=False, name="ca")
        self.f1 = Dense(d_ff, activation='gelu', name="f1")
        self.f2 = Dense(d_model, name="f2")

    def call(self, inputs):
        x_s, m_s, x_c, m_c, adj = inputs
        dtype = x_s.dtype
        x_s = x_s + tf.cast(self.alpha_self, dtype) * self.sa(x_s, x_s, x_s, m_s, adj)
        x_s = x_s + tf.cast(self.alpha_cross, dtype) * self.ca(x_s, x_c, x_c, m_c, None)
        x_s = x_s + tf.cast(self.alpha_ffn, dtype) * self.f2(self.f1(x_s))
        return x_s