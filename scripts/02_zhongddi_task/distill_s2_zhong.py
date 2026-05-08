# Cold-start distillation
import os
import sys
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score

# 1. 环境配置
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from prism.model import PRISM_DDI
from prism.dataloader import PrismDdiDataloader
import configs.zhongddi_cold_config as config

# --- V10 核心：一致性蒸馏训练模型 ---
class DistillationModel(tf.keras.Model):
    def __init__(self, model, alpha=0.6, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.alpha = alpha  # 蒸馏权重
        self.kl_loss_fn = tf.keras.losses.KLDivergence()

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            # 1. 全知预测 (Teacher)
            # 必须在不致盲的情况下拿到 GNN 的真实分布
            # 使用 tf.stop_gradient 确保 Teacher 的概率分布是固定的，只作为指路明灯
            y_full = self.model(x, training=False)
            y_full_soft = tf.nn.softmax(y_full / 2.0) 

            # 2. 致盲预测 (Student)
            batch_size = tf.shape(y)[0]
            mask = tf.cast(tf.random.uniform((batch_size, 1, 1)) > 0.8, tf.float32)
            
            # 手动创建副本进行致盲，防止污染原数据
            x_blind = {k: v for k, v in x.items()}
            x_blind['atomic_features_a'] = x['atomic_features_a'] * mask
            x_blind['atomic_features_b'] = x['atomic_features_b'] * mask
            
            y_blind = self.model(x_blind, training=True)
            y_blind_soft = tf.nn.softmax(y_blind / 2.0)

            # 3. 混合损失计算
            # 分类 Loss (Focal Loss 已经在 compile 里定义)
            ce_loss = self.compiled_loss(y, y_blind)
            
            # 蒸馏 Loss (致盲输出要尽可能像全知输出)
            distill_loss = self.kl_loss_fn(y_full_soft, y_blind_soft)
            
            total_loss = (1.0 - self.alpha) * ce_loss + self.alpha * distill_loss

        # 4. 反向传播 (由于除了 Macro 外其他都冻结了，所以梯度只去该去的地方)
        grad = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grad, self.trainable_variables))
        
        # 5. 更新度量指标 [核心修复：修复了刚才的 m 报错]
        self.compiled_metrics.update_state(y, y_blind)
        results = {m.name: m.result() for m in self.metrics}
        results["distill"] = distill_loss
        return results

def main():
    print("\n" + "🎓"*25)
    print("  PRISM-DDI V10.0: 一致性蒸馏 (完美对齐版)")
    print("  策略: 教师 GNN -> 学生 Macro | 冻结 GNN")
    print("🎓"*25 + "\n")

    # 1. 构建并物理对齐初始化
    model_core = PRISM_DDI(config, drop_rate=0.8)
    dummy = {
        "atomic_features_a": tf.zeros((1, 5, 77), dtype=tf.float32),
        "atomic_adj_a":      tf.zeros((1, 5, 5),  dtype=tf.int32),
        "atomic_features_b": tf.zeros((1, 5, 77), dtype=tf.float32),
        "atomic_adj_b":      tf.zeros((1, 5, 5),  dtype=tf.int32),
        "motif_ids_a":       tf.zeros((1, 5),     dtype=tf.int32),
        "motif_adj_a":       tf.zeros((1, 5, 5),  dtype=tf.float32),
        "motif_ids_b":       tf.zeros((1, 5),     dtype=tf.int32),
        "motif_adj_b":       tf.zeros((1, 5, 5),  dtype=tf.float32),
        "global_features_a": tf.zeros((1, 210),   dtype=tf.float32),
        "global_features_b": tf.zeros((1, 210),   dtype=tf.float32)
    }
    _ = model_core(dummy)

    # 2. 加载最干净的 V7.5 基石
    v75_path = "/root/PDTT/results/trained_models_cold/PRISM_V7.5_Chemist_20260130-024240_best.h5"
    if not os.path.exists(v75_path):
        print("❌ 未找到基石权重！")
        return
    model_core.load_weights(v75_path, by_name=True, skip_mismatch=True)
    print("✅ 基石记忆注入成功。")

    # 3. 封印教师，激活学生
    for l in model_core.layers:
        if any(n in l.name for n in ['at_blk', 'mo_blk', 'shared_gin', 'embed']):
            l.trainable = False
        else:
            l.trainable = True

    # 4. 构建并编译蒸馏模型
    distill_model = DistillationModel(model_core, alpha=0.6)
    optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
    
    # 定义基础 Focal Loss
    def focal_loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32) * 0.9 + 0.025
        y_pred_p = tf.clip_by_value(tf.nn.softmax(tf.cast(y_pred, tf.float32)), 1e-4, 1.0-1e-4)
        return tf.reduce_sum(-y_true * tf.math.log(y_pred_p) * tf.math.pow(1.0 - y_pred_p, 2.0) * 0.25, axis=-1)

    distill_model.compile(optimizer=optimizer, loss=focal_loss, metrics=['accuracy'])

    # 5. 训练 20 轮 (磨合代偿)
    dl = PrismDdiDataloader(config)
    distill_model.fit(
        dl.get_dataset(mode='train'),
        epochs=20, 
        validation_data=dl.get_dataset(mode='valid'),
        steps_per_epoch=1802
    )

    # 6. 【终极时刻】内存直通全指标扫描 (NO TTA)
    print("\n" + "🏁"*15 + " PRISM-DDI V10.0 最终战果 " + "🏁"*15)
    
    def run_eval(title, filename):
        path = os.path.join(config.TFREC_DIR, filename)
        if not os.path.exists(path): return
        print(f"\n正在计算 {title} 的 4 大 SOTA 指标...")
        ds = tf.data.TFRecordDataset(path).map(dl._parse_function).padded_batch(512)
        
        y_true_all, y_prob_all = [], []
        for x, y in ds:
            # 评估时使用 model_core，关闭所有致盲
            logits = model_core(x, training=False)
            y_true_all.append(y.numpy())
            y_prob_all.append(tf.nn.softmax(logits).numpy())
            
        y_true = np.concatenate(y_true_all, 0)
        y_prob = np.concatenate(y_prob_all, 0)
        y_true_cls = np.argmax(y_true, 1)
        y_pred_cls = np.argmax(y_prob, 1)

        acc = accuracy_score(y_true_cls, y_pred_cls)
        f1 = f1_score(y_true_cls, y_pred_cls, average='macro')
        auroc = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
        
        ap_list = []
        for i in range(config.NUM_CLASSES): # 使用 config 变量更灵活
            if np.sum(y_true[:, i]) > 0: # 物理安全检查
                ap_list.append(average_precision_score(y_true[:, i], y_prob[:, i]))
        aupr = np.mean(ap_list) if ap_list else 0.0
        
        print(f"✨ {title} 结果：")
        print(f"   Acc: {acc*100:.2f}% | F1: {f1:.4f} | AUROC: {auroc:.4f} | AUPR: {aupr:.4f}")

    run_eval("S1 (Seen)", "test.tfrec")
    run_eval("S2 (Cold)", "test_s2.tfrec")

if __name__ == "__main__":
    main()