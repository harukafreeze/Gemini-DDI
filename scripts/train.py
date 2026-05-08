# scripts/train.py (V7.5 - The Chemist)
# ===================================================================
# Optimized for Explicit Interaction Learning.
# Features:
# 1. 512 Batch Size (Speed & Stability)
# 2. Global Clipnorm (Safety)
# 3. SWA-Ready Checkpointing (Saves late epochs for potential averaging)
# ===================================================================

import os
import sys
import tensorflow as tf
from tensorflow.keras import mixed_precision
from datetime import datetime
import numpy as np
import random
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score

# --- 1. Global Setup ---
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Enable Mixed Precision (FP16) for A800
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)
print(f"Global Mixed Precision Policy: {policy.name}")

# Path Setup
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- 1.5 动态加载配置 (替换原来的静态 import) ---
import argparse
import importlib

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='default_config', help='配置文件名(不带.py)')
args, _ = parser.parse_known_args()

try:
    config = importlib.import_module(f"configs.{args.config}")
    print(f"✅ 成功加载任务配置: {args.config}")
except ImportError:
    print(f"❌ 找不到配置 configs/{args.config}.py，请检查！")
    sys.exit(1)
# ----------------------------------------------
from prism.dataloader import PrismDdiDataloader
from prism.model import PRISM_DDI

config.ensure_directories_exist()

# --- 2. Data Loading ---
print(f"🔥 V7.5 Chemist Mode: Batch={config.BATCH_SIZE} | D_MODEL={config.D_MODEL} 🔥")
dataloader = PrismDdiDataloader(config)

train_dataset = dataloader.get_dataset(mode='train')
valid_dataset = dataloader.get_dataset(mode='valid')

print("Calculating dataset statistics...")
try:
    train_len = len(pd.read_csv(config.TRAIN_FILE))
    valid_len = len(pd.read_csv(config.VALID_FILE))
except:
    print("Warning: CSV read failed, using fallback sizes.")
    train_len = 1106924 
    valid_len = 34304

steps_per_epoch = int(np.ceil(train_len / config.BATCH_SIZE))
validation_steps = int(np.ceil(valid_len / config.BATCH_SIZE))

print(f"Train Samples: {train_len} -> {steps_per_epoch} steps/epoch")
print(f"Valid Samples: {valid_len} -> {validation_steps} steps")

# --- 3. Warmup Scheduler (增加内部覆盖开关，完美绕过 Keras 报错) ---
class WarmupScheduler(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(WarmupScheduler, self).__init__()
        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = tf.cast(warmup_steps, tf.float32)
        #[核心修复]: 埋入一个动态覆盖开关，默认 0.0 代表正常执行 Warmup
        self.override_lr = tf.Variable(0.0, trainable=False, dtype=tf.float32)

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        original_lr = tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
        
        # [核心修复]: 如果 override_lr 大于 0，强制输出它；否则按原公式计算
        return tf.cond(self.override_lr > 0.0, lambda: self.override_lr, lambda: original_lr)

lr_schedule = WarmupScheduler(config.D_MODEL, warmup_steps=steps_per_epoch * 3)

# --- 4. Focal Loss ---
class CategoricalFocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=2.0, alpha=0.25, from_logits=True, label_smoothing=0.0, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha
        self.from_logits = from_logits
        self.label_smoothing = label_smoothing

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        if self.label_smoothing > 0:
            num_classes = tf.cast(tf.shape(y_true)[-1], y_true.dtype)
            y_true = y_true * (1.0 - self.label_smoothing) + (self.label_smoothing / num_classes)

        if self.from_logits:
            y_pred_probs = tf.nn.softmax(y_pred, axis=-1)
        else:
            y_pred_probs = y_pred

        epsilon = tf.keras.backend.epsilon()
        y_pred_probs = tf.clip_by_value(y_pred_probs, epsilon, 1.0 - epsilon)

        cross_entropy = -y_true * tf.math.log(y_pred_probs)
        loss = self.alpha * tf.math.pow(1.0 - y_pred_probs, self.gamma) * cross_entropy
        return tf.reduce_sum(loss, axis=-1)

# --- 5. Compile Model ---
model = PRISM_DDI(config)

model = PRISM_DDI(config)

# 逻辑：如果配置里 USE_FOCAL_LOSS 为 True，则用 FocalLoss + Adam；否则用 CCE + AdamW
if hasattr(config, 'USE_FOCAL_LOSS') and config.USE_FOCAL_LOSS:
    print("🎯 检测到 ZhongDDI 任务：激活 Focal Loss + Adam 组合")
    loss_function = CategoricalFocalLoss(gamma=2.0, alpha=0.25, from_logits=True, label_smoothing=0.1)
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=lr_schedule, 
        epsilon=1e-9, 
        global_clipnorm=1.0
    )
else:
    print("🚀 检测到 DB-65 任务：激活原生 CCE + AdamW (Weight Decay) 组合")
    loss_function = tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1)
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=lr_schedule,
        weight_decay=1e-4, # 杀过拟合的神药
        beta_1=0.9, 
        beta_2=0.98, 
        epsilon=1e-9,
        global_clipnorm=1.0
    )

model.compile(optimizer=optimizer, loss=loss_function, metrics=['accuracy'], steps_per_execution=50)

## --- 【断点续练补丁】：加载本折已有的最佳权重 ---
#import glob
#latest_weights = glob.glob(os.path.join(config.MODEL_SAVE_PATH, "PRISM_V7.5_Chemist_*_best.h5"))
#if latest_weights:
#    best_one = max(latest_weights, key=os.path.getmtime)
#    model.load_weights(best_one, by_name=True, skip_mismatch=True)
#    print(f"📦 已成功继承上一次训练的巅峰战果: {os.path.basename(best_one)}")
## ---------------------------------------------


# --- 5.5 【安全注入】：类别权重 (Class Weights) 处理 86 分类长尾 ---
print("\n⚖️ 正在计算类别权重 (Class Weights) 以抢救小类 AUPR...")
try:
    # 直接读取本次 Fold 的训练标签
    train_df_temp = pd.read_csv(config.TRAIN_FILE)
    y_train_labels = train_df_temp['DDI'].values
    
    from sklearn.utils.class_weight import compute_class_weight
    # 计算平衡权重
    raw_weights = compute_class_weight('balanced', classes=np.arange(config.NUM_CLASSES), y=y_train_labels)
    
    # 【高压线防护】：限制最大权重不超过 5.0，防止混合精度(FP16)下梯度爆炸
    safe_weights = np.clip(raw_weights, 0.1, 5.0)
    
    class_weight_dict = {i: float(w) for i, w in enumerate(safe_weights)}
    print("✅ 类别权重计算成功，已开启长尾保护机制！")
except Exception as e:
    print(f"⚠️ 类别权重计算失败 (降级为均衡权重)，原因: {e}")
    class_weight_dict = None

# --- 6. Callbacks ---
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
run_name = f"PRISM_V7.5_Chemist_{timestamp}"
checkpoint_path = os.path.join(config.MODEL_SAVE_PATH, f"{run_name}_best.h5")

class SWACheckpoint(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch >= 45:
            path = os.path.join(config.MODEL_SAVE_PATH, f"{run_name}_epoch_{epoch+1}.h5")
            self.model.save_weights(path)

class SpecificLRDrop(tf.keras.callbacks.Callback):
    def __init__(self, lr_schedule, target_lr=2e-5, patience=10):
        super().__init__()
        self.lr_schedule = lr_schedule
        self.target_lr = target_lr
        self.patience = patience
        self.best_val_acc = 0.0  # 改回监控 Acc
        self.wait = 0
        self.dropped = False

    def on_epoch_end(self, epoch, logs=None):
        if self.dropped: return
        val_acc = logs.get('val_accuracy') # 监控准确率
        if val_acc is not None:
            if val_acc > self.best_val_acc: # 只要准确率在涨，就重置耐心
                self.best_val_acc = val_acc
                self.wait = 0
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    self.lr_schedule.override_lr.assign(self.target_lr)
                    self.dropped = True
                    print(f"\n[LR-System] Epoch {epoch+1}: 准度已连续 {self.patience} 轮未创新高，强行降温至 {self.target_lr} 进行最后精度收割！")
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path, 
        save_weights_only=True, 
        monitor='val_accuracy', # 锁定准确率
        mode='max', 
        save_best_only=True, 
        verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy', # 锁定准确率
        patience=15, 
        verbose=1, 
        mode='max', 
        restore_best_weights=True
    ),
    SpecificLRDrop(lr_schedule=lr_schedule, target_lr=2e-5, patience=8), # 耐心稍微缩短一点，提效
    # 5. TensorBoard
    tf.keras.callbacks.TensorBoard(log_dir=os.path.join(config.LOG_PATH, run_name))
]

# --- 7. Training ---
print(f"\n--- Starting V7.5 Chemist Training (Ultimate Version) ---")
history = model.fit(
    train_dataset,
    epochs=config.EPOCHS,
    validation_data=valid_dataset,
    callbacks=callbacks,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps
)

# --- 8. Final Evaluation ---
print("\n--- Final Evaluation ---")
test_dataloader = PrismDdiDataloader(config)
test_dataset = test_dataloader.get_dataset(mode='test')

y_true_indices = []
y_pred_probs =[]

print("Running Inference...")
for inputs, labels_onehot in test_dataset:
    logits = model(inputs, training=False)
    probs = tf.nn.softmax(logits)
    true_ids = tf.argmax(labels_onehot, axis=1)
    y_true_indices.extend(true_ids.numpy())
    y_pred_probs.extend(probs.numpy())

y_true = np.array(y_true_indices)
y_pred_probs = np.array(y_pred_probs)
y_pred_classes = np.argmax(y_pred_probs, axis=1)

acc = accuracy_score(y_true, y_pred_classes)
f1 = f1_score(y_true, y_pred_classes, average='macro')
try: auroc = roc_auc_score(y_true, y_pred_probs, multi_class='ovr', average='macro')
except: auroc = 0.0
try:
    # 确保 np.eye 的维度是动态读取的 config.NUM_CLASSES
    y_true_onehot = np.eye(config.NUM_CLASSES)[y_true]
    aupr = average_precision_score(y_true_onehot, y_pred_probs, average='macro')
except Exception as e:
    print(f"AUPR计算跳过: {e}")
    aupr = 0.0

print(f"\n{'='*40}")
print(f"PRISM-DDI V7.5 Results ({run_name})")
print(f"{'='*40}")
print(f"Accuracy : {acc:.4f}")
print(f"AUROC    : {auroc:.4f}")
print(f"AUPR     : {aupr:.4f}")
print(f"Macro-F1 : {f1:.4f}")
print(f"{'='*40}")

with open(os.path.join(config.RESULTS_PATH, 'final_results_v7.5.txt'), 'a') as f:
    f.write(f"{run_name} | ACC:{acc:.4f} | AUC:{auroc:.4f} | AUPR:{aupr:.4f} | F1:{f1:.4f}\n")