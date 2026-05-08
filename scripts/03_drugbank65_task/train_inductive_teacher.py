# Train teacher without S2 drugs
import os, sys, tensorflow as tf
from tensorflow.keras import mixed_precision

# 环境初始化
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
tf.keras.backend.clear_session()
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)
import configs.db65_cold_config as config
from prism.model import PRISM_DDI
from prism.dataloader import PrismDdiDataloader

class SpecificLRDrop(tf.keras.callbacks.Callback):
    def __init__(self, target_lr=2e-5, patience=6):
        super().__init__()
        self.target_lr, self.patience, self.best_val_acc, self.wait, self.dropped = target_lr, patience, 0.0, 0, False
    def on_epoch_end(self, epoch, logs=None):
        if self.dropped: return
        val_acc = logs.get('val_accuracy')
        if val_acc and val_acc > self.best_val_acc:
            self.best_val_acc, self.wait = val_acc, 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.model.optimizer.learning_rate.assign(self.target_lr)
                self.dropped = True
                print(f"\n[LR-System] 降温至 {self.target_lr}")

def main():
    for fold in range(5):
        print(f"\n🚀 Training Inductive Teacher: Fold {fold}")
        fold_dir = os.path.join(config.INDUCTIVE_EXP_DIR, 'tfrecords', f'fold_{fold}')
        dl = PrismDdiDataloader(config); dl.tfrec_dir = fold_dir
        train_ds = dl.get_dataset(mode='train').repeat()
        valid_ds = dl.get_dataset(mode='valid')

        model = PRISM_DDI(config, drop_rate=0.15)
        _ = model(next(iter(valid_ds.take(1)))[0], training=False)
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), 
                      loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.02),
                      metrics=['accuracy'])
        
        t_path = os.path.join(config.MODEL_SAVE_PATH, f"Inductive_Teacher_F{fold}.h5")
        cbs = [tf.keras.callbacks.ModelCheckpoint(t_path, save_best_only=True, monitor='val_accuracy', save_weights_only=True),
               tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=12, restore_best_weights=True),
               SpecificLRDrop()]
        
        model.fit(train_ds, validation_data=valid_ds, epochs=80, steps_per_epoch=1000, callbacks=cbs, verbose=1)

if __name__ == "__main__": main()