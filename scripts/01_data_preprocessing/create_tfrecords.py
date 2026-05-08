# Convert CSVs to TFRecords
# scripts/01_data_preprocessing/create_tfrecords.py
import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

# --- 动态寻址：确保能找到根目录 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 动态加载配置 (优先尝试 S2 隔离配置，其次默认配置)
try:
    import configs.db65_inductive_config as config
except ImportError:
    import configs.zhongddi_warm_config as config

def _bytes_feature(value):
    """序列化底层工具"""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() 
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def serialize_example(data_a, data_b, label):
    """
    全动态双视图序列化引擎
    """
    # --- 药物 A ---
    adj_a = data_a['atomic_adj'].astype(np.int32)
    motif_adj_a = data_a['motif_adj'].astype(np.float32)
    np.fill_diagonal(motif_adj_a, 1.0) 
    global_a = data_a['global_features'].astype(np.float32)

    # --- 药物 B ---
    adj_b = data_b['atomic_adj'].astype(np.int32)
    motif_adj_b = data_b['motif_adj'].astype(np.float32)
    np.fill_diagonal(motif_adj_b, 1.0)
    global_b = data_b['global_features'].astype(np.float32)

    # 【核心：全动态标签维度对齐】
    label_onehot = tf.one_hot(int(label), config.NUM_CLASSES).numpy().astype(np.float32)

    feature = {
        'atomic_features_a': _bytes_feature(data_a['atomic_features'].astype(np.float32).tobytes()),
        'atomic_adj_a':      _bytes_feature(adj_a.tobytes()),
        'motif_ids_a':       _bytes_feature(data_a['motif_ids'].astype(np.int32).tobytes()),
        'motif_adj_a':       _bytes_feature(motif_adj_a.tobytes()),
        'global_features_a': _bytes_feature(global_a.tobytes()), 
        
        'atomic_features_b': _bytes_feature(data_b['atomic_features'].astype(np.float32).tobytes()),
        'atomic_adj_b':      _bytes_feature(adj_b.tobytes()),
        'motif_ids_b':       _bytes_feature(data_b['motif_ids'].astype(np.int32).tobytes()),
        'motif_adj_b':       _bytes_feature(motif_adj_b.tobytes()),
        'global_features_b': _bytes_feature(global_b.tobytes()), 
        
        'label':             _bytes_feature(label_onehot.tobytes())
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))

def convert_dataset(mode, csv_path, feat_dict, output_path):
    if not csv_path or not os.path.exists(csv_path):
        print(f"⏭️ 提示: {mode} 的 CSV 文件不存在，跳过转换。")
        return

    print(f"⚙️ 正在转换 {mode} 数据集: {os.path.basename(csv_path)} -> {os.path.basename(output_path)}")
    df = pd.read_csv(csv_path)
    written_count = 0
    
    with tf.io.TFRecordWriter(output_path) as writer:
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Writing {mode}"):
            da, db, lbl = str(row['drug_A']).strip(), str(row['drug_B']).strip(), row['DDI']
            try:
                example = serialize_example(feat_dict[da], feat_dict[db], lbl)
                writer.write(example.SerializeToString())
                written_count += 1
            except KeyError:
                pass # 忽略缺失特征的边
    print(f"✅ {mode} 转换完成，写入 {written_count} 条数据。\n")

def main():
    print("\n" + "📦"*15)
    print(f" 启动全能 TFRecord 生成器 (当前目标维度: {config.NUM_CLASSES})")
    print("📦"*15 + "\n")
    
    if not os.path.exists(config.PRECOMPUTED_FILE_PATH):
        print(f"❌ 致命错误: 找不到特征库 {config.PRECOMPUTED_FILE_PATH}")
        sys.exit(1)
        
    feat_dict = np.load(config.PRECOMPUTED_FILE_PATH, allow_pickle=True).item()
    print(f"📂 已加载双视图特征库，包含实体数: {len(feat_dict)}")
    
    # 确保输出目录存在
    os.makedirs(config.TFREC_DIR, exist_ok=True)
    
    # 智能执行转换 (有什么 CSV 就转什么)
    convert_dataset('Train', config.TRAIN_FILE, feat_dict, os.path.join(config.TFREC_DIR, 'train.tfrec'))
    convert_dataset('Valid', config.VALID_FILE, feat_dict, os.path.join(config.TFREC_DIR, 'valid.tfrec'))
    
    # 根据是否有 S2 测试集动态判断
    if hasattr(config, 'TEST_FILE') and config.TEST_FILE:
        convert_dataset('Test/S1', config.TEST_FILE, feat_dict, os.path.join(config.TFREC_DIR, 'test.tfrec'))
    
    if hasattr(config, 'TEST_S2_FILE') and config.TEST_S2_FILE:
        convert_dataset('Test_S2', config.TEST_S2_FILE, feat_dict, os.path.join(config.TFREC_DIR, 'test_s2.tfrec'))

    print("🎉 所有 TFRecord 数据流生成完毕！")

if __name__ == "__main__":
    main()