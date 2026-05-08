# Generate dual-view features
import os
import sys
import numpy as np
import tensorflow as tf
import h5py

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from prism.model import PRISM_DDI
from prism.dataloader import PrismDdiDataloader
import configs.cold_config as config

def smart_weight_injector(model, h5_path):
    """
    自研补丁：扫描H5文件，按形状(Shape)匹配强行注入权重，彻底无视层名。
    """
    print(f"📦 启动智能权重注入: {os.path.basename(h5_path)}")
    with h5py.File(h5_path, 'r') as f:
        # 提取H5中所有带权重的组
        h5_weight_groups = [f[k] for k in f.keys() if 'weight_names' in f[k].attrs]
        
        # 遍历模型中所有有参数的层
        match_count = 0
        for layer in model.layers:
            if not layer.weights: continue
            
            # 获取当前层需要的权重形状列表
            target_shapes = [w.shape.as_list() for w in layer.weights]
            
            # 在H5中寻找形状完全匹配的组
            for g in h5_weight_groups:
                w_names = [n.decode('utf8') if isinstance(n, bytes) else n for n in g.attrs['weight_names']]
                source_weights = [g[w][:] for w in w_names]
                source_shapes = [list(w.shape) for w in source_weights]
                
                if source_shapes == target_shapes:
                    layer.set_weights(source_weights)
                    match_count += 1
                    # print(f"  ✅ 注入成功: {layer.name}")
                    break
        print(f"🚀 强力注入完成！共成功对接 {match_count} 个参数层。")

def extract(model_weight_path, save_prefix, sample_x):
    print(f"\n" + "="*50)
    print(f"🚀 正在提取特征: {save_prefix}")
    
    # 1. 重建模型
    model_core = PRISM_DDI(config, drop_rate=0.0)
    _ = model_core(sample_x, training=False) # 必须先初始化形状

    # 2. 加载权重
    if save_prefix == "ours":
        # 先用 V7.5 垫底 (保证结构分支不出错)
        v75_path = "/root/PDTT/results/trained_models_cold/PRISM_V7.5_Chemist_20260130-024240_best.h5"
        model_core.load_weights(v75_path, by_name=True, skip_mismatch=True)
        print("✅ 结构底座 V7.5 加载完毕")
        
        # 使用智能注入器强行塞入 V10.1 的蒸馏逻辑
        smart_weight_injector(model_core, model_weight_path)
    else:
        # Base 正常加载
        model_core.load_weights(model_weight_path, by_name=True, skip_mismatch=True)
        print("✅ Base 权重加载完毕")

    # 3. 截取特征
    feature_model = tf.keras.Model(inputs=model_core.input, outputs=model_core.get_layer('head_d2').output)

    # 4. 提取数据
    dl = PrismDdiDataloader(config)
    s2_path = os.path.join(config.TFREC_DIR, "test_s2.tfrec")
    ds = tf.data.TFRecordDataset(s2_path).map(dl._parse_function).padded_batch(512)

    features_list, labels_list = [], []
    for x, y in ds:
        feats = feature_model(x, training=False).numpy()
        # 处理可能的极个别 NaN
        feats = np.nan_to_num(feats, nan=0.0)
        features_list.append(feats)
        labels_list.append(y.numpy())

    X = np.concatenate(features_list, axis=0)
    Y = np.argmax(np.concatenate(labels_list, axis=0), axis=1)
    
    max_val = np.max(np.abs(X))
    print(f"📊 {save_prefix} 提取战报: 最大值={max_val:.4f}, 样本数={len(X)}")
    
    if max_val < 1e-7:
        print("🚨 警告：数据依然为 0，请检查 V10.1 文件是否保存正确。")

    np.save(f'{save_prefix}_X.npy', X)
    np.save(f'{save_prefix}_Y.npy', Y)

if __name__ == "__main__":
    v75 = "/root/PDTT/results/trained_models_cold/PRISM_V7.5_Chemist_20260130-024240_best.h5"
    v10 = "/root/PDTT/results/trained_models_cold/PRISM_V10.1_Distill_Final.h5"
    
    # 拿一笔真实数据初始化
    dl_init = PrismDdiDataloader(config)
    s2_ds = tf.data.TFRecordDataset(os.path.join(config.TFREC_DIR, "test_s2.tfrec")).map(dl_init._parse_function).padded_batch(2)
    sample_x, _ = next(iter(s2_ds))

    extract(v75, "base", sample_x)
    extract(v10, "ours", sample_x)