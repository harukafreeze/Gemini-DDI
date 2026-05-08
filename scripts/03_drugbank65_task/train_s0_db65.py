import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import shutil

# ===================================================================
# [配置区] DrugBank-65 机制分类任务专属流水线
# ===================================================================
RAW_DIR = "data/raw/drugbank"
# 自动读取 filter_deepddi_65.py 生成的过滤后数据
SOURCE_FILES = ["drugbank_65.csv"] 

# 结果保存根目录 (与 configs/deepddi_config.py 保持绝对对齐)
RESULTS_BASE = "results/trained_models_db65_clean"
LOG_BASE = "results/logs_db65"

# 环境执行指令 (A800 专用环境)
CMD_PREFIX = "LD_LIBRARY_PATH=/root/anaconda3/envs/prism_gpu/lib python"

def run_cmd(cmd):
    print(f"\n🚀 执行指令: {cmd}")
    ret = os.system(cmd)
    if ret != 0: 
        raise RuntimeError(f"❌ 指令执行失败: {cmd}")

def parse_all_metrics(log_path):
    """解析日志中的全指标数据 (兼容 Macro 和新加入的 Micro 指标)"""
    metrics = {'Acc': 0.0, 'Macro-AUC': 0.0, 'Macro-AUPR': 0.0, 'Micro-AUC': 0.0, 'Micro-AUPR': 0.0, 'F1': 0.0}
    try:
        with open(log_path, "r") as f:
            for line in f:
                if "Accuracy :" in line: metrics['Acc'] = float(line.split(":")[1].strip())
                if "Macro-AUC" in line:   metrics['Macro-AUC'] = float(line.split(":")[1].strip())
                if "Macro-AUPR" in line:  metrics['Macro-AUPR'] = float(line.split(":")[1].strip())
                if "Micro-AUC" in line:   metrics['Micro-AUC'] = float(line.split(":")[1].strip().split()[0])
                if "Micro-AUPR" in line:  metrics['Micro-AUPR'] = float(line.split(":")[1].strip().split()[0])
                if "F1-Score :" in line:  metrics['F1'] = float(line.split(":")[1].strip())
    except Exception as e:
        print(f"⚠️ 指标解析异常: {e}")
    return metrics

def main():
    print("\n" + "🛡️"*30)
    print("  [PRISM-DDI] 启动 DrugBank-65 机制分类五折流水线")
    print("  任务状态：高准度温启动基石训练")
    print("🛡️"*30 + "\n")
    
    # 0. 准备环境
    os.makedirs(RESULTS_BASE, exist_ok=True)
    os.makedirs(LOG_BASE, exist_ok=True)

    # 1. 载入 DrugBank-65 数据
    data_path = os.path.join(RAW_DIR, SOURCE_FILES[0])
    if not os.path.exists(data_path):
        print(f"❌ 找不到 65 分类数据 {data_path}！请先运行 filter_deepddi_65.py")
        sys.exit(1)
        
    print(f"📥 正在读取数据: {data_path}")
    df = pd.read_csv(data_path)
    
    # 统一列名为模型所需的 drug_A, drug_B, DDI
    # 假设 filter_deepddi_65.py 输出的列名是 d1, d2, type
    if 'd1' in df.columns:
        df = df.rename(columns={'d1': 'drug_A', 'd2': 'drug_B', 'type': 'DDI'})
    
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"📊 65类主流机制样本总量: {len(df)}")

    # 2. 五折循环
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    summary_file = os.path.join(LOG_BASE, "5fold_summary_db65.csv")
    with open(summary_file, "w") as f:
        f.write("Fold,Accuracy,Macro-AUC,Micro-AUC,Macro-AUPR,Micro-AUPR,F1\n")

    all_fold_metrics = []

    for fold_idx, (train_full_idx, test_idx) in enumerate(kf.split(df)):
        fold_num = fold_idx + 1
        print(f"\n\n" + "🌀"*30)
        print(f"正在处理第 {fold_num} 折 / 共 5 折 (DrugBank-65)")
        print("🌀"*30 + "\n")

        # A. 切分数据
        df_test = df.iloc[test_idx]
        df_train_all = df.iloc[train_full_idx]
        
        split_n = int(len(df_train_all) * 0.9)
        df_train = df_train_all.iloc[:split_n]
        df_valid = df_train_all.iloc[split_n:]
        
        # B. 保存临时 CSV (供 augment_data.py 和 TFRecord 脚本读取)
        df_train.to_csv("data/raw/temp_fold_train_raw.csv", index=False)
        df_valid.to_csv("data/raw/temp_fold_val.csv", index=False)
        df_test.to_csv("data/raw/temp_fold_test.csv", index=False)

        # C. 运行数据增强 (5x 满血版)
        run_cmd(f"{CMD_PREFIX} scripts/augment_data.py")

        # D. 运行 TFRecord 生成 (清理 db65 专属目录)
        os.system("rm -rf data/db65_exp/tfrecords/*")
        run_cmd(f"{CMD_PREFIX} scripts/create_tfrecords.py")

        # E. 训练 (CCE + 0.15 Dropout 黄金配置)
        run_cmd(f"{CMD_PREFIX} scripts/train.py")

        # F. 全指标评估 (包含 Micro 和 Top-3)
        log_path = os.path.join(LOG_BASE, f"fold_{fold_num}_eval.txt")
        run_cmd(f"{CMD_PREFIX} scripts/evaluate_final_deep.py > {log_path}")
        
        # G. 解析并存档结果
        res = parse_all_metrics(log_path)
        all_fold_metrics.append(res)
        print(f"✅ 第 {fold_num} 折战报:")
        print(f"   Acc: {res['Acc']:.4f} | Macro-AUC: {res['Macro-AUC']:.4f} | Micro-AUC: {res['Micro-AUC']:.4f}")

        # H. 存档当前折模型 (防止被下一折覆盖)
        import glob
        # 寻找训练脚本刚生成的 best 权重
        model_list = glob.glob("results/trained_models/PRISM_V7.5_Chemist_*_best.h5")
        if model_list:
            latest_model = max(model_list, key=os.path.getmtime)
            archive_path = os.path.join(RESULTS_BASE, f"DB65_Warm_Fold_{fold_idx}.h5")
            shutil.copy2(latest_model, archive_path) # 使用 copy2 保留元数据
            print(f"💾 65类基石权重已存档: {archive_path}")

        with open(summary_file, "a") as f:
            f.write(f"{fold_num},{res['Acc']},{res['Macro-AUC']},{res['Micro-AUC']},{res['Macro-AUPR']},{res['Micro-AUPR']},{res['F1']}\n")

    # 3. 最终汇总
    print("\n\n" + "🏆"*15)
    print("  DrugBank-65 五折温启动最终汇总")
    print("🏆"*15)
    
    for m_key in ['Acc', 'Macro-AUC', 'Micro-AUC', 'Macro-AUPR', 'Micro-AUPR', 'F1']:
        vals = [item[m_key] for item in all_fold_metrics]
        print(f"{m_key.ljust(12)} : {np.mean(vals):.4f} +/- {np.std(vals):.4f}")

if __name__ == "__main__":
    main()