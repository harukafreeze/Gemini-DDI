# Drug-wise strict isolation (S0/S1/S2)
# scripts/01_data_preprocessing/split_inductive.py
import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

# 动态路径处理
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.append(project_root)

# 加载配置
try:
    import configs.db65_inductive_config as config
except:
    import configs.zhongddi_warm_config as config

def main():
    print(f"🛡️ 正在执行【冷启动严格物理隔离】5-Fold 切分...")
    
    # 1. 读取原始 CSV (自动适配列名)
    raw_csv = os.path.join(project_root, "data/raw/drugbank/drugbank_65.csv")
    df = pd.read_csv(raw_csv)
    
    # 自动对齐列名: 确保输出统一为 drug_A, drug_B, DDI
    col_map = {'d1': 'drug_A', 'd2': 'drug_B', 'type': 'DDI'}
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    # 2. 提取唯一的药物实体集
    unique_drugs = np.array(sorted(list(set(df['drug_A'].unique()) | set(df['drug_B'].unique()))))
    print(f"📊 共有 {len(unique_drugs)} 种药物参与冷启动隔离。")

    # 3. 执行 5-Fold Drug-wise 切分
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    for fold_idx, (train_drug_idx, unseen_drug_idx) in enumerate(kf.split(unique_drugs)):
        print(f"⚙️ 正在生成 Fold {fold_idx}...")
        seen_drugs = set(unique_drugs[train_drug_idx])
        unseen_drugs = set(unique_drugs[unseen_drug_idx])

        # 定义隔离逻辑
        s0_pairs = df[df['drug_A'].isin(seen_drugs) & df['drug_B'].isin(seen_drugs)]
        s2_pairs = df[df['drug_A'].isin(unseen_drugs) & df['drug_B'].isin(unseen_drugs)]
        s1_pairs = df[~(df.index.isin(s0_pairs.index) | df.index.isin(s2_pairs.index))]

        # 4. 建立文件夹
        fold_dir = os.path.join(config.INDUCTIVE_EXP_DIR, 'folds', f'fold_{fold_idx}')
        os.makedirs(fold_dir, exist_ok=True)

        # 5. S0 进一步切分 Train(90%) / Valid(10%)
        s0_pairs = s0_pairs.sample(frac=1, random_state=42)
        split_pt = int(len(s0_pairs) * 0.9)
        
        # 保存 CSV
        s0_pairs.iloc[:split_pt].to_csv(os.path.join(fold_dir, "train_raw.csv"), index=False)
        s0_pairs.iloc[split_pt:].to_csv(os.path.join(fold_dir, "valid.csv"), index=False)
        s1_pairs.to_csv(os.path.join(fold_dir, "test_s1.csv"), index=False)
        s2_pairs.to_csv(os.path.join(fold_dir, "test_s2.csv"), index=False)

    print(f"✅ 5-Fold CSV 切分完成！存放于: {config.INDUCTIVE_EXP_DIR}/folds/")

if __name__ == "__main__":
    main()