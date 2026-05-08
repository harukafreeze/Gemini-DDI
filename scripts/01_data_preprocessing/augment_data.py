# 5x random SMILES augmentation
# scripts/01_data_preprocessing/augment_data.py
import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from rdkit import Chem, RDLogger

# 关闭 RDKit 底层警告，保持终端清爽
RDLogger.DisableLog('rdApp.*')

# --- 动态寻址：确保无论从哪里执行，都能找到项目根目录 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
if project_root not in sys.path: 
    sys.path.insert(0, project_root)

# 动态加载配置 (优先尝试加载 DB65，其次默认配置)
try:
    import configs.db65_inductive_config as config
except ImportError:
    import configs.zhongddi_warm_config as config

from prism.utils.chemical_tools import get_atomic_graph_from_smiles, get_motif_graph_from_smiles, get_global_features, MolTokenizer

AUGMENT_TIMES = 4 # 原样本 1 + 变体 4 = 5倍扩充

def get_random_smiles(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return None
        return Chem.MolToSmiles(mol, doRandom=True, canonical=False)
    except: 
        return None

def extract_features(smiles, tokenizer):
    a_feat, a_adj = get_atomic_graph_from_smiles(smiles)
    m_ids, m_adj = get_motif_graph_from_smiles(smiles, tokenizer)
    g_feat = get_global_features(smiles)
    return {
        'atomic_features': a_feat.astype(np.float32), 
        'atomic_adj': a_adj.astype(np.int32),
        'motif_ids': m_ids.astype(np.int32), 
        'motif_adj': m_adj.astype(np.float32),
        'global_features': g_feat.astype(np.float32)
    }

def main():
    print("\n" + "🧬"*15)
    print(" 启动全能版数据增强 (完美兼容 DrugID 与 SMILES)")
    print("🧬"*15 + "\n")
    
    # 锁定物理隔离的流水线临时文件
    input_csv = os.path.join(project_root, "data/raw/temp_fold_train_raw.csv")
    output_csv = os.path.join(project_root, "data/raw/temp_fold_train_aug.csv")
    smiles_csv = os.path.join(project_root, "data/raw/drugbank/drug_smiles.csv")
    
    if not os.path.exists(input_csv):
        print(f"❌ 找不到训练文件: {input_csv}。请通过流水线主程序启动。")
        sys.exit(1)
        
    # --- 1. 尝试加载 DrugID 映射表 (针对 DrugBank 任务) ---
    id2smiles = {}
    if os.path.exists(smiles_csv):
        df_smiles = pd.read_csv(smiles_csv)
        # 兼容以 Tab 或 逗号 分隔的不同格式
        if 'smiles' not in df_smiles.columns and '\t' in df_smiles.columns[0]:
            df_smiles = pd.read_csv(smiles_csv, sep='\t')
        if 'drug_id' in df_smiles.columns and 'smiles' in df_smiles.columns:
            id2smiles = dict(zip(df_smiles['drug_id'].astype(str).str.strip(), df_smiles['smiles']))
    
    # --- 2. 尝试加载历史特征缓存 (极大加速计算) ---
    feat_dict = {}
    if os.path.exists(config.PRECOMPUTED_FILE_PATH):
        print(f"📂 发现历史特征库，启动增量计算模式...")
        feat_dict = np.load(config.PRECOMPUTED_FILE_PATH, allow_pickle=True).item()
        
    tokenizer = MolTokenizer(config.TOKEN_ID_FILE)
    
    df_train = pd.read_csv(input_csv)
    unique_drugs = set(df_train['drug_A'].astype(str).unique()) | set(df_train['drug_B'].astype(str).unique())
    drug_variants = {}
    
    print(f"⚙️ 正在为 {len(unique_drugs)} 种实体生成随机拓扑变体...")
    for drug_id_str in tqdm(unique_drugs):
        drug_id_str = drug_id_str.strip()
        variants = [drug_id_str]
        
        # 【核心：智能嗅探】判断输入的是 ID 还是 SMILES
        if drug_id_str in id2smiles:
            canonical_smiles = id2smiles[drug_id_str] # 是 DrugBank ID，查表
        else:
            if Chem.MolFromSmiles(drug_id_str) is not None:
                canonical_smiles = drug_id_str # 是合法的化学式，直接用 (ZhongDDI模式)
            else:
                canonical_smiles = None # 无效分子
        
        if canonical_smiles:
            # 确保基石节点在字典中
            if drug_id_str not in feat_dict:
                try: feat_dict[drug_id_str] = extract_features(canonical_smiles, tokenizer)
                except: pass

            # 拓扑扭曲引擎启动
            for i in range(AUGMENT_TIMES):
                rand_s = get_random_smiles(canonical_smiles)
                if rand_s:
                    aug_id = f"{drug_id_str}_aug_{i}"  # 为异构体签发新 ID
                    if aug_id not in feat_dict:
                        try:
                            feat_dict[aug_id] = extract_features(rand_s, tokenizer)
                            variants.append(aug_id)
                        except: pass
                    else:
                        variants.append(aug_id)
                        
        drug_variants[drug_id_str] = list(set(variants))

    # --- 3. 存档特征库 ---
    os.makedirs(os.path.dirname(config.PRECOMPUTED_FILE_PATH), exist_ok=True)
    np.save(config.PRECOMPUTED_FILE_PATH, feat_dict)
    
    # --- 4. 扩充 CSV 配对关系 ---
    print("🚀 正在几何级扩充训练配对图谱...")
    new_rows =[]
    for _, row in tqdm(df_train.iterrows(), total=len(df_train)):
        da = str(row['drug_A']).strip()
        db = str(row['drug_B']).strip()
        lbl = row['DDI']
        
        vars_a = drug_variants.get(da, [da])
        vars_b = drug_variants.get(db, [db])
        
        new_rows.append([da, db, lbl]) # 保留原始真实配对
        
        # 随机抽取变体组合
        for _ in range(AUGMENT_TIMES):
            va = np.random.choice(vars_a)
            vb = np.random.choice(vars_b)
            if va == da and vb == db: continue # 防重复
            new_rows.append([va, vb, lbl])
            
    df_aug = pd.DataFrame(new_rows, columns=['drug_A', 'drug_B', 'DDI'])
    df_aug.drop_duplicates(inplace=True)
    df_aug.to_csv(output_csv, index=False)
    
    print(f"🎉 训练集扩充完成: 样本量从 {len(df_train)} 跃升至 {len(df_aug)} 条！")
    print(f"💾 特征库已无损保存至: {config.PRECOMPUTED_FILE_PATH}\n")

if __name__ == "__main__":
    main()