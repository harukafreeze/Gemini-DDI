# DrugBank-65 Configuration
# configs/deepddi_config.py
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# ===================================================================
# 1. DrugBank-65 专属数据路径 (物理隔离，确保 65 分类实验环境纯净)
# ===================================================================
RAW_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw', 'drugbank')

# 隔离运行目录：专门存放 65 分类的中间文件、TFRecord 和特征
DB65_EXP_DIR = os.path.join(PROJECT_ROOT, 'data', 'db65_exp')
PROCESSED_DATA_PATH = DB65_EXP_DIR
TFREC_DIR = os.path.join(DB65_EXP_DIR, 'tfrecords')

# 中间切分文件 (将由 run_deepddi_5fold_pipeline.py 动态管理)
TRAIN_FILE = os.path.join(DB65_EXP_DIR, 'train.csv')
VALID_FILE = os.path.join(DB65_EXP_DIR, 'valid.csv')
TEST_FILE  = os.path.join(DB65_EXP_DIR, 'test_s1.csv')
TEST_S2_FILE = os.path.join(DB65_EXP_DIR, 'test_s2.csv')

# 词表文件 (复用 ZhongDDI 的 RDKit 化学词表)
TOKEN_ID_FILE = os.path.join(PROJECT_ROOT, 'data', 'raw', 'token_id.json')

# 专属 65 分类的特征缓存文件 (务必运行 preprocess_deepddi_features.py 重新生成)
PRECOMPUTED_FILE_PATH = os.path.join(DB65_EXP_DIR, 'db65_features.npy')

# ===================================================================
# 2. 结果保存路径 (锁定 65 分类专属文件夹，防止权重覆写)
# ===================================================================
RESULTS_PATH = os.path.join(PROJECT_ROOT, 'results')
MODEL_SAVE_PATH = os.path.join(RESULTS_PATH, 'trained_models_db65') 
LOG_PATH = os.path.join(RESULTS_PATH, 'logs_db65')
FIGURE_PATH = os.path.join(RESULTS_PATH, 'figures_db65')

# ===================================================================
# 3. 核心模型参数 (与论文中 ZhongDDI 架构保持 100% 对齐)
# ===================================================================
ATOM_FEATURE_DIM = 77  
GLOBAL_FEATURE_DIM = 210
MOTIF_VOCAB_SIZE = 287 

# >>> [核心修改] 锁定为过滤后的 65 种样本量 >= 500 的主流机制 <<<
NUM_CLASSES = 65        
USE_FOCAL_LOSS = False # 使用高性能 AdamW 模式
NUM_ENCODER_LAYERS = 6    
D_MODEL = 320            
NUM_HEADS = 8              
D_FF = D_MODEL * 4         

# 黄金比例 Dropout，确保温启动 S0 阶段拟合上限最高
DROPOUT_RATE = 0.15       

# ===================================================================
# 4. 训练超参数 (A800 优化，温启动基石阶段)
# ===================================================================
EPOCHS = 80           # 80 轮对 65 分类非常稳
BATCH_SIZE = 256      # A800 80GB 最佳吞吐量
LEARNING_RATE = 1e-4       
ADAM_BETA_1 = 0.9
ADAM_BETA_2 = 0.98
ADAM_EPSILON = 1e-9
EARLY_STOPPING_PATIENCE = 20
ABLATION_MODE = 'full'

def ensure_directories_exist():
    """确保所有 65 分类实验所需的物理目录都已建立"""
    os.makedirs(DB65_EXP_DIR, exist_ok=True)
    os.makedirs(TFREC_DIR, exist_ok=True)
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    os.makedirs(LOG_PATH, exist_ok=True)
    os.makedirs(FIGURE_PATH, exist_ok=True)
    print(f"✅ DB65 实验目录检查完成: {DB65_EXP_DIR}")