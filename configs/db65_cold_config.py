# configs/db65_inductive_config.py
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# ===================================================================
# 1. 绝对隔离的冷启动 (S2) 数据区
# ===================================================================
INDUCTIVE_EXP_DIR = os.path.join(PROJECT_ROOT, 'data', 'db65_inductive_exp')
PROCESSED_DATA_PATH = INDUCTIVE_EXP_DIR  # 让 dataloader 乖乖来这里读数据
TFREC_DIR = os.path.join(INDUCTIVE_EXP_DIR, 'tfrecords')

# 复用全局特征库 (这个不需要隔离，因为特征是不变的)
PRECOMPUTED_FILE_PATH = os.path.join(PROJECT_ROOT, 'data', 'db65_exp', 'db65_features.npy')
TOKEN_ID_FILE = os.path.join(PROJECT_ROOT, 'data', 'raw', 'token_id.json')

# ===================================================================
# 2. 绝对隔离的结果区 (Teacher 和 S2 权重)
# ===================================================================
RESULTS_PATH = os.path.join(PROJECT_ROOT, 'results')
MODEL_SAVE_PATH = os.path.join(RESULTS_PATH, 'trained_models_db65_inductive')
LOG_PATH = os.path.join(RESULTS_PATH, 'logs_db65_inductive')

os.makedirs(TFREC_DIR, exist_ok=True)
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
os.makedirs(LOG_PATH, exist_ok=True)

# ===================================================================
# 3. 模型与训练核心参数 (对齐温启动)
# ===================================================================
NUM_CLASSES = 65        
ATOM_FEATURE_DIM = 77  
GLOBAL_FEATURE_DIM = 210
MOTIF_VOCAB_SIZE = 287 
NUM_ENCODER_LAYERS = 6    
D_MODEL = 320            
NUM_HEADS = 8              
D_FF = D_MODEL * 4         
DROPOUT_RATE = 0.15  # Teacher 阶段用 0.15，学生蒸馏时代码会强制设为 0.8
BATCH_SIZE = 128     # 蒸馏防爆显存专用