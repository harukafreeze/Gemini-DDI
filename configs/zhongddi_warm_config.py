# ZhongDDI Configuration
# configs/default_config.py (V7.5 Final)
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# 1. Paths (Must point to V6 Augmented Data)
RAW_DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'raw')
TRAIN_FILE = os.path.join(RAW_DATA_PATH, 'tr_dataset_aug_v6.csv') 
VALID_FILE = os.path.join(RAW_DATA_PATH, 'val_dataset.csv')
TEST_FILE = os.path.join(RAW_DATA_PATH, 'tst_dataset.csv')
TOKEN_ID_FILE = os.path.join(RAW_DATA_PATH, 'token_id.json')

PROCESSED_DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed')
PRECOMPUTED_FILE_PATH = os.path.join(PROCESSED_DATA_PATH, 'augmented_features_v6.npy')

RESULTS_PATH = os.path.join(PROJECT_ROOT, 'results')
MODEL_SAVE_PATH = os.path.join(RESULTS_PATH, 'trained_models')
LOG_PATH = os.path.join(RESULTS_PATH, 'logs')
FIGURE_PATH = os.path.join(RESULTS_PATH, 'figures')

# 2. Data Params
ATOM_FEATURE_DIM = 77  
GLOBAL_FEATURE_DIM = 210 # RDKit descriptors count

# 3. Model Params (Expanded for V7.5)
NUM_CLASSES = 4
USE_FOCAL_LOSS = True  # 开启 Focal Loss 模式        
MOTIF_VOCAB_SIZE = 287 
NUM_ENCODER_LAYERS = 6    
D_MODEL = 320            # [UPGRADE] 256 -> 320
NUM_HEADS = 8              
D_FF = D_MODEL * 4         
DROPOUT_RATE = 0.3       

# 4. Training Params
EPOCHS = 100               
BATCH_SIZE = 512          
LEARNING_RATE = 1e-4       
ADAM_BETA_1 = 0.9
ADAM_BETA_2 = 0.98
ADAM_EPSILON = 1e-9
EARLY_STOPPING_PATIENCE = 15 

def ensure_directories_exist():
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    os.makedirs(LOG_PATH, exist_ok=True)
    os.makedirs(FIGURE_PATH, exist_ok=True)
    print("Project directories are ensured to exist.")