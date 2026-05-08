# configs/cold_config.py
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# --- Point to ISOLATED Directory ---
COLD_DIR = os.path.join(PROJECT_ROOT, 'data', 'cold_exp')
PROCESSED_DATA_PATH = COLD_DIR
# Train on Augmented Seen-Seen
TRAIN_FILE = os.path.join(COLD_DIR, 'train_aug.csv')
# Valid on Canonical Seen-Seen
VALID_FILE = os.path.join(COLD_DIR, 'valid.csv')
# Default Test to S1 (Can change to S2 later)
TEST_FILE = os.path.join(COLD_DIR, 'test_s1.csv') 

TOKEN_ID_FILE = os.path.join(PROJECT_ROOT, 'data', 'raw', 'token_id.json')
PRECOMPUTED_FILE_PATH = os.path.join(COLD_DIR, 'features.npy')

# TFRecord Output Dir (Isolated)
TFREC_DIR = os.path.join(COLD_DIR, 'tfrecords')

# Results
RESULTS_PATH = os.path.join(PROJECT_ROOT, 'results')
MODEL_SAVE_PATH = os.path.join(RESULTS_PATH, 'trained_models_cold') # Separate folder
LOG_PATH = os.path.join(RESULTS_PATH, 'logs_cold')
FIGURE_PATH = os.path.join(RESULTS_PATH, 'figures')

# Params (Same as V7.5)
ATOM_FEATURE_DIM = 77  
GLOBAL_FEATURE_DIM = 210
NUM_CLASSES = 4        
MOTIF_VOCAB_SIZE = 287 
NUM_ENCODER_LAYERS = 6    
D_MODEL = 320            
NUM_HEADS = 8              
D_FF = D_MODEL * 4         
DROPOUT_RATE = 0.3       
EPOCHS = 80           
BATCH_SIZE = 512      
LEARNING_RATE = 1e-4       
ADAM_BETA_1 = 0.9
ADAM_BETA_2 = 0.98
ADAM_EPSILON = 1e-9
EARLY_STOPPING_PATIENCE = 15
ABLATION_MODE = 'full'

def ensure_directories_exist():
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    os.makedirs(LOG_PATH, exist_ok=True)
    os.makedirs(FIGURE_PATH, exist_ok=True)
    os.makedirs(TFREC_DIR, exist_ok=True)