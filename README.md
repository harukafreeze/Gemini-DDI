
# Gemini-DDI: A Dual-view Framework for Drug-Drug Interaction Prediction

> **Official implementation for the paper:**  
> **"Gemini-DDI: A Dual-view Framework for Drug-Drug Interaction Prediction"**  

---

## Installation

**1. Clone the repository**
```bash
git clone https://github.com/harukafreeze/Gemini-DDI.git
cd Gemini-DDI
```

**2. Setup Environment**
```bash
conda create -n gemini_ddi python=3.8
conda activate gemini_ddi
pip install -r requirements.txt
# Requirements: tensorflow==2.13.0, rdkit-pypi, scikit-learn, pandas, matplotlib
```

## Reproduction Guide

Follow these stages to reproduce the results for both **ZhongDDI** (Macro-level) and **DrugBank-65** (Mechanism-specific) benchmarks.

### Stage 1: Data Preparation & Preprocessing
Regardless of the task, you must first initialize the dual-view features:

```bash
# 1. Extract 210-D descriptors and Motif graphs (Common for all tasks)
python scripts/01_data_preprocessing/extract_features.py

# 2. For DrugBank-65 task: Filter rare mechanisms (< 500 samples)
python scripts/01_data_preprocessing/filter_db_65.py
```

### Stage 2: Macro-Level ADME Task (ZhongDDI)

```bash
# 1. Execute the 5-fold cross-validation for Warm-start (S0)
python scripts/02_zhongddi_task/train_s0_zhong.py

# 2. Run the logic-alignment distillation for Cold-start (S2)
python scripts/02_zhongddi_task/distill_s2_zhong.py

# 3. (Optional) Run all ablation studies
python scripts/02_zhongddi_task/run_ablations_zhong.py --mode no_distill
python scripts/02_zhongddi_task/run_ablations_zhong.py --mode no_se
```

### Stage 3: Fine-Grained Mechanism Task (DrugBank-65)

#### A. Data Pipeline (Inductive Split + 5x Augmentation)
```bash
# 1. Generate strictly isolated drug-wise splits (Folds 0-4)
python scripts/01_data_preprocessing/split_inductive.py

# 2. Generate 5x SMILES variants for training sets
python scripts/01_data_preprocessing/augment_data.py

# 3. Serialize all folds into TFRecords
python scripts/01_data_preprocessing/create_tfrecords.py
```

#### B. Training & Distillation
```bash
# 1. Reproduce S0 Warm-start results (Table 4, 98.13% Acc)
python scripts/03_drugbank65_task/train_s0_db65.py

# 2. Reproduce S2 Cold-start results:
# First, train Inductive Teachers (Clean models for zero-shot guide)
python scripts/03_drugbank65_task/train_inductive_teacher.py
# Then, run Pro-level Consistency Distillation with Radar monitoring
python scripts/03_drugbank65_task/distill_s2_db65.py
```

---

