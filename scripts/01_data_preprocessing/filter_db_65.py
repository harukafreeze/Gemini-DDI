# Filter classes with >= 500 samples
# scripts/filter_deepddi_65.py
import pandas as pd
import os

raw_path = "data/raw/drugbank/ddis.csv"
out_path = "data/raw/drugbank/drugbank_65.csv"

df = pd.read_csv(raw_path)
# 统计每个机制的频次
counts = df['type'].value_counts()
# 只保留样本数 >= 500 的类别 (这是学术界通行标准)
keep_types = counts[counts >= 500].index
df_65 = df[df['type'].isin(keep_types)].reset_index(drop=True)

# 重新对 Label 进行编码 (0-64)
df_65['type'] = pd.factorize(df_65['type'])[0]

df_65.to_csv(out_path, index=False)
print(f"✅ 过滤完成！剩余机制种类: {len(keep_types)}")
print(f"📊 总样本量: {len(df_65)}")
print(f"🚀 新数据集已生成: {out_path}")