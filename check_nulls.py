import pandas as pd
import numpy as np

df = pd.read_parquet(r'E:\test\data\lifecycle\user_7d.parquet')

print(f'总用户数: {len(df)}')
print(f'\n字段空值统计(空值率>50%的字段):')
print('='*80)

for col in df.columns:
    if df[col].dtype == 'object':
        null_count = (df[col] == '').sum() + df[col].isnull().sum()
    else:
        null_count = df[col].isnull().sum() + (df[col] == 0).sum()
    
    null_pct = (null_count / len(df) * 100)
    if null_pct > 50:
        print(f'  {col}: 空值/零值 {null_count} ({null_pct:.1f}%)')

print(f'\n\n数值字段统计(前10个):')
print('='*80)
numeric_cols = df.select_dtypes(include=[np.number]).columns[:10]
for col in numeric_cols:
    print(f'  {col}: min={df[col].min():.2f}, max={df[col].max():.2f}, mean={df[col].mean():.2f}')

print(f'\n\n字符串字段非空统计:')
print('='*80)
for col in df.select_dtypes(include=['object']).columns:
    non_empty = (df[col] != '').sum()
    print(f'  {col}: {non_empty} ({non_empty/len(df)*100:.1f}%)')
