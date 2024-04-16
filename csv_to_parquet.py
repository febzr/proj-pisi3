import pandas as pd

df = pd.read_csv('heart_2020_cleaned.csv')
df.to_parquet('heart_2020_cleaned.parquet')