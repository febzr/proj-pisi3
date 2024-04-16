import pandas as pd

df = pd.read_csv('proj-pisi3\heart_2022_no_nans.csv')
df.to_parquet('heart_2022_no_nans.parquet')