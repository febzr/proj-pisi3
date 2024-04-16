import pandas as pd
from ydata_profiling import ProfileReport

df = pd.read_parquet('heart_2020_cleaned.parquet')
perfil = ProfileReport(df, title='Dados obtidos da base: Indicators of Heart Disease 2020 (cleaned)')
perfil.to_file('perfil_dados_base.html')