import numpy as np
import pandas as pd
from ydata_profiling import ProfileReport

df = pd.read_csv('heart_2022_no_nans.csv')
perfil = ProfileReport(df, title='Dados obtidos da base: Indicators of Heart Disease 2022 (no NaNs)')
perfil.to_file('perfil_dados_base.html')
