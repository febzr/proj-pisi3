import pipeline
import pandas as pd
import clusterização
import classificacao
from ydata_profiling import ProfileReport

data = pipeline.pipelines(pd.read_parquet('heart_2022_no_nans.parquet'))
X = data.create()

perfil = ProfileReport(X, title='Dados normalizados da base de dados: Indicators of Heart Disease 2022 (no NaNs)')
perfil.to_file('perfil_dados_base_normalizado.html')