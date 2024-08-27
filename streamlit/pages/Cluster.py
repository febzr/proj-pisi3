import streamlit as st
import plotly.express as px
import pipeline
import pandas as pd
from sklearn.cluster import KMeans
import plotly.graph_objects as go
from sklearn.decomposition import PCA
import plotly.express as px
from sklearn.cluster import KMeans
import plotly.graph_objects as go
import plotly.io as pio
from imblearn.under_sampling import RandomUnderSampler

data_cru = pd.read_parquet('heart_2022_no_nans.parquet')

data = pipeline.pipelines(data_cru)
X = data.create()

data_cru = data_cru.drop(columns=['State'])

dados_sem_estados = X.drop(columns=['State'])

dados_em_matriz=dados_sem_estados.values

x = dados_sem_estados.drop('HadHeartAttack', axis=1)
y = dados_sem_estados['HadHeartAttack']

rus = RandomUnderSampler(random_state=42)

cotovelo = pio.read_json('graph/cotovelo.json')

# PARTE DO PLOTLY

st.title("CLUSTERIZAÇÃO")
st.text("Gráfico do método do cotovelo para auxiliar na escolha do número de clusters")
st.plotly_chart(cotovelo)
