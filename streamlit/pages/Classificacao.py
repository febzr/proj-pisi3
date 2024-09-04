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

rdfor = pio.read_json('graph/comparacao_random_forest.json')
knn = pio.read_json('graph/comparacao_knn.json')
catboost = pio.read_json('graph/comparacao_catboost.json')
rn = pio.read_json('graph/comparaco_redes_neurais.json')
reglog = pio.read_json('graph/comparacao_regressao_logistica.json')

frdfor = pio.read_json('graph/importancia_features_rf.json')
freglog = pio.read_json('graph/importancia_features_reglog.json')
fcb = pio.read_json('graph/importancia_features_cb.json')

gtreino = pio.read_json('graph/graf_grid_treino.json')
gteste = pio.read_json('graph/graf_grid_teste.json')

#PARTE DO STREAMLIT
st.set_page_config(page_title='Classificação', page_icon=':heart:', layout='centered', initial_sidebar_state='auto')

st.title("CLASSIFICAÇÃO")

gtreino.update_layout(width=670)
st.plotly_chart(gtreino)
gteste.update_layout(width=670)
st.plotly_chart(gteste)

#RDForest
st.subheader('Feature Importances')
dist1 = st.expander('Random Forest')

with dist1:
    frdfor.update_layout(width=670)
    st.plotly_chart(frdfor)

#CatBoost
dist3 = st.expander('CatBoost')

with dist3:
    fcb.update_layout(width=670)
    st.plotly_chart(fcb)

#Regressão Logística
dist5 = st.expander('Regressão Logística')

with dist5:
    freglog.update_layout(width=670)
    st.plotly_chart(freglog)
