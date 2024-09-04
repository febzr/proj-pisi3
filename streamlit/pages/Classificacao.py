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

dist6 = st.expander('Grid Search')

with dist6:
    gtreino.update_layout(width=670)
    st.plotly_chart(gtreino)
    gteste.update_layout(width=670)
    st.plotly_chart(gteste)

#RDForest
dist1 = st.expander('Random Forest')

with dist1:
    rdfor.update_layout(width=670)
    st.plotly_chart(rdfor)
    frdfor.update_layout(width=670)
    st.plotly_chart(frdfor)

#KNN
dist2 = st.expander('KNN')

with dist2:
    knn.update_layout(width=670)
    st.plotly_chart(knn)

#CatBoost
dist3 = st.expander('CatBoost')

with dist3:
    catboost.update_layout(width=670)
    st.plotly_chart(catboost)
    fcb.update_layout(width=670)
    st.plotly_chart(fcb)

#Rede Neural
dist4 = st.expander('Rede Neural')

with dist4:
    rn.update_layout(width=670)
    st.plotly_chart(rn)

#Regressão Logística
dist5 = st.expander('Regressão Logística')

with dist5:
    reglog.update_layout(width=670)
    st.plotly_chart(reglog)
    freglog.update_layout(width=670)
    st.plotly_chart(freglog)
