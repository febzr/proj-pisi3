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


#PARTE DO STREAMLIT
st.title("CLASSIFICAÇÃO")
st.text("Gráfico geral do Grid Search")
st.text("Aqui você pode comparar os resultados dos modelos de classificação")
#st.plotly_chart(cotovelo)

#RDForest
dist1 = st.expander('Random Forest')

with dist1:
    rdfor.update_layout(width=670)
    st.plotly_chart(rdfor)

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
