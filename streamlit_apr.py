import streamlit as st
import plotly.express as px
import pandas as pd
import pipeline
import clusterização
import classificacao

data = pipeline.pipelines(pd.read_parquet("heart_2022_no_nans.parquet"))
X = data.create()
X = X.drop(columns=["State"])

corr = px.imshow(
    X.corr(numeric_only=True), text_auto=True, color_continuous_scale="RdBu_r"
)
corr.update_xaxes(showticklabels=False)
corr.update_yaxes(tickfont=dict(size=7))
corr.update_layout(
    title="Heatmap de Correlação",
)

cluster = clusterização.clusters(X, 5)

clas = classificacao.classificar(X, 5, "knn")

rdfor = classificacao.classificar(X, 100, "florest")

# inicia a parte do streamlit

st.title("Heart Disease Indicators - 2022")

st.plotly_chart(corr)

st.header("Clusterização:")
st.plotly_chart(cluster.grafico_cluster(X))

st.header("Classificação:")
st.write("Cuidado ao ativar os pontos, pode resultar em travamentos!!")
st.write(
    "Recomendamos que seja ativado apenas para visualizar os pontos deste gráfico."
)
st.plotly_chart(clas.knn_fronteira_grafico())

st.plotly_chart(rdfor.randomflorest_importancia_feature_grafico())
