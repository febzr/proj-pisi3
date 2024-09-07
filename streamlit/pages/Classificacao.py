import streamlit as st
import plotly.io as pio

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
st.subheader('Gráfico Shap do Melhor Modelo')
st.markdown('O gráfico abaixo mostra a importância de cada variável para o melhor modelo de classificação (Redes Neurais).')
st.image('graph/shap.png', width=670)