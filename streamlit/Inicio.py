import streamlit as st
import plotly.express as px
import pandas as pd
import pipeline_gen
import classificacao

df = pd.read_parquet('heart_2022_no_nans.parquet')
df2 = df
df2.drop(columns=['State'], inplace=True)
data = pipeline_gen.pipelines(df2)
dpip = data.create()
print(dpip)

st.title('Heart Disease Indicators, 2022')
st.write('Interaja com os dados da base e veja os gráficos gerados.')

correlacao = st.container()

with correlacao:
    st.header('Matriz de correlação')
    fig = px.imshow(dpip.corr())
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(tickfont=dict(size=7))
    st.plotly_chart(fig)

distribuicao = st.container(border=True)
coluna1, coluna2 = st.columns(2)

with distribuicao:
    with coluna1:
        st.header('Descrição dos dados')
        st.write(df.describe())
    
    with coluna2:
        st.header('Dicionário dos dados')
        st.write(df.dtypes)

dist2 = st.container(border=True)

with dist2:
    st.header('Distribuição das variáveis')
    dist_sel = st.selectbox('Selecione a variável:', df.columns)
    if dist_sel:
        fig = px.histogram(df, x=dist_sel, nbins=20, width=650)
        st.plotly_chart(fig)

disp = st.form(border=True, key='disp')

with disp:
    st.header('Gráficos de Violino')
    dispx = st.selectbox('Selecione o eixo x:', df.columns)
    dispy = st.selectbox('Selecione o eixo y:', df.columns)

    but = disp.form_submit_button('Gerar gráfico')
    if but:

        fig = px.violin(df, x=dispx, y=dispy, width=650)
        st.plotly_chart(fig)

featimp = st.form(border=True, key='featimp')

with featimp:
    st.header('Feature Importance')
    nfeat = st.number_input('Número de estimadores:', min_value=1, max_value=200, value=10, key='nfeat')
    rdfor = classificacao.classificar(dpip, nfeat, 'florest')
    if featimp.form_submit_button('Gerar gráfico'):
        fig = rdfor.randomflorest_importancia_feature_grafico()
        st.plotly_chart(fig)