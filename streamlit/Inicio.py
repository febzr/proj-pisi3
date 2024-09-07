import streamlit as st
import plotly.express as px
import pandas as pd
import pipeline_gen

df = pd.read_parquet('heart_2022_no_nans.parquet')
df2 = df
df2.drop(columns=['State'], inplace=True)
data = pipeline_gen.pipelines(df2)
dpip = data.create()
print(dpip)


# PARTE DO STREAMLIT
st.set_page_config(page_title='Início - HDI2022', page_icon=':heart:', layout='centered', initial_sidebar_state='auto')

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
    st.header('Gráfico de Violino')
    dispx = st.selectbox('Selecione o eixo x:', df.columns)
    dispy = st.selectbox('Selecione o eixo y:', df.columns)

    but = disp.form_submit_button('Gerar gráfico')
    if but:

        fig = px.violin(df, x=dispx, y=dispy, width=650)
        st.plotly_chart(fig)