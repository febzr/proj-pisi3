import streamlit as st
import pipeline
import pandas as pd
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

#IMPORTS
cotovelo = pio.read_json('graph/cotovelo.json')
actcluster = pio.read_json('graph/grafico_atividades_fisicas.json')
alcoholcluster = pio.read_json('graph/grafico_alcohol.json')
attackcluster = pio.read_json('graph/grafico_hadheartattack.json')
idadescluster = pio.read_json('graph/idadecluster.json')
sexcluster = pio.read_json('graph/grafico_sexo.json')
tamanhocluster = pio.read_json('graph/tamanhocluster.json')

# PARTE DO STREAMLIT
st.set_page_config(page_title='Clusterização', page_icon=':heart:', layout='centered', initial_sidebar_state='auto')

st.title("CLUSTERIZAÇÃO")
st.markdown("Gráfico do método do cotovelo para auxiliar na escolha do número de clusters")
st.plotly_chart(cotovelo)
st.markdown('Gráficos de Silhueta para melhor determinar a quantidade de clusters')

containersilhueta = st.container(border=True)

with containersilhueta:
    st.image('graph/silhouette_4.png', width=670)
    st.image('graph/silhouette_5.png', width=670)
    st.image('graph/silhouette_6.png', width=670)

st.plotly_chart(tamanhocluster)
contexto = st.container()

with contexto:
    st.markdown('Cluster 0 - Apenas homens idosos, alto índice de doenças físicas, mobilidade reduzida, prática regular de exercícios físicos, checkups requentes e baixos índices de doenças mentais.')
    st.markdown('Cluster 1 - Maioria de mulheres idosas, vários problemas de saúde física e mental, mantinham checkups regulares, poucos exercícios físicos e histórico de tabgismo. Maior atenção à saúde e hábitos de prevenção.')
    st.markdown('Cluster 2 - Maioria de jovens, com ocorrência mediana de problemas mentais e físicos e poucas visitas médicas. Poucas condições raras de saúde, e maioria sem histórico de doenças físicas. Tabagismo recorrente, baixa vacinação contra a gripe e maior probabilidade de riscos relacionados ao HIV.')
    st.markdown('Cluster 3 - Apenas mulheres de meia idade, com maor quantidade de dados absolutos, poucos problemas físicos e alto nível de problemas mentais. Rotina regular de exercícios físicos e menor frequência de checkups. Baixo índice de doenças físicas.')
    st.markdown('Cluster 4 - Apenas homens de meia idade, poucos problemas físicos e mentais, mantendo rotina regular de exercícios físicos e menos checkups que o cluster 2. Poucas doenças médicas graves e baixo índice de tabagismo, porém alto consumo de álcool e bons valores de saúde geral.')

st.plotly_chart(actcluster)
st.plotly_chart(alcoholcluster)
st.plotly_chart(attackcluster)
st.plotly_chart(idadescluster)
st.plotly_chart(sexcluster)