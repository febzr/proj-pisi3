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
from imblearn.under_sampling import RandomUnderSampler

def cotovelo(n_cluster_teste,dados):
    sse = []

    # Definindo o range de k a ser testado
    k_range = range(1, n_cluster_teste)

    for k in k_range:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(dados)
        sse.append(kmeans.inertia_)

    # Plotando o gráfico do cotovelo
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=list(k_range),
        y=sse,
        mode='lines+markers',
        marker=dict(color='blue'),
        line=dict(color='blue'),
        name='Inércia'
    ))

    fig.update_layout(
        title='Método do Cotovelo',
        xaxis_title='Número de Clusters (k)',
        yaxis_title='Inércia',
        template='plotly_white'
    )

    return fig

data_cru = pd.read_parquet('heart_2022_no_nans.parquet')

data = pipeline.pipelines(data_cru)
X = data.create()

data_cru = data_cru.drop(columns=['State'])

dados_sem_estados = X.drop(columns=['State'])

#quando for treinar os clusters é necessario que os dados estejam no formato de matriz. Ou seja, nao pode ser um dataframe

dados_em_matriz=dados_sem_estados.values

x = dados_sem_estados.drop('HadHeartAttack', axis=1)
y = dados_sem_estados['HadHeartAttack']

rus = RandomUnderSampler(random_state=42)

# PARTE DO PLOTLY

st.title("CLUSTERIZAÇÃO")
st.text("Gráfico do método do cotovelo para auxiliar na escolha do número de clusters")
st.plotly_chart(cotovelo( 30 , dados_em_matriz))
num_clust = 5

if True:
    kmeans = KMeans(n_clusters=num_clust, random_state=42)
    label = kmeans.fit_predict(dados_sem_estados)
    label_series = pd.Series(label, name='cluster')

    clusters = pd.concat([dados_sem_estados,label_series], join='inner',axis=1)

    clusters_cru = pd.concat([data_cru,label_series], join='inner',axis=1)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(dados_sem_estados)

    df_pca = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
    df_pca["Cluster"] = label

    # Plotar com Plotly
    fig = px.scatter(
    df_pca,
    x="PC1",
    y="PC2",
    color="Cluster",
    title="Clusters de Dados com PCA",
    labels={"Cluster": "Cluster"},
    color_continuous_scale=px.colors.sequential.Viridis,
    )

    centers_pca = pca.transform(kmeans.cluster_centers_)
    fig.add_scatter(
    x=centers_pca[:, 0],
    y=centers_pca[:, 1],
    mode="markers",
    marker=dict(color="black", symbol="x", size=10),
    name="Centroides",
    )

    fig.update_layout(
    legend=dict(x=0, y=1.0),
    xaxis_title="PC1",
    yaxis_title="PC2",
    title="Clusters de Dados com PCA",
    )

    st.plotly_chart(fig)

    tamanho_cluster = []
    for k in range(num_clust):
        z = clusters_cru.loc[clusters_cru['cluster'] == k, 'cluster']
        tamanho_cluster.append(z.value_counts().values[0])

    # Criando o gráfico de barras com Plotly
    fig = go.Figure(data=[go.Bar(
        x=list(range(num_clust)),  # Lista de clusters
        y=tamanho_cluster,  # Tamanho dos clusters
        text=tamanho_cluster,  # Texto nas barras
        textposition='auto',  # Posição do texto
    )])

    fig.update_layout(
        title='Tamanho dos Clusters',
        xaxis_title='Cluster',
        yaxis_title='Número de Pontos'
    )

    st.text("")
    st.plotly_chart(fig)
    
    labelsgrandes = []

    colunas = data_cru.columns
    pd.set_option('display.max_rows', 1000)
    for coluna in colunas:
        u=clusters_cru[[coluna,'cluster']]
        todos_labels=u[coluna].value_counts().index
        if len(todos_labels)>100:
            labelsgrandes.append(coluna)

        if len(todos_labels)<100:
            fig = go.Figure()
            for clusteres in range(num_clust):
                cluster0 = u.loc[u['cluster'] == clusteres, coluna]
                h = cluster0.value_counts()
                labels = todos_labels
                value = []
                df_sorted = h.sort_index()
                    
                for tipo_classe in todos_labels:
                        classe = cluster0.loc[cluster0 == tipo_classe]
                        value.append(classe.value_counts().sum())   
                
                porcento=[]
                for un in value:
                    
                    porcento.append(f'{un/tamanho_cluster[clusteres]:.2f}')
        
                fig.add_trace(go.Bar(
                    x=labels,
                    y=value,
                    name=f'cluster {clusteres}',
                    customdata=porcento,
                
                    text=value,  # Adiciona rótulos de texto acima das barras
                    textposition='auto',  # Exibe os rótulos automaticamente
                    hovertemplate='<b>Classe:</b> %{x}<br>' +
                                '<b>Cluster:</b> %{name}<br>' +
                                '<b>Contagem:</b> %{y}<br>' +
                                '<b>Porcentagem:</b> %{customdata}<extra></extra>'
                    
                ))
            
            fig.update_layout(
                title=f'distribuição de {coluna} por clusters',
                xaxis_title='Grupos',
                yaxis_title='Valores',
                barmode='group',  # Agrupa as barras
            )

            # Mostrar o gráfico
            st.plotly_chart(fig)

    colunas = labelsgrandes
    # Loop através das colunas
    for coluna in colunas:
        # Filtra os dados para a coluna e o cluster
        u = clusters_cru[[coluna, 'cluster']]
        
        # Cria o gráfico boxplot
        fig = go.Figure()

        # Itera sobre os clusters para adicionar uma trace para cada cluster
        for cluster in range(num_clust):
            cluster_data = u.loc[u['cluster'] == cluster, coluna]
            
            fig.add_trace(go.Box(
                y=cluster_data,
                name=f'Cluster {cluster}',
                boxmean='sd',  # Adiciona a média e o desvio padrão
            ))

        fig.update_layout(
            title=f'Distribuição de {coluna} por Clusters',
            xaxis_title='Cluster',
            yaxis_title='Valores',
            boxmode='group',  # Agrupa as caixas
            xaxis=dict(tickvals=[f'Cluster {i}' for i in range(num_clust)]),  # Define os rótulos dos clusters no eixo x
        )

        # Mostrar o gráfico
        st.plotly_chart(fig)
