from sklearn.cluster import KMeans
from sklearn.model_selection import KFold
from sklearn.metrics import silhouette_score
import numpy as np
import pandas as pd
from sklearn.metrics import davies_bouldin_score
import plotly.express as px
from sklearn.decomposition import PCA
import plotly.graph_objects as go


class clusters:
    def __init__(self, dados):
        self.dados = dados
        np.random.seed(42)
        


    def cotovelo(self,n_cluster_teste):
        sse = []

        # Definindo o range de k a ser testado
        k_range = range(1, n_cluster_teste)

        for k in k_range:
            kmeans = KMeans(n_clusters=k)
            kmeans.fit(self.dados)
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
    
    def acuracia(self,n_cluster):
        modelo = KMeans(n_clusters=n_cluster,random_state=42)
        label = modelo.fit_predict(self.dados)
        score = silhouette_score(self.dados, label)
        dbi = davies_bouldin_score(self.dados, label)
        print(f"Silhouette Score: {score}")
        print(f"davies bouldin Score: {dbi}")

    def labels(self, csv,n_cluster):
        modelo = KMeans(random_state=42,n_clusters=n_cluster)
        label = modelo.fit_predict(self.dados)
        frame = pd.read_csv(csv)
        frame["cluster"] = label
        frame.to_csv(f"{n_cluster}_clusters_data.csv")

    def treinar(self,n_cluster):
        modelo = KMeans(random_state=42,n_clusters=n_cluster)
        label = modelo.fit_predict(self.dados)
        return label

    def grafico_cluster(self, dados_pipeline,n_cluster):

        df = dados_pipeline
        modelo = KMeans(random_state=42,n_clusters=n_cluster)
        label = modelo.fit_predict(self.dados)

        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(df)

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

        centers_pca = pca.transform(modelo.cluster_centers_)
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

        return fig

#        fig.show()     DEPRECATED
