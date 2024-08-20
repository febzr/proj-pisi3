from altair import renderers
from matplotlib import colors
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier
import plotly.express as px
import plotly.graph_objects as go


class classificar:
    def __init__(self, dados, n_estimators, modelo_escolhido):
        self.dados = dados
        self.n_estimatos = n_estimators
        self.modelo_escolhido = modelo_escolhido
        np.random.seed(42)
        if modelo_escolhido == "florest":
            self.modelo = RandomForestClassifier(
                n_estimators=self.n_estimatos, random_state=42
            )
        elif modelo_escolhido == "knn":
            self.modelo = KNeighborsClassifier(n_neighbors=self.n_estimatos)

    def cross_validation(self, n):
        y = self.dados["HadHeartAttack"]
        X = self.dados.drop("HadHeartAttack", axis=1)
        scores = cross_val_score(self.modelo, X, y, cv=n, scoring="accuracy")
        print(f"Accuracy Scores for each fold: {scores}")
        print(f"Mean Accuracy: {np.mean(scores)}")
        print(f"Standard Deviation: {np.std(scores)}")

    def acuracia(self):
        y = self.dados["HadHeartAttack"]
        X = self.dados.drop("HadHeartAttack", axis=1)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        self.modelo.fit(X_train, y_train)
        y_pred = self.modelo.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred)
        print(f"Accuracy: {accuracy}")
        print("-----------")
        print("Confusion Matrix:")
        print(conf_matrix)
        print("------------")
        print("Classification Report:")
        print(class_report)

    def predicao(self, dado):
        self.acuracia()
        prediction = self.modelo.predict(dado)

        print("Previsão para o novo dado:", prediction[0])

    def randomflorest_importancia_feature_grafico(self):
        if self.modelo_escolhido == "florest":
            y = self.dados["HadHeartAttack"]
            X = self.dados.drop("HadHeartAttack", axis=1)
            self.modelo.fit(X, y)
            feature_importance = pd.Series(
                self.modelo.feature_importances_, index=X.columns
            ).sort_values(ascending=False)
            feature_importance_df = feature_importance.reset_index()
            feature_importance_df.columns = ["Feature", "Importance"]

            # Plotando com Plotly
            fig = px.bar(
                feature_importance_df,
                x="Feature",
                y="Importance",
                width=650,
                title="Importância das Features - Random Forest",
                labels={"Importância": "Pontuação da importância"},
            )
            
            return fig
            
        else:
            print("para usar essa classe o modelo escolhido deve ser random florest")

    def knn_fronteira_grafico(self):
        #esse grafico chegou a pesar 2.7 de ram. Que deus tenha misericordia do seu pc
        if self.modelo_escolhido == "knn":
            y = self.dados["HadHeartAttack"]
            X = self.dados.drop("HadHeartAttack", axis=1)

            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X)
            self.modelo.fit(X_pca, y)
            x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
            y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
            xx, yy = np.meshgrid(
                np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1)
            )

            # Fazendo previsões para cada ponto na grade
            Z = self.modelo.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)

            fig = go.Figure()

            # Adicionando a fronteira de decisão
            fig.add_trace(
                go.Contour(
                    x=np.arange(x_min, x_max, 0.1),
                    y=np.arange(y_min, y_max, 0.1),
                    z=Z,
                    colorscale="Viridis",
                    opacity=0.8,
                    showscale=False,
                    showlegend=True,
                    name="Decision Boundary",
                )
            )
            X_pca_0 = X_pca[y == 0]
            X_pca_1 = X_pca[y == 1]

            # Adicionando os pontos de dados
            fig.add_trace(
                go.Scatter(
                    x=X_pca_0[:, 0],
                    y=X_pca_0[:, 1],
                    mode="markers",
                    name="não teve",
                    showlegend=True,
                    visible='legendonly',
                    marker=dict(
                        color="purple", line=dict(width=1, color="black"), size=5
                    ),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=X_pca_1[:, 0],
                    y=X_pca_1[:, 1],
                    mode="markers",
                    name="teve",
                    showlegend=True,
                    visible='legendonly',
                    marker=dict(
                        color="yellow", line=dict(width=1, color="black"), size=5
                    ),
                )
            )

            # Ajustando layout do gráfico
            fig.update_layout(
                title="Fronteira de Decisão do KNN - Usando PCA",
                xaxis_title="",
                yaxis_title="",
                legend=dict(x=0, y=1.0, traceorder="normal", orientation="h"),
            )

            return fig
        else:
            print("para usar essa classe o modelo escolhido deve ser random florest")
