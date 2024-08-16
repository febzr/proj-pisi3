import pipeline
import pandas as pd
import clusterização
import classificacao

data = pipeline.pipelines(pd.read_parquet('heart_2022_no_nans.parquet'))
X = data.create()
X = X.drop(columns=['State'])

cluster = clusterização.clusters(X, 15)

cluster.grafico_cluster(X)

clas = classificacao.classificar(X, 100, "knn")

clas.knn_fronteira_grafico()

rdfor = classificacao.classificar(X, 100, "florest")

rdfor.randomflorest_importancia_feature_grafico()