import pipeline
import pandas as pd
import clusterização

data = pipeline.pipelines(pd.read_parquet('heart_2022_no_nans.parquet'))
X = data.create()
X = X.drop(columns=['State'])

y=X.values

cluster = clusterização.clusters(y)

cluster.grafico_cluster(X,5).show()