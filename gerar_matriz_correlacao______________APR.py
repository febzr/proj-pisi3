import pipeline
import pandas as pd
import plotly.express as px


data = pipeline.pipelines(pd.read_parquet('heart_2022_no_nans.parquet'))
X = data.create()
X = X.drop(columns=['State'])

fig = px.imshow(X.corr(numeric_only=True),text_auto=True)
fig.update_layout(
    title="Heatmap de Correlação",
    width=2000,
    height=2000,
)

fig.show()