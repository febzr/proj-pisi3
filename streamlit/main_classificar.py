import classificacao
import pipeline
import pandas as pd


data = pipeline.pipelines(pd.read_parquet("heart_2022_no_nans.parquet"))
X = data.create()
X = X.drop(columns=["State"])

linha_especifica = X.iloc[7]
print(linha_especifica["HadHeartAttack"])
linha_df = pd.DataFrame([linha_especifica])
linha_df = linha_df.drop("HadHeartAttack", axis=1)

clasifica = classificacao.classificar(X, 5, "knn")

clasifica.acuracia()
