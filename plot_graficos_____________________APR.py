import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def analyze_data(file_path):
  # Carregar o dataset
  df = pd.read_parquet(file_path)

  # Informações básicas do dataset
  print("Informações básicas do dataset:")
  print(df.info())

  # Estatísticas descritivas
  print("\nEstatísticas descritivas:")
  print(df.describe())

  # Verificar valores ausentes
  print("\nValores ausentes:")
  print(df.isnull().sum())

  # Identificar colunas numéricas e categóricas
  numerical_cols = df.select_dtypes(include=['number']).columns
  categorical_cols = df.select_dtypes(exclude=['number']).columns

  # Analisar a distribuição das variáveis numéricas
  print("\nDistribuição das variáveis numéricas:")
  for col in numerical_cols:
    plt.figure()
    sns.histplot(df[col])
    plt.title(f'Distribuição de {col}')
    plt.show()

  # Analisar a frequência das variáveis categóricas
  print("\nFrequência das variáveis categóricas:")
  for col in categorical_cols:
    plt.figure()
    sns.countplot(x=col, data=df)
    plt.title(f'Frequência de {col}')
    plt.xticks(rotation=45, ha='right')
    plt.show()

  # Correlação entre variáveis numéricas
  print("\nCorrelação entre variáveis numéricas:")
  plt.figure(figsize=(12, 10))
  sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
  plt.title('Mapa de Calor da Correlação')
  plt.show()

  # Analisar padrões para valores específicos em uma coluna (exemplo)
  target_column = 'target_column'  # Substitua pelo nome da coluna desejada
  target_value = 1  # Substitua pelo valor desejado

  if target_column in df.columns:
    print(f"\nAnalisando padrões para {target_column} = {target_value}:")
    for col in df.columns:
      if col != target_column:
        print(f"\n-- {col} --")
        if col in numerical_cols:
          print(df.groupby(target_column)[col].describe())
        else:
          print(df.groupby(col)[target_column].value_counts(normalize=True))

# Exemplo de uso
analyze_data('heart_2022_no_nans.parquet')  # Substitua pelo caminho do seu arquivo CSV