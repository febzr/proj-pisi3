from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
class oneHotEncode():
    def __init__(self,dados):
        self.dados = dados
    
    def create(self, coluna_categorica):
        data =self.dados
        for k in coluna_categorica:
            encoder = OneHotEncoder(sparse_output=False)
            
            coluna_df = data[[k]]
            
            
            encoded_data = encoder.fit_transform(coluna_df)
            
        
            encoded_columns = encoder.get_feature_names_out([k])
            
            
            df_encoded = pd.DataFrame(encoded_data, columns=encoded_columns)
            
            data = pd.concat([data,df_encoded],axis=1)
            data = data.drop(columns=[k])
        return data
    
class yesOrNoTo01():
    def __init__(self,dados):
        self.dados=dados
    
    def transform(self,lista):
        data = self.dados
        for k in lista:
            maps= {'Yes':1.0 , 'No':0.0 , 'Tested positive using home test without a health professional':0.5}
            
            data[k] = data[k].map(maps)
        return data
    
    
class MinMax():
    def __init__(self,dados):
        self.dados =dados
    
    def normalization(self,lista):
        data=self.dados
        for k in lista:
            scaler = MinMaxScaler()
            data[k] = scaler.fit_transform(data[[k]])
        return data
    

class stand():
    def __init__(self,dados):
        self.dados =dados
    
    def padronizacao(self,lista):
        data=self.dados
        for k in lista:
            scaler = StandardScaler()
            data[k] = scaler.fit_transform(data[[k]])
        return data
    