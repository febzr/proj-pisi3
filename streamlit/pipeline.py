import pandas as pd
import utility
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer


class pipelines():
    
    def __init__(self,data):
        self.data= data
        
    def create(self):
        frame = self.data

        def one_hot_encode(dados):
            one = utility.oneHotEncode(dados)
            creates = one.create(['Sex','LastCheckupTime','RemovedTeeth','SmokerStatus','ECigaretteUsage','RaceEthnicityCategory','AgeCategory','TetanusLast10Tdap','HadDiabetes'])

            return creates
            
        def GeneralHealth_transformation(dados):
            health_map = {'Very good':3.0, 'Fair':1.0 ,'Good':2.0 ,'Excellent':4.0, 'Poor':0.0}
            
            dados['GeneralHealth'] = dados['GeneralHealth'].map(health_map)
            return dados

        def PhysicalActivities_transformation(dados):
            yesorno = utility.yesOrNoTo01(dados)
            data = yesorno.transform(['PhysicalActivities','HadAngina','HadStroke','HadAsthma','HadSkinCancer','HadCOPD','HadDepressiveDisorder','HadKidneyDisease','HadArthritis','DeafOrHardOfHearing','BlindOrVisionDifficulty','DifficultyConcentrating','DifficultyWalking','DifficultyDressingBathing','DifficultyErrands','ChestScan','AlcoholDrinkers','HIVTesting','FluVaxLast12','PneumoVaxEver','CovidPos','HighRiskLastYear'])
            return data

        def normalization(dados):
            stand = utility.MinMax(dados)
            datas = stand.normalization(['BMI','WeightInKilograms','HeightInMeters','GeneralHealth','PhysicalHealthDays','MentalHealthDays','SleepHours'])
            return datas

        one_hot_encode_transformation = FunctionTransformer(func=one_hot_encode, validate=False)
        GeneralHealth_transformer = FunctionTransformer(func=GeneralHealth_transformation, validate=False)
        yesorno_transformation = FunctionTransformer(func=PhysicalActivities_transformation, validate=False)
        normalization_transformation = FunctionTransformer(func=normalization, validate=False)


        pipeline = Pipeline(steps=[
            ('one hot encode', one_hot_encode_transformation),
            ('GeneralHealth numerado',GeneralHealth_transformer),
            ('transforma yes em 1 no em 0',yesorno_transformation),
            ('reduz tudo para o intervalo de 0 e 1', normalization_transformation)
        ])

        frame2 = pipeline.fit_transform(frame)
        return frame2

