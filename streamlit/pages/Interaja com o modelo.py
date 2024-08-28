import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
from sklearn.ensemble import RandomForestClassifier
import pipeline
from catboost import CatBoostClassifier
import tensorflow as tf
import keras
import gzip

# Carregar os modelos
modrar = gzip.open('./models/rf.pkl.gz', 'rb')

modrf = pickle.load(modrar)
modlog = pickle.load(open('./models/lr.pkl', 'rb'))
modknn = pickle.load(open('./models/knn_model.pkl', 'rb'))

modcat = CatBoostClassifier()
modcat.load_model('./models/catboost_model')

modredes = keras.models.load_model('./models/redes.keras', custom_objects={'f1_scorez': 'f1_scorez'})

st.title('Previsão de Ataque Cardíaco')
st.write('Baseado nos dados inseridos, o modelo prevê se o paciente terá um ataque cardíaco ou não.')

st.write('Insira os dados do paciente:')

forms = st.form(key='cardio')

with forms:
    sex = st.radio(label='Sexo:', options=["Masculino", "Feminino"], index=None, horizontal=True)
    if sex == "Masculino":
        sex = "Male"
    else:
        sex = "Female"

    genhealth = st.radio(label='Saúde Física:', options=["Ótima", "Boa", "Regular", "Ruim"], index=None, horizontal=True)
    if genhealth == "Ótima":
        genhealth = "Excellent"
    elif genhealth == "Boa":
        genhealth = "Good"
    elif genhealth == "Regular":
        genhealth = "Fair"
    elif genhealth == "Ruim":
        genhealth = "Poor"

    physhealthdays = st.number_input(label='Quantos dias de saúde física ruim nos últimos 30 dias?', min_value=0, max_value=30, value=0, step=1, format=None, key=None)

    menthealthdays = st.number_input(label='Quantos dias de saúde mental ruim nos últimos 30 dias?', min_value=0, max_value=30, value=0, step=1, format=None, key=None)

    lastcheckup = st.radio(label='Quando foi a última vez que você fez um check-up?', options=["Menos de um ano", "Dentro dos últimos 2 anos", "Dentre os últimos 5 anos", "Há 5 anos ou mais"], index=None, horizontal=False)
    if lastcheckup == "Menos de um ano":
        lastcheckup = "Within past year (anytime less than 12 months ago)"
    elif lastcheckup == "Dentro dos últimos 2 anos":
        lastcheckup = "Within past 2 years (1 year but less than 2 years ago)"
    elif lastcheckup == "Dentre os últimos 5 anos":
        lastcheckup = "Within past 5 years (2 years but less than 5 years ago)"
    elif lastcheckup == "Há 5 anos ou mais":
        lastcheckup = "5 or more years ago"

    physact = st.radio(label='Você fez alguma atividade física nos últimos 30 dias?', options=["Sim", "Não"], index=None, horizontal=True)
    if physact == "Sim":
        physact = "Yes"
    elif physact == "Não":
        physact = "No"

    sleephours = st.number_input(label='Quantas horas você dorme por dia?', min_value=0, max_value=24, value=0, step=1, format=None, key=None)

    removedteeth = st.radio(label='Você já teve dentes removidos?', options=["Todos", "6 ou mais dentes, mas não todos", "De 1 a 5 dentes", "Nenhum"], index=None, horizontal=True)
    if removedteeth == "Todos":
        removedteeth = "All"
    elif removedteeth == "6 ou mais dentes, mas não todos":
        removedteeth = "6 or more, but not all"
    elif removedteeth == "De 1 a 5 dentes":
        removedteeth = "1 to 5"
    elif removedteeth == "Nenhum":
        removedteeth = "None of them"

    angina = st.radio(label='Você sente dores no peito?', options=["Sim", "Não"], index=None, horizontal=True)
    if angina == "Sim":
        angina = "Yes"
    elif angina == "Não":
        angina = "No"

    stroke = st.radio(label='Você já teve um AVC?', options=["Sim", "Não"], index=None, horizontal=True)
    if stroke == "Sim":
        stroke = "Yes"
    elif stroke == "Não":
        stroke = "No"

    asthma = st.radio(label='Você tem asma?', options=["Sim", "Não"], index=None, horizontal=True)
    if asthma == "Sim":
        asthma = "Yes"
    elif asthma == "Não":
        asthma = "No"

    skincancer = st.radio(label='Você tem ou teve câncer de pele?', options=["Sim", "Não"], index=None, horizontal=True)
    if skincancer == "Sim":
        skincancer = "Yes"
    elif skincancer == "Não":
        skincancer = "No"

    copd = st.radio(label='Você tem ou teve enfisema pulmonar?', options=["Sim", "Não"], index=None, horizontal=True)
    if copd == "Sim":
        copd = "Yes"
    elif copd == "Não":
        copd = "No"

    depressivedisorder = st.radio(label='Você tem ou teve um transtorno depressivo?', options=["Sim", "Não"], index=None, horizontal=True)
    if depressivedisorder == "Sim":
        depressivedisorder = "Yes"
    elif depressivedisorder == "Não":
        depressivedisorder = "No"

    kidneydisease = st.radio(label='Você tem ou teve doença renal crônica?', options=["Sim", "Não"], index=None, horizontal=True)
    if kidneydisease == "Sim":
        kidneydisease = "Yes"
    elif kidneydisease == "Não":
        kidneydisease = "No"

    arthitis = st.radio(label='Você tem ou teve artrite?', options=["Sim", "Não"], index=None, horizontal=True)
    if arthitis == "Sim":
        arthitis = "Yes"
    elif arthitis == "Não":
        arthitis = "No"

    diabetes = st.radio(label='Você tem ou teve diabetes?', options=["Sim", "Sim, porém apenas durante a gravidez", "Não, pré-diabetes ou quase diabético", "Não"], index=None, horizontal=False)
    if diabetes == "Sim":
        diabetes = "Yes"
    elif diabetes == "Sim, porém apenas durante a gravidez":
        diabetes = "Yes, but only during pregnancy (female)"
    elif diabetes == "Não, pré-diabetes ou quase diabético":
        diabetes = "No, pre-diabetes or borderline diabetes"
    elif diabetes == "Não":
        diabetes = "No"

    deaf = st.radio(label='Você é surdo ou tem dificuldade auditiva?', options=["Sim", "Não"], index=None, horizontal=True)
    if deaf == "Sim":
        deaf = "Yes"
    elif deaf == "Não":
        deaf = "No"

    blind = st.radio(label='Você é cego ou tem dificuldade visual?', options=["Sim", "Não"], index=None, horizontal=True)
    if blind == "Sim":
        blind = "Yes"
    elif blind == "Não":
        blind = "No"

    concentrating = st.radio(label='Você tem dificuldade de concentração?', options=["Sim", "Não"], index=None, horizontal=True)
    if concentrating == "Sim":
        concentrating = "Yes"
    elif concentrating == "Não":
        concentrating = "No"

    walking = st.radio(label='Você tem dificuldade para andar?', options=["Sim", "Não"], index=None, horizontal=True)
    if walking == "Sim":
        walking = "Yes"
    elif walking == "Não":
        walking = "No"

    dressing = st.radio(label='Você tem dificuldade para se vestir?', options=["Sim", "Não"], index=None, horizontal=True)
    if dressing == "Sim":
        dressing = "Yes"
    elif dressing == "Não":
        dressing = "No"

    errands = st.radio(label='Você tem dificuldade para fazer tarefas domésticas?', options=["Sim", "Não"], index=None, horizontal=True)
    if errands == "Sim":
        errands = "Yes"
    elif errands == "Não":
        errands = "No"

    smoker = st.radio(label='Você fuma?', options=["Fumante - fuma todos os dias", "Fumante - fuma ocasionalmente", "Já fumou", "Nunca fumou"], index=None, horizontal=False)
    if smoker == "Fumante - fuma todos os dias":
        smoker = "Current smoker - now smokes every day"
    elif smoker == "Fumante - fuma ocasionalmente":
        smoker = "Current smoker - now smokes some days"
    elif smoker == "Já fumou":
        smoker = "Former smoker"
    elif smoker == "Nunca fumou":
        smoker = "Never smoked"

    ecig = st.radio(label='Você usa cigarros eletrônicos?', options=["Usa todos os dias", "Usa ocasionalmente", "Já usou", "Nunca usou"], index=None, horizontal=False)
    if ecig == "Usa todos os dias":
        ecig = "Use them every day"
    elif ecig == "Usa ocasionalmente":
        ecig = "Use them some days"
    elif ecig == "Já usou":
        ecig = "Not at all (right now)"
    elif ecig == "Nunca usou":
        ecig = "Never used e-cigarettes in my entire life"

    chestscan = st.radio(label='Você já fez uma tomografia computadorizada do tórax?', options=["Sim", "Não"], index=None, horizontal=True)
    if chestscan == "Sim":
        chestscan = "Yes"
    elif chestscan == "Não":
        chestscan = "No"

    race = st.radio(label='Qual a sua raça?', options=["Preto", "Branco", "Pardo", "Hispânico"], index=None, horizontal=False)
    if race == "Preto":
        race = "Black only, Non-Hispanic"
    if race == "Branco":
        race = "White only, Non-Hispanic"
    if race == "Pardo":
        race = "Multiracial, Non-Hispanic"
    if race == "Hispânico":
        race == "Hispanic"

    agecat = st.radio(label='Qual a sua idade?', options=["18 a 24 anos", "25 a 29 anos", "30 a 34 anos", "35 a 39 anos", "40 a 44 anos", "45 a 49 anos", "50 a 54 anos", "55 a 59 anos", "60 a 64 anos", "65 a 69 anos", "70 a 74 anos", "75 a 79 anos", "80 anos ou mais"], index=None, horizontal=False)
    if agecat == "18 a 24 anos":
        agecat = "Age 18 to 24"
    elif agecat == "25 a 29 anos":
        agecat = "Age 25 to 29"
    elif agecat == "30 a 34 anos":
        agecat = "Age 30 to 34"
    elif agecat == "35 a 39 anos":
        agecat = "Age 35 to 39"
    elif agecat == "40 a 44 anos":
        agecat = "Age 40 to 44"
    elif agecat == "45 a 49 anos":
        agecat = "Age 45 to 49"
    elif agecat == "50 a 54 anos":
        agecat = "Age 50 to 54"
    elif agecat == "55 a 59 anos":
        agecat = "Age 55 to 59"
    elif agecat == "60 a 64 anos":
        agecat = "Age 60 to 64"
    elif agecat == "65 a 69 anos":
        agecat = "Age 65 to 69"
    elif agecat == "70 a 74 anos":
        agecat = "Age 70 to 74"
    elif agecat == "75 a 79 anos":
        agecat = "Age 75 to 79"
    elif agecat == "80 anos ou mais":
        agecat = "Age 80 or older"

    height = st.number_input(label='Qual a sua altura em centímetros?', min_value=0, max_value=300, value=0, step=1, format=None, key=None)
    height = height / 100

    weight = st.number_input(label='Qual o seu peso em quilogramas?', min_value=0, max_value=300, value=0, step=1, format=None, key=None)

    alcohol = st.radio(label='Você bebe álcool?', options=["Sim", "Não"], index=None, horizontal=True)
    if alcohol == "Sim":
        alcohol = "Yes"
    elif alcohol == "Não":
        alcohol = "No"

    hivtest = st.radio(label='Você já fez um teste de HIV?', options=["Sim", "Não"], index=None, horizontal=True)
    if hivtest == "Sim":
        hivtest = "Yes"
    elif hivtest == "Não":
        hivtest = "No"

    fluvax = st.radio(label='Você tomou a vacina contra a gripe nos últimos 12 meses?', options=["Sim", "Não"], index=None, horizontal=True)
    if fluvax == "Sim":
        fluvax = "Yes"
    elif fluvax == "Não":
        fluvax = "No"

    pneumovax = st.radio(label='Você tomou a vacina pneumocócica nos últimos 12 meses?', options=["Sim", "Não"], index=None, horizontal=True)
    if pneumovax == "Sim":
        pneumovax = "Yes"
    elif pneumovax == "Não":
        pneumovax = "No"

    covidpos = st.radio(label='Você testou positivo para COVID-19?', options=["Sim", "Não"], index=None, horizontal=True)
    if covidpos == "Sim":
        covidpos = "Yes"
    elif covidpos == "Não":
        covidpos = "No"

    tetanus = st.radio(label='Você tomou a vacina contra o tétano nos últimos 10 anos?', options=["Sim", "Não"], index=None, horizontal=True)
    if tetanus == "Sim":
        tetanus = "Yes, received Tdap"
    elif tetanus == "Não":
        tetanus = "No, did not receive any tetanus shot in the past 10 years"

    highrisklastyear = st.radio(label='Você foi um paciente de alto risco nos útlimos 12 meses?', options=["Sim", "Não"], index=None, horizontal=True)
    if highrisklastyear == "Sim":
        highrisklastyear = "Yes"
    elif highrisklastyear == "Não":
        highrisklastyear = "No"

if forms.form_submit_button('Prever Ataque Cardíaco'):
    if (sex == None) or (genhealth == None) or (physhealthdays == None) or (menthealthdays == None) or (lastcheckup == None) or (physact == None) or (sleephours == None) or (removedteeth == None) or (angina == None) or (stroke == None) or (asthma == None) or (skincancer == None) or (copd == None) or (depressivedisorder == None) or (kidneydisease == None) or (arthitis == None) or (diabetes == None) or (deaf == None) or (blind == None) or (concentrating == None) or (walking == None) or (dressing == None) or (errands == None) or (smoker == None) or (ecig == None) or (chestscan == None):
        st.write('Por favor, preencha todos os campos.')
        st.rerun()

    data_org = pd.read_parquet('heart_2022_no_nans.parquet')
    data_org.drop(columns=['State'], inplace=True)
    data_org.drop(columns=['HadHeartAttack'], inplace=True)

    bmi = weight / (height ** 2)

    # Criar um dataframe com os dados inseridos

    data = [sex, genhealth, physhealthdays, menthealthdays, lastcheckup, physact, sleephours, removedteeth, angina, stroke, asthma, skincancer, copd, depressivedisorder, kidneydisease, arthitis, diabetes, deaf, blind, concentrating, walking, dressing, errands, smoker, ecig, chestscan, race, agecat, height, weight, bmi, alcohol, hivtest, fluvax, pneumovax, tetanus, highrisklastyear, covidpos]

    df1 = pd.DataFrame([data], columns=['Sex', 'GeneralHealth', 'PhysicalHealthDays', 'MentalHealthDays', 'LastCheckupTime', 'PhysicalActivities', 'SleepHours', 'RemovedTeeth', 'HadAngina', 'HadStroke', 'HadAsthma', 'HadSkinCancer', 'HadCOPD', 'HadDepressiveDisorder', 'HadKidneyDisease', 'HadArthritis', 'HadDiabetes', 'DeafOrHardOfHearing', 'BlindOrVisionDifficulty', 'DifficultyConcentrating', 'DifficultyWalking', 'DifficultyDressingBathing', 'DifficultyErrands', 'SmokerStatus', 'ECigaretteUsage', 'ChestScan', 'RaceEthnicityCategory', 'AgeCategory', 'HeightInMeters', 'WeightInKilograms', 'BMI', 'AlcoholDrinkers', 'HIVTesting', 'FluVaxLast12', 'PneumoVaxEver', 'TetanusLast10Tdap', 'HighRiskLastYear', 'CovidPos'])

    data_concat = pd.concat([df1, data_org], ignore_index=True)

    # Aplicar o pipeline
    df_tratada = pipeline.pipelines(data_concat)

    df = df_tratada.create()
    df = df.loc[[0]]
    print(df)

    promissores = 0

    # Prever o ataque cardíaco
    catboost_pred = modcat.predict(df)[0]

    knn_pred = modknn.predict(df)[0]

    log_pred = modlog.predict(df)[0]

    rf_pred = modrf.predict(df)[0]

    redes_pred = (modredes.predict(df) >0.5).astype("int32")
    redes_pred = redes_pred[0][0]

    if catboost_pred == 1:
        st.write('**Catboost:** :red[O modelo CatBoost prevê que o paciente terá um ataque cardíaco.]')
        promissores += 1
    elif catboost_pred == 0:
        st.write('**Catboost:** :green[O modelo CatBoost prevê que o paciente não terá um ataque cardíaco.]')

    if knn_pred == 1:
        st.write('**KNN:** :red[O modelo KNN prevê que o paciente terá um ataque cardíaco.]')
        promissores += 1
    elif knn_pred == 0:
        st.write('**KNN:** :green[O modelo KNN prevê que o paciente não terá um ataque cardíaco.]')
    
    if log_pred == 1:
        st.write('**Regressão Logística:** :red[O modelo de Regressão Logística prevê que o paciente terá um ataque cardíaco.]')
        promissores += 1
    elif log_pred == 0:
        st.write('**Regressão Logística:** :green[O modelo de Regressão Logística prevê que o paciente não terá um ataque cardíaco.]')

    if rf_pred == 1:
        st.write('**Random Forest:** :red[O modelo Random Forest prevê que o paciente terá um ataque cardíaco.]')
        promissores += 1
    elif rf_pred == 0:
        st.write('**Random Forest:** :green[O modelo Random Forest prevê que o paciente não terá um ataque cardíaco.]')

    if redes_pred == 1:
        st.write('**Redes Neurais:** :red[O modelo de Redes Neurais prevê que o paciente terá um ataque cardíaco.]')
        promissores += 1
    elif redes_pred == 0:
        st.write('**Redes Neurais:** :green[O modelo de Redes Neurais prevê que o paciente não terá um ataque cardíaco.]')

    st.write(f'  **Segundo a média dos modelos, há uma probabilidade de {round((promissores/5)*100, 2)}% de o paciente ter um ataque cardíaco.**')