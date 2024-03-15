import numpy as np
from pycaret.classification import load_model, predict_model
import streamlit as st
import pandas as pd

st.set_page_config(layout="wide")
from PIL import Image

model = load_model('D:/Python_Project/ICU/Abhishek_Yadav_ICU/ICUPredictionDeepAarogya/Data/ICU_prediction_finetune_xgb')

cat_map = {
    "No": 0,
    "Yes": 1,
    "Not Available": np.nan,
    "Male": 1,
    "Female": 0,
    "NORMAL": 0,
    "HIGH":1,
    "SAFE":0,
    "UNSAFE":1,
    "OPTIMUM":0,
    "LOW":1,
}


# def predict(model, input_df):
#     model.memory = "Data/"
#     predictions_df = predict_model(estimator=model, data=input_df)
#     predictions = predictions_df['prediction_label'][0]
#     confidence = predictions_df['prediction_score'][0]
#     return predictions, confidence


def get_data():
    data = pd.read_csv("D:\Python_Project\ICU\ESIC_ICU\ICU_data\ICU_DDUH_1_Sheet1.csv")
    data.columns = list(map(str.strip, list(data.columns)))
    data = data[['SEX', 'AGE', 'AKI_on_CKD', 'CKD', 'AKI','Pregnancy/post_delivery','Heamatemesis',	'PHTN',	
                 'Malena/Rectal_bleed', 'Heamoptysis',	'Pneumonia/infective_etio','Bronchial_Asthama','COPD',
                 'ARDS','Respiratory_failure_type2','Respiratory_failure_type1', 'ECHO_abnormalties',	
                 'Coronary_artery_disease','Valvular_heart_disease',	'Embolism',	'IC_bleed',	'Infarct',	
                 'HR_max','HR_min', 'HR_spot','BP_max', 'BP_min',	'BP_spot',	'SPO2_max',	'SPO2_min',	'SPO2_spot', 
                 'RR_max', 'RR_min', 'RR_spot',	'GCS','Eye_opening', 'Verbal_response',	'Motor_response','Fever',	
                 'SOB',	'Airway','Perfusion', 'Ascitis', 'Effusion/pleural/pericardial',	'Chest_Pain', 'Lactate',
                 'PCO2','APACHE_II','qSOFA','Urine_Output',	'Procalcitonin','PF_ratio',	'Ketoacidosis',	'Met_Acidosis',
                 'Resp_Acidosis', 'Cultures', 'Hb',	'TLC',	'PLats','Urea',	'Creatinine', 'Sodium',	'Pottasium', 
                 'Albumin',	'Bil',	'AST',	'ALT',	'ALP',	'Diagnosis']]			
    return data

data = get_data()
print(data)

def main():
    data = get_data()
    image2 = Image.open('D:/Python_Project/ICU/Abhishek_Yadav_ICU/ICUPredictionDeepAarogya/Images/icu.png')
    st.sidebar.info('This app is created to predict a particular patient need ICU treatment or no. [DeepAarogya]] - Version 2')
    st.sidebar.image(image2)
    st.title("ICU Prediction V3")

    st.sidebar.title("Check Analysis:")

    check_data = st.sidebar.checkbox('Check Feature Importance')
    if check_data:
        st.header("Feature Importance:")
        db = Image.open('D:/Python_Project/ICU/Abhishek_Yadav_ICU/ICUPredictionDeepAarogya/Images/ft_importance.png')
        st.image(db)

    test_model = st.sidebar.checkbox('Test Model V2 Online', True)
    if test_model:
        cols = data.columns
        print(cols)

        # Index(['Age', 'Gender..female.0..male1.', 'Fever', 'Cough', 'SOB', 'Fatigue',
        #        'Sputum', 'Myalgia', 'Diarrhea', 'Nausea.Vomiting', 'Sore.throat',
        #        'Chest.discomfort..chest.pain', 'smoking_history', 'hypertensionhx',
        #        'diabeteshx', 'coronaryheartdiseasehx', 'copdhx', 'carcinomahx',
        #        'ckdhx', 'ALT', 'HR', 'Lymphocyte', 'SpO2', 'Procalcitonin', 'RR',
        #        'Systolic.BP', 'Temperature']

        # ['SEX', 'AGE', 'AKI_on_CKD', 'CKD', 'AKI','Pregnancy/post_delivery','Heamatemesis',	'PHTN',	
        #          'Malena/Rectal_bleed', 'Heamoptysis',	'Pneumonia/infective_etio','Bronchial_Asthama','COPD',
        #          'ARDS','Respiratory_failure_type2','Respiratory_failure_type1', 'ECHO_abnormalties',	
        #          'Coronary_artery_disease','Valvular_heart_disease',	'Embolism',	'IC_bleed',	'Infarct',	
        #          'HR_max','HR_min', 'HR_spot','BP_max', 'BP_min',	'BP_spot',	'SPO2_max',	'SPO2_min',	'SPO2_spot', 
        #          'RR_max', 'RR_min', 'RR_spot',	'GCS','Eye_opening', 'Verbal_response',	'Motor_response','Fever',	
        #          'SOB',	'Airway','Perfusion', 'Ascitis', 'Effusion/pleural/pericardial',	'Chest_Pain', 'Lactate',
        #          'PCO2','APACHE_II','qSOFA','Urine_Output',	'Procalcitonin','PF_ratio',	'Ketoacidosis',	'Met_Acidosis',
        #          'Resp_Acidosis', 'Cultures', 'Hb',	'TLC',	'PLats','Urea',	'Creatinine', 'Sodium',	'Pottasium', 
        #          'Albumin',	'Bil',	'AST',	'ALT',	'ALP']  

        #len = 68 column

        if st.checkbox("Do you have patient Age?", False):
            AGE = st.number_input('AGE:', min_value=data.describe()["AGE"].loc["min"],
                                        max_value=data.describe()["AGE"].loc["max"],
                                        value=data.describe()["AGE"].loc["50%"])
        else:
            AGE = np.nan

        
        
        SEX = st.selectbox('SEX:', ["Not Available", "Male", "Female"])
        AKI_on_CKD = st.selectbox('AKI_on_CKD:', ["Not Available", "No", "Yes"])
        CKD = st.selectbox('CKD:', ["Not Available", "No", "Yes"])
        AKI = st.selectbox('AKI:', ["Not Available", "No", "Yes"])
        Pregnancy_post_delivery = st.selectbox('Pregnancy/post_delivery:', ["Not Available", "No", "Yes"])
        Heamatemesis = st.selectbox('Heamatemesis:', ["Not Available", "No", "Yes"])
        PHTN = st.selectbox('PHTN:', ["Not Available", "No", "Yes"])
        Malena_Rectal_bleed = st.selectbox('Malena/Rectal_bleed:', ["Not Available", "No", "Yes"])
        Heamoptysis = st.selectbox('Heamoptysis:', ["Not Available", "No", "Yes"])
        Pneumonia_infective_etio = st.selectbox('Pneumonia/infective_etio:', ["Not Available", "No", "Yes"])
        Bronchial_Asthama = st.selectbox('Bronchial_Asthama:', ["Not Available", "No", "Yes"])
        COPD = st.selectbox('COPD:', ["Not Available", "No", "Yes"])
        ARDS = st.selectbox('ARDS:', ["Not Available", "No", "Yes"])
        Respiratory_failure_type2 = st.selectbox('Respiratory_failure_type2:', ["Not Available", "No", "Yes"])
        Respiratory_failure_type1 = st.selectbox('Respiratory_failure_type1:', ["Not Available", "No", "Yes"])
        ECHO_abnormalties = st.selectbox('ECHO_abnormalties:', ["Not Available", "No", "Yes"])
        Coronary_artery_disease = st.selectbox('Coronary_artery_disease:', ["Not Available", "No", "Yes"])
        Valvular_heart_disease = st.selectbox('Valvular_heart_disease:', ["Not Available", "No", "Yes"])
        Embolism = st.selectbox('Embolism:', ["Not Available", "No", "Yes"])
        IC_bleed = st.selectbox('IC_bleed:', ["Not Available", "No", "Yes"])
        Infarct = st.selectbox('Infarct:', ["Not Available", "No", "Yes"])
        Fever = st.selectbox('Fever:', ["Not Available", "No", "Yes"])
        SOB = st.selectbox('SOB:', ["Not Available", "No", "Yes"])
        Airway = st.selectbox('Airway:', ["Not Available", "No", "Yes"])
        Perfusion = st.selectbox('Perfusion:', ["Not Available", "No", "Yes"])
        Ascitis= st.selectbox('Ascitis:', ["Not Available", "No", "Yes"])
        Effusion_pleural_pericardial = st.selectbox('Effusion_pleural_pericardial:', ["Not Available", "No", "Yes"])
        Chest_Pain = st.selectbox('Chest_Pain:', ["Not Available", "No", "Yes"])
        Lactate = st.selectbox('Lactate:', ["Not Available", "No", "Yes"])
        PCO2 = st.selectbox('PCO2:', ["Not Available", "No", "Yes"])
        qSOFA = st.selectbox('qSOFA:', ["Not Available", "No", "Yes"])
        PF_ratio = st.selectbox('PF_ratio:', ["Not Available", "No", "Yes"])
        Urine_Output = st.selectbox('Urine_Output:', ["Not Available", "No", "Yes"])
        Ketoacidosis = st.selectbox('Ketoacidosis:', ["Not Available", "No", "Yes"])
        Met_Acidosis = st.selectbox('Met_Acidosis:', ["Not Available", "No", "Yes"])
        Resp_Acidosis = st.selectbox('Resp_Acidosis:', ["Not Available", "No", "Yes"])
        Cultures = st.selectbox('Cultures:', ["Not Available", "No", "Yes"])
        

        # 37 yes, no  and 31 values = total 68
        
        if st.checkbox("Do you have HR_max?", False):
            HR_max = st.number_input('HR_max:', min_value=data.describe()["HR_max"].loc["min"],
                                        max_value=data.describe()["HR_max"].loc["max"],
                                        value=data.describe()["HR_max"].loc["50%"])
        else:
            HR_max = np.nan

        if st.checkbox("Do you have HR_min?", False):
            HR_min = st.number_input('HR_min:', min_value=data.describe()["HR_min"].loc["min"],
                                    max_value=data.describe()["HR_min"].loc["max"],
                                    value=data.describe()["HR_min"].loc["50%"])
        else:
            HR_min = np.nan
        if st.checkbox("Do you have patient HR_spot?", False):
            HR_spot = st.number_input('HR_spot:', min_value=data.describe()["HR_spot"].loc["min"],
                                        max_value=data.describe()["HR_spot"].loc["max"],
                                        value=data.describe()["HR_spot"].loc["50%"])
        else:
            HR_spot = np.nan

        if st.checkbox("Do you have BP_max?", False):
            BP_max = st.number_input('BP_max:', min_value=data.describe()["BP_max"].loc["min"],
                                        max_value=data.describe()["BP_max"].loc["max"],
                                        value=data.describe()["BP_max"].loc["50%"])
        else:
            BP_max = np.nan

        if st.checkbox("Do you have BP_min?", False):
            BP_min = st.number_input('BP_min:', min_value=data.describe()["BP_min"].loc["min"],
                                    max_value=data.describe()["BP_min"].loc["max"],
                                    value=data.describe()["BP_min"].loc["50%"])
        else:
            BP_min = np.nan
        if st.checkbox("Do you have patient BP_spot?", False):
            BP_spot = st.number_input('BP_spot:', min_value=data.describe()["BP_spot"].loc["min"],
                                        max_value=data.describe()["BP_spot"].loc["max"],
                                        value=data.describe()["BP_spot"].loc["50%"])
        else:
            BP_spot = np.nan   


        if st.checkbox("Do you have SPO2_max?", False):
            SPO2_max = st.number_input('SPO2_max:', min_value=data.describe()["SPO2_max"].loc["min"],
                                        max_value=data.describe()["SPO2_max"].loc["max"],
                                        value=data.describe()["SPO2_max"].loc["50%"])
        else:
            SPO2_max = np.nan

        if st.checkbox("Do you have SPO2_min?", False):
            SPO2_min = st.number_input('SPO2_min:', min_value=data.describe()["SPO2_min"].loc["min"],
                                    max_value=data.describe()["SPO2_min"].loc["max"],
                                    value=data.describe()["SPO2_min"].loc["50%"])
        else:
            SPO2_min = np.nan
        if st.checkbox("Do you have patient SPO2_spot?", False):
            SPO2_spot = st.number_input('SPO2_spot:', min_value=data.describe()["SPO2_spot"].loc["min"],
                                        max_value=data.describe()["SPO2_spot"].loc["max"],
                                        value=data.describe()["SPO2_spot"].loc["50%"])
        else:
            SPO2_spot = np.nan    

        if st.checkbox("Do you have RR_max?", False):
            RR_max = st.number_input('RR_max:', min_value=data.describe()["RR_max"].loc["min"],
                                        max_value=data.describe()["RR_max"].loc["max"],
                                        value=data.describe()["RR_max"].loc["50%"])
        else:
            RR_max = np.nan

        if st.checkbox("Do you have RR_min?", False):
            RR_min = st.number_input('RR_min:', min_value=data.describe()["RR_min"].loc["min"],
                                    max_value=data.describe()["RR_min"].loc["max"],
                                    value=data.describe()["RR_min"].loc["50%"])
        else:
            RR_min = np.nan

        if st.checkbox("Do you have patient RR_spot?", False):
            RR_spot = st.number_input('RR_spot:', min_value=data.describe()["RR_spot"].loc["min"],
                                        max_value=data.describe()["RR_spot"].loc["max"],
                                        value=data.describe()["RR_spot"].loc["50%"])
        else:
            RR_spot = np.nan     

        if st.checkbox("Do you have patient GCS?", False):
            GCS = st.number_input('GCS:', min_value=data.describe()["GCS"].loc["min"],
                                    max_value=data.describe()["GCS"].loc["max"],
                                    value=data.describe()["GCS"].loc["50%"])
        else:
            GCS = np.nan

        if st.checkbox("Do you have patient Eye_opening?", False):
            Eye_opening = st.number_input('Eye_opening:', min_value=data.describe()["Eye_opening"].loc["min"],
                                max_value=data.describe()["Eye_opening"].loc["max"],
                                value=data.describe()["Eye_opening"].loc["50%"])
        else:
            Eye_opening = np.nan

        if st.checkbox("Do you have patient Verbal_response?", False):
            Verbal_response = st.number_input('Verbal_response:', min_value=data.describe()["Verbal_response"].loc["min"],
                                max_value=data.describe()["Verbal_response"].loc["max"],
                                value=data.describe()["Verbal_response"].loc["50%"])
        else:
            Verbal_response = np.nan

        if st.checkbox("Do you have patient Motor_response?", False):
            Motor_response = st.number_input('Motor_response:', min_value=data.describe()["Motor_response"].loc["min"],
                                max_value=data.describe()["Motor_response"].loc["max"],
                                value=data.describe()["Motor_response"].loc["50%"])
        else:
            Motor_response = np.nan

        if st.checkbox("Do you have patient APACHE_II?", False):
            APACHE_II = st.number_input('APACHE_II:', min_value=data.describe()["Hb"].loc["min"],
                        max_value=data.describe()["APACHE_II"].loc["max"],
                        value=data.describe()["APACHE_II"].loc["50%"])
        else:
            APACHE_II = np.nan

        if st.checkbox("Do you have patient Procalcitonin?", False):
            Procalcitonin = st.number_input('Procalcitonin:', min_value=data.describe()["Procalcitonin"].loc["min"],
                                        max_value=data.describe()["Procalcitonin"].loc["max"],
                                        value=data.describe()["Procalcitonin"].loc["50%"])
        else:
            Procalcitonin = np.nan    
    
            
        if st.checkbox("Do you have patient Hb?", False):
            Hb = st.number_input('Hb:', min_value=data.describe()["Hb"].loc["min"],
                        max_value=data.describe()["Hb"].loc["max"],
                        value=data.describe()["Hb"].loc["50%"])
        else:
            Hb = np.nan

        if st.checkbox("Do you have patient TLC?", False):
            TLC = st.number_input('TLC:', min_value=data.describe()["TLC"].loc["min"],
                                max_value=data.describe()["TLC"].loc["max"],
                                value=data.describe()["TLC"].loc["50%"])
        else:
            TLC = np.nan

        if st.checkbox("Do you have patient PLats?", False):
            PLats = st.number_input('PLats:', min_value=data.describe()["PLats"].loc["min"],
                                max_value=data.describe()["PLats"].loc["max"],
                                value=data.describe()["PLats"].loc["50%"])
        else:
            PLats = np.nan

        if st.checkbox("Do you have patient Urea?", False):
            Urea = st.number_input('Urea:', min_value=data.describe()["Urea"].loc["min"],
                                max_value=data.describe()["Urea"].loc["max"],
                                value=data.describe()["Urea"].loc["50%"])
        else:
            Urea = np.nan

        if st.checkbox("Do you have patient Creatinine?", False):
            Creatinine = st.number_input('Creatinine:', min_value=data.describe()["Creatinine"].loc["min"],
                                max_value=data.describe()["Creatinine"].loc["max"],
                                value=data.describe()["Creatinine"].loc["50%"])
        else:
            Creatinine = np.nan

        if st.checkbox("Do you have patient Sodium?", False):
            Sodium = st.number_input('Sodium:', min_value=data.describe()["Sodium"].loc["min"],
                                max_value=data.describe()["Sodium"].loc["max"],
                                value=data.describe()["Sodium"].loc["50%"])
        else:
            Sodium = np.nan

        if st.checkbox("Do you have patient Potassium?", False):
            Potassium = st.number_input('Potassium:', min_value=data.describe()["Potassium"].loc["min"],
                                max_value=data.describe()["Potassium"].loc["max"],
                                value=data.describe()["Potassium"].loc["50%"])
        else:
            Potassium = np.nan

        if st.checkbox("Do you have patient Albumin?", False):
            Albumin = st.number_input('Albumin:', min_value=data.describe()["Albumin"].loc["min"],
                                max_value=data.describe()["Albumin"].loc["max"],
                                value=data.describe()["Albumin"].loc["50%"])
        else:
            Albumin = np.nan

        if st.checkbox("Do you have patient Bil?", False):
            Bil = st.number_input('Bil:', min_value=data.describe()["Bil"].loc["min"],
                                max_value=data.describe()["Bil"].loc["max"],
                                value=data.describe()["Bil"].loc["50%"])
        else:
            Bil = np.nan

        if st.checkbox("Do you have patient AST?", False):
            AST = st.number_input('AST:', min_value=data.describe()["AST"].loc["min"],
                                max_value=data.describe()["AST"].loc["max"],
                                value=data.describe()["AST"].loc["50%"])
        else:
            AST = np.nan

        if st.checkbox("Do you have patient ALT?", False):
            ALT = st.number_input('ALT:', min_value=data.describe()["ALT"].loc["min"],
                                max_value=data.describe()["ALT"].loc["max"],
                                value=data.describe()["ALT"].loc["50%"])
        else:
            ALT = np.nan

        if st.checkbox("Do you have patient ALP?", False):
            ALP = st.number_input('ALP:', min_value=data.describe()["ALP"].loc["min"],
                                max_value=data.describe()["ALP"].loc["max"],
                                value=data.describe()["ALP"].loc["50%"])
        else:
            ALP = np.nan

            




#         if st.checkbox("Do you have patient Lymphocyte?", False):
#             Lymphocyte = st.number_input('Lymphocyte:', min_value=data.describe()["Lymphocyte"].loc["min"],
#                                         max_value=data.describe()["Lymphocyte"].loc["max"],
#                                         value=data.describe()["Lymphocyte"].loc["50%"])
#         else:
#             Lymphocyte = np.nan

#         if st.checkbox("Do you have patient SpO2?", False):
#             SpO2 = st.number_input('SpO2:', min_value=data.describe()["SpO2"].loc["min"],
#                                         max_value=data.describe()["SpO2"].loc["max"],
#                                         value=data.describe()["SpO2"].loc["50%"])
#         else:
#             SpO2 = np.nan

#         if st.checkbox("Do you have patient Procalcitonin?", False):
#             Procalcitonin = st.number_input('Procalcitonin:', min_value=data.describe()["Procalcitonin"].loc["min"],
#                                         max_value=data.describe()["Procalcitonin"].loc["max"],
#                                         value=data.describe()["Procalcitonin"].loc["50%"])
#         else:
#             Procalcitonin = np.nan

#         if st.checkbox("Do you have patient RR?", False):
#             RR = st.number_input('RR:', min_value=data.describe()["RR"].loc["min"],
#                                         max_value=data.describe()["RR"].loc["max"],
#                                         value=data.describe()["RR"].loc["50%"])
#         else:
#             RR = np.nan

#         if st.checkbox("Do you have patient Systolic BP?", False):
#             Systolic_BP = st.number_input('Systolic BP:', min_value=data.describe()["Systolic.BP"].loc["min"],
#                                         max_value=data.describe()["Systolic.BP"].loc["max"],
#                                         value=data.describe()["Systolic.BP"].loc["50%"])
#         else:
#             Systolic_BP = np.nan

#         if st.checkbox("Do you have patient Temperature?", False):
#             Temperature = st.number_input('Temperature:', min_value=data.describe()["Temperature"].loc["min"],
#                                           max_value=data.describe()["Temperature"].loc["max"],
#                                           value=data.describe()["Temperature"].loc["50%"])
#         else:
#             Temperature = np.nan

#         output = ""

        input_dict = {
            'SEX': cat_map[SEX],
            'AKI_on_CKD': cat_map[AKI_on_CKD],
            'CKD': cat_map[CKD],
            'AKI': cat_map[AKI],
            'Pregnancy_post_delivery':cat_map[Pregnancy_post_delivery],
            'Heamatemesis': cat_map[Heamatemesis],	
            'PHTN': cat_map[PHTN],
            'Malena_Rectal_bleed': cat_map[Malena_Rectal_bleed], 
            'Heamoptysis': cat_map[Heamoptysis],	
            'Pneumonia_infective_etio': cat_map[Pneumonia_infective_etio],
            'Bronchial_Asthama': cat_map[Bronchial_Asthama],
            'COPD': cat_map[COPD],
            'ARDS': cat_map[ARDS],
            'Respiratory_failure_type2': cat_map[Respiratory_failure_type2],
            'Respiratory_failure_type1': cat_map[Respiratory_failure_type1], 
            'ECHO_abnormalties': cat_map[ECHO_abnormalties],	
            'Coronary_artery_disease': cat_map[Coronary_artery_disease],
            'Valvular_heart_disease': cat_map[Valvular_heart_disease],	
            'Embolism': cat_map[Embolism],	
            'IC_bleed': cat_map[IC_bleed],	
            'Infarct': cat_map[Infarct],
            'Fever': cat_map[Fever],
            'SOB' :	cat_map[SOB],
            'Airway':cat_map[Airway],
            'Perfusion':cat_map[Perfusion],
            'Ascitis':cat_map[Ascitis],
            'Effusion_pleural_pericardial':cat_map[Effusion_pleural_pericardial], 
            'Chest_Pain': cat_map[Chest_Pain],
            'Lactate': cat_map[Lactate],
            'PCO2': cat_map[PCO2],
            'qSOFA': cat_map[qSOFA],
            'Urine_Output': cat_map[Urine_Output],
            'PF_ratio': cat_map[PF_ratio],
            'Ketoacidosis': cat_map[Ketoacidosis],
            'Met_Acidosis': cat_map[Met_Acidosis],
            'Resp_Acidosis': cat_map[Resp_Acidosis],
            'Cultures': cat_map[Cultures],


            
            'AGE': AGE,
            'HR_max': HR_max ,
            'HR_min': HR_min, 
            'HR_spot':HR_spot,
            'BP_max': BP_max, 
            'BP_min': BP_min,	
            'BP_spot': BP_spot,	
            'SPO2_max': SPO2_max,	
            'SPO2_min': SPO2_min,	
            'SPO2_spot': SPO2_spot, 
            'RR_max': RR_max, 
            'RR_min': RR_min, 
            'RR_spot': RR_spot,	
            'GCS': GCS,
            'APACHE_II' : APACHE_II,
            'Eye_opening': Eye_opening, 
            'Verbal_response': Verbal_response,	
            'Motor_response': Motor_response,
            'Procalcitonin': Procalcitonin,
            'Hb' : Hb,
            'TLC': TLC,	
            'PLats': PLats,
            'Urea': Urea,	
            'Creatinine': Creatinine, 
            'Sodium': Sodium,	
            'Pottasium': Potassium, 
            'Albumin': Albumin,	
            'Bil': Bil,	
            'AST': AST,	
            'ALT': ALT,	
            'ALP': ALP

        }
#             'SOB': cat_map[SOB],
#             'Fatigue': cat_map[Fatigue],
#             'Sputum': cat_map[Sputum],
#             'Myalgia': cat_map[Myalgia],
#             'Diarrhea': cat_map[Diarrhea],
#             'Nausea.Vomiting': cat_map[Nausea],
#             'Sore.throat': cat_map[Sore_throat],
#             'Chest.discomfort..chest.pain': cat_map[Chest_discomfort_chest_pain],
#             'smoking_history': cat_map[smoking_history],
#             'hypertensionhx': cat_map[hypertensionhx],
#             'diabeteshx': cat_map[diabeteshx],
#             'coronaryheartdiseasehx': cat_map[coronaryheartdiseasehx],
#             'copdhx': cat_map[copdhx],
#             'carcinomahx': cat_map[carcinomahx],
#             'ckdhx': cat_map[ckdhx],
#             'ALT': ALT,
#             'HR': HR,
#             'Lymphocyte': Lymphocyte,
#             'SpO2': SpO2,
#             'Procalcitonin': Procalcitonin,
#             'RR': RR,
#             'Systolic.BP': Systolic_BP,
#             'Temperature': Temperature
#         }


        input_df = pd.DataFrame([input_dict])
        if st.button("Predict"):
            output, confidence = predict(model=model, input_df=input_df)
            print(output)
            if output == 1:
                st.warning(f"⚠️ Patient need to be in ICU !!! (Confidence = {confidence*100} %)", )
            else:
                st.success(f"✅ Patient is fine, not recommended for ICU !!! (Confidence = {confidence*100} %)")


if __name__ == '__main__':
    main()