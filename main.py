import streamlit as st
import pickle as pickle
import pandas as pd
import plotly.graph_objects as go
import numpy as np



# importation des données 
def get_data_clean():
    # importtation des donnéées 
    df = pd.read_csv("diabetes.csv", sep=";")
    
    return df


def add_sidebar():
  st.sidebar.header("Données Demographiques et Composantes Sanguines")
  
  df = get_data_clean()
  
  slider_labels = [
        ("Nombre de grossese", "nb_grossesse"),
        ("Glucose", "glucose"),
        ("Pression Arterielle", "pression_art"),
        ("Triceps", "triceps"),
        ("Insuline", "insuline"),
        ("Indice de masse corporelle", "imc"),
        ("Pedigre", "pedigree"),
        ("Age du patient", "age"),

    ]

  input_dict = {}

  for label, key in slider_labels:
    input_dict[key] = st.sidebar.slider(
      label,
      min_value=float(0),
      max_value=float(df[key].max()),
      value=float(df[key].mean())
    )
    
  return input_dict

##### standardisation des donnéées pour la prediction 

def get_scaled_values(input_dict):
  df = get_data_clean()
  
  X = df.drop(['diabete'], axis=1)
  
  scaled_dict = {}
  
  for key, value in input_dict.items():
    max_val = X[key].max()
    min_val = X[key].min()
    scaled_value = (value - min_val) / (max_val - min_val)
    scaled_dict[key] = scaled_value
  
  return scaled_dict


######

def get_radar_chart(input_data):
  
  input_data = get_scaled_values(input_data)
  
  categories = ['Nombre de grossese', 'Glucose','pression_art', 'Triceps', 'Insuline', 
                'Indice de masse corporelle', 'Pedigre', 
                'Age du patient']

  fig = go.Figure()


  fig.add_trace(go.Scatterpolar(
        r=[
          input_data['nb_grossesse'], input_data['glucose'],input_data['pression_art'], input_data['triceps'],
          input_data['insuline'], input_data['imc'], input_data['pedigree'],
          input_data['age']
        ],
        theta=categories,
        fill='toself',
        name='Mean Value'
  ))



  fig.update_layout(
    polar=dict(
      radialaxis=dict(
        visible=True,
        range=[0, 1]
      )),
    showlegend=True
  )
  
  return fig


def add_predictions(input_data):
  model = pickle.load(open("C:/Users/anass/Desktop/Portofolio/diabete/model.pkl", "rb"))
  scaler = pickle.load(open("C:/Users/anass/Desktop/Portofolio/diabete//scaler.pkl", "rb"))
  
  input_array = np.array(list(input_data.values())).reshape(1, -1)
  
  input_array_scaled = scaler.transform(input_array)
  
  prediction = model.predict(input_array_scaled)
  
  st.subheader(":blue[Prediction]")
  st.write("Le patient est diagnostiqué :")
  
  if prediction[0] == 0:
    st.write("<span class='diabete positif'>Diabetique</span>", unsafe_allow_html=True)
  else:
    st.write("<span class='diabete negatif'>Non Diabetique</span>", unsafe_allow_html=True)
    
  
  st.write("Avec une probabilité d'être diabetique de : ", model.predict_proba(input_array_scaled)[0][0])
  st.write("Et une probabilité d'etre non diabetique de : ", model.predict_proba(input_array_scaled)[0][1])
  
  st.write("cette application ne remplace en aucun cas l'avis d'un professionel de la santé")





def main():
  st.set_page_config(
    page_title="Prediction du diabete",
    page_icon=":Slack:",
    layout="wide",
    initial_sidebar_state="expanded"
  )
  
  with open("C:/Users/anass/Desktop/Portofolio/diabete/style.css") as f:
    st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)

  
  input_data = add_sidebar()
  
  with st.container():
    st.title(":blue[DiabetePredict]")
    st.write("DiabetePredict est une application web conviviale conçue pour aider les professionnels de la santé à prédire la probabilité qu'un individu soit atteint de diabète en se basant sur un ensemble de facteurs de risque spécifiques. Utilisant des techniques avancées d'apprentissage automatique, l'application analyse quelques données démographiques et les résultats de tests sanguins pour générer une estimation de la probabilité de diabete. ")
  
  col1, col2 = st.columns([4,1])
  
  with col1:
    radar_chart = get_radar_chart(input_data)
    st.plotly_chart(radar_chart)
  with col2:
    add_predictions(input_data)

 
if __name__ == '__main__':
  main()