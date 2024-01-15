import datetime
import pickle
from flask import Flask, render_template, request,jsonify
from flask_cors import CORS, cross_origin
import pandas as pd
import joblib
import numpy as np

app = Flask(__name__)
cors = CORS(app)
car = pd.read_csv("C:/Users/ayari/NoteBook_Py/Py_Ensi/PS_Tayara/Cleaned_voitures.csv")

# Charger votre modèle de machine learning entraîné
with open('C:/Users/ayari/NoteBook_Py/Py_Ensi/PS_Tayara/RandomForestRegressorModel.pkl', 'rb') as file:
    model = pickle.load(file)
@app.route('/', methods=['GET', 'POST'])
def index():
    Marque_voiture = sorted(car['Marque_voiture'].unique())
    Modele = sorted(car['Modele'].unique())
    Cylindre = sorted(car['Cylindre'].unique())
    Annee = sorted(car['Annee'].unique(), reverse=True)
    carburant = sorted(car['Carburant'].unique())
    Puissance = sorted(car['Puissance'].unique())
    Couleur = sorted(car['Couleur'].unique())
    TypeBoite = sorted(car['TypeBoite'].unique())
    Marque_voiture.insert(0, 'Select Marque_voiture')

    # Render le modèle sans prédiction si le formulaire n'est pas soumis
    return render_template('index.html', Puissance=Puissance, Marque_voiture=Marque_voiture, Modele=Modele,
                           Cylindre=Cylindre, Annee=Annee, carburant=carburant, years=Annee, TypeBoite=TypeBoite, Couleur=Couleur)

@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    Marque_voiture = request.form.get('Marque_voiture')
    Modele = request.form.get('Modele')
    Annee = request.form.get('Annee')
    Carburant = request.form.get('Carburant')
    Cylindre = request.form.get('Cylindre')
    Puissance = request.form.get('Puissance')
    Kilometrage = request.form.get('Kilometrage')
    Couleur = request.form.get('Couleur')
    TypeBoite = request.form.get('TypeBoite')
    Age = request.form.get('Age')

    try:
        Annee = int(Annee)
        Cylindre = float(Cylindre)
        Puissance = int(Puissance)
        Kilometrage = int(Kilometrage)
        #Age = int(Age)
        current_year = datetime.datetime.now().year
        Age = int(current_year) - Annee
    except ValueError:
        return "Veuillez entrer des valeurs numériques valides pour les champs requis."

    # Préparer les données pour la prédiction
    input_data = pd.DataFrame(data=np.array([Marque_voiture, Modele, Cylindre, Annee, Carburant, Puissance, Kilometrage, Age, Couleur, TypeBoite]).reshape(1, -1),
                              columns=['Marque_voiture', 'Modele', 'Cylindre', 'Annee', 'Carburant', 'Puissance', 'Kilometrage', 'Age', 'Couleur', 'TypeBoite'])

    print(input_data)

    prediction = model.predict(input_data)

    # Return the predicted price as JSON
    return jsonify({'prediction': np.round(prediction[0], 2)})

if __name__ == "__main__":
    app.run(debug=True)
