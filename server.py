from apiflask import APIFlask, Schema
from apiflask.fields import Integer, Float
from flask import request
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np

def lire_fichier_csv(chemin_fichier, skip_header=True):
    with open(chemin_fichier) as fic:
        lines = fic.readlines()
        data_csv = [line.strip().split(";") for line in lines]
        return data_csv[1:] if skip_header else data_csv

def convert_grav(val):
    return {"1": 1, "2": 100, "3": 10, "4": 5}.get(val, -1)

def convert_annee(val):
    return int(val) if len(val) == 4 else -1

data_usagers = lire_fichier_csv("data/usagers-2023.csv")
data_usagers = [d for d in data_usagers if len(d) > 8]
xy = [
    [convert_annee(d[8][1:-1]), convert_grav(d[6][1:-1])]
    for d in data_usagers
]
xy = [d for d in xy if d[0] > -1 and d[1] > -1]
x_annee = [d[0] for d in xy]
y_gravite = [d[1] for d in xy]

x_train, x_test, y_train, y_test = train_test_split(
    x_annee, y_gravite, test_size=0.2, random_state=42
)
x_train_arr = np.array(x_train).reshape(-1, 1)
model = LinearRegression()
model.fit(x_train_arr, y_train)

app = APIFlask("GraviteAPI")

class PredictionOut(Schema):
    annee = Integer()
    gravite_predite = Float()

@app.get("/")
def hello():
    return "<h3>Bienvenue sur l'API Gravité</h3>"

@app.get("/sample")
def get_sample():
    return [{"annee": a, "gravite": g} for a, g in zip(x_annee[:10], y_gravite[:10])]

@app.get("/predict")
@app.output(PredictionOut)
@app.doc(description="Prédire la gravité via un paramètre dans l'URL. Exemple : /predict?annee=2023")
def predict():
    annee = request.args.get("annee", type=int)
    if annee is None:
        return {"error": "Paramètre 'annee' requis"}, 400
    pred = model.predict(np.array([[annee]]))[0]
    return {"annee": annee, "gravite_predite": round(pred, 2)}

if __name__ == "__main__":
    app.run(port=9090, debug=True)
