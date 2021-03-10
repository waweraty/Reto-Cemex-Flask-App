from flask import Flask 
from flask import Flask, request, render_template
import pickle5 as pickle
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

#TEMPLATE = '/templates'
#STATIC = '/static'

class Regresor:
    def __init__(self, regresor):
        self.regressor = regresor
        
    def calculaOptimo(self, datos_entrada):
        datos = np.array(datos_entrada)
        datos = datos.reshape(1, -1)
        datos_salida = self.regressor.predict(datos)
        return datos_salida
    
    def imprimeOptimo(self, datos_entrada):
        datos = np.array(datos_entrada)
        datos = datos.reshape(1, -1)
        EE = self.regressor.predict(datos)[0][0]
        EC = self.regressor.predict(datos)[0][1]
        cost = self.regressor.predict(datos)[0][2]
        print('EE: %s\nEC: %s\nCosto Ponderado unitario: %s'% (EE,EC,cost))

class CustomUnpickler(pickle.Unpickler):

    def find_class(self, module, name):
        if name == 'Manager':
            from settings import Manager
            return Manager
        return super().find_class(module, name)

#, template_folder=TEMPLATE, static_folder=STATIC
app = Flask(__name__)
ROOT=os.path.dirname(os.path.abspath("ObjectFile.picl"))

MODEL_PATH = os.path.join(ROOT, "ObjectFile.picl")  
# set path to the model
#model = pickle.load(open(MODEL_PATH, 'rb'))
model = CustomUnpickler(open("ObjectFile.picl", 'rb')).load()
# load the pickled model

@app.route("/", methods=['GET', 'POST'])                        
def index():
     return render_template('index.html')

@app.route("/dashboard", methods=['GET', 'POST'])                        
def dashboard():
     return render_template('dashboard.html')

@app.route('/submit', methods=['GET', 'POST'])  
def make_prediction():
  features = [float(x) for x in request.form.values()]
  final_features = [features]       
  prediction = model.calculaOptimo(final_features)
  pee=round(prediction[0][0],4)
  pec=round(prediction[0][1],4)
  pcpu=round(prediction[0][2],4)
  return render_template('predictionpro.html', pee=pee,pec=pec,pcpu=pcpu,c=final_features[0][0],d=final_features[0][1],tp=final_features[0][2])

if __name__ == '__main__':
     app.run(host="127.0.0.1",port=8080,debug=True)