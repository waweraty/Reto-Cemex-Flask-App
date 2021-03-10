from flask import Flask 
from flask import Flask, request, render_template
import pickle5 as pickle
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
import RegresorClass
from RegresorClass import Regresor

#TEMPLATE = '/templates'
#STATIC = '/static'

class CustomUnpickler(pickle.Unpickler):

    def find_class(self, module, name):
        try:
            return super().find_class(__name__, name)
        except AttributeError:
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
     app.run(host="127.0.0.1",port=8080)