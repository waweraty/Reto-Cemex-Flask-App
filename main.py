from flask import Flask 
from flask import Flask, request, render_template
import pickle5 as pickle
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from RegresorClass import Regresor

#TEMPLATE = '/templates'
#STATIC = '/static'

#, template_folder=TEMPLATE, static_folder=STATIC
app = Flask(__name__)

MODEL_PATH = os.path.join(os.getcwd(), "ObjectFile.picl")  
# set path to the model
model = pickle.load(open(MODEL_PATH, 'rb'))
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