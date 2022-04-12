import pandas as pd
import os
import numpy as np
from flask import Flask, render_template, request
app = Flask(__name__)
import pickle
model = pickle.load(open('model.pkl','rb'))


@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict',methods=["POST","GET"])# route to show the predictions in a web UI
def predict():
    #  reading the inputs given by the user
    input_feature=[x for x in request.form.values() ]  
    features_values=[np.array(input_feature)]
    names = [['Housing', 'Sex', 'Saving account', 'Checking account','Age','Job',
       'Credit amount', 'Duration', 'Purpose']]
    data = pd.DataFrame(features_values,columns=names)
     # predictions using the loaded model file
    prediction=model.predict(data)
    if (prediction == 1):
         return render_template("predgood.html")
    else:
         return render_template("predbad.html")
     # showing the prediction results in a UI
if __name__=="__main__":
    
    # app.run(host='0.0.0.0', port=8000,debug=True)    # running the app
    port=int(os.environ.get('PORT',8080))
    app.run(port=port,debug=True,use_reloader=False)