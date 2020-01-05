import os
import pickle
import flask
import numpy as np
from flask import Flask,render_template,request
from random import randint
#Creating instance of the class
app = Flask(__name__)

#to tell the flask what url should trigger the function index
@app.route('/')
@app.route('/index')
def index():
    return flask.render_template('index.html', rand=randint(0,194893))

def ValuePredictor(to_predict_list): 
    to_predict = np.array(to_predict_list).reshape(1, 12) 
    loaded_model = pickle.load(open("model.pkl", "rb")) 
    result = loaded_model.predict(to_predict) 
    return result[0] 
  
@app.route('/result', methods = ['POST']) 
def result(): 
    if request.method == 'POST': 
        to_predict_list = request.form.to_dict() 
        to_predict_list = list(to_predict_list.values()) 
        to_predict_list = list(map(int, to_predict_list)) 
        result = ValuePredictor(to_predict_list)         
        if int(result)== 1: 
            prediction ='Income more than 50K'
        else: 
            prediction ='Income less than 50K'            
        return render_template("result.html", prediction = prediction) 
if __name__=="__main__":
    app.run(debug=True,host='0.0.0.0', port=5003)
    
    
