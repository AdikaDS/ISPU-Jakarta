import pandas as pd
import joblib
import os
from flask import Flask, redirect, url_for, request, render_template

app = Flask(__name__)
#load index.html/ first page. receive input variable from user
@app.route("/ispa/")
def index():
	return render_template('index.html')

#load result.html. the result of prediction is presented here. 
@app.route('/ispa/result/', methods=["POST"])
def prediction_result():
    #receiving parameters sent by client
    pm_duakomalima = request.form.get('inputpm25')
    pm_sepuluh = request.form.get('inputpm10')
    sulfur_dioksida = request.form.get('inputso2')
    karbon_monoksida = request.form.get('inputco')
    ozon = request.form.get('inputo3')
    nitrogen_dioksida = request.form.get('inputno2')

    #load the trained model.
    current_directory = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(current_directory, 'model.pkl')
    loaded_model= joblib.load(filename)
    #create new dataframe
    df_input = pd.DataFrame(columns = ['pm_duakomalima', 'pm_sepuluh', 'sulfur_dioksida', 'karbon_monoksida', 'ozon', 'nitrogen_dioksida'])
    df_input.loc[0] = [pm_duakomalima, pm_sepuluh, sulfur_dioksida, karbon_monoksida, ozon, nitrogen_dioksida]
    # print(df_input)
    result = loaded_model.predict(df_input)
    #print(result)
    for i in result:
        int_result = int(i)
        if(int_result==0):
            decision='BAIK'
        elif (int_result==1):
            decision='SEDANG'
        elif (int_result==2):
            decision='TIDAK SEHAT'
        elif (int_result==3):
            decision='SANGAT TIDAK SEHAT'
        elif (int_result==4):
            decision='BERBAHAYA'
        else:
            decision='Not defined'
    print('Level kualitas udara adalah ', decision)
    #return the output and load result.html
    return render_template('result.html', pm_duakomalima=pm_duakomalima, pm_sepuluh=pm_sepuluh, sulfur_dioksida=sulfur_dioksida, 
                           karbon_monoksida=karbon_monoksida, ozon=ozon, nitrogen_dioksida=nitrogen_dioksida, status=decision)

if __name__ == "__main__":
    #host= ip address, port = port number
    #app.run(host='127.0.0.1', port='5001')
    app.run()
