import pandas as pd
import joblib
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
    pm_duakomalima = request.form.get('pm_duakomalima')
    pm_sepuluh = request.form.get('pm_sepuluh')
    sulfur_dioksida = request.form.get('sulfur_dioksida')
    karbon_monoksida = request.form.get('karbon_monoksida')
    ozon = request.form.get('ozon')
    nitrogen_dioksida = request.form.get('nitrogen_dioksida')

    #load the trained model.
    filename = 'dt_model.model'
    loaded_model= joblib.load(filename)
    #create new dataframe
    df_input = pd.DataFrame(columns = ['pm_duakomalima', 'pm_sepuluh', 'sulfur_dioksida', 'karbon_monoksida', 'ozon', 'nitrogen_dioksida'])
    df_input.loc[0] = [pm_duakomalima, pm_sepuluh, sulfur_dioksida, karbon_monoksida, ozon, nitrogen_dioksida]
    #print(df_input)
    result = loaded_model.predict(df_input)
    #print(result)
    # for i in result:
    #     int_result = int(i)
    #     if(int_result==0):
    #         decision='Suspect Blood Donor'
    #     elif (int_result==1):
    #         decision='Hepatitis'
    #     elif (int_result==2):
    #         decision='Fibrosis'
    #     elif (int_result==3):
    #         decision='Cirrhosis'
    #     else:
    #         decision='Not defined'
    #print('Disease is ', decision)
    #return the output and load result.html
    # return render_template('result.html', age=age, sex=sex, alb=alb, alp=alp, alt=alt, ast=ast, bil=bil, 
    #                        che=che, chol=chol, crea=crea, ggt=ggt, prot=prot, status=decision)

if __name__ == "__main__":
    #host= ip address, port = port number
    #app.run(host='127.0.0.1', port='5001')
    app.run()
