import pandas as pd
import joblib
import os
from flask import Flask, redirect, url_for, request, render_template
from sklearn.preprocessing import LabelEncoder, StandardScaler

app = Flask(__name__)

# Load index.html/ first page. receive input variable from user
@app.route("/ispa/")
def index():
    return render_template('index.html')

# Load result.html. the result of prediction is presented here.
@app.route('/ispa/result/', methods=["POST"])
def prediction_result():
    # Receiving parameters sent by client
    pm_dualima = request.form.get('inputpm25')
    pm_sepuluh = request.form.get('inputpm10')
    sulfur_dioksida = request.form.get('inputso2')
    karbon_monoksida = request.form.get('inputco')
    ozon = request.form.get('inputo3')
    nitrogen_dioksida = request.form.get('inputno2')

    # Load the trained model
    current_directory = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_directory, 'modelRN.pkl')
    scaler_path = os.path.join(current_directory, 'scaler.pkl')
    # encoder_path = os.path.join(current_directory, 'encode.pkl')

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    # Create new DataFrame
    df_input = pd.DataFrame(columns=['pm_sepuluh', 'pm_dualima', 'sulfur_dioksida', 'karbon_monoksida', 'ozon', 'nitrogen_dioksida'])
    df_input.loc[0] = [pm_sepuluh, pm_dualima, sulfur_dioksida, karbon_monoksida, ozon, nitrogen_dioksida]
    
    # Encode
   
    df_input = df_input.apply(pd.to_numeric, errors='coerce')

     # Normalize input data using StandardScaler
    input_sc = scaler.transform(df_input)

    # Predict the result
    result = model.predict(input_sc)

    # Map the prediction to the decision
    for i in result:
        int_result = int(i)
        if int_result == 0:
            decision = 'Baik'
        elif int_result == 1:
            decision = 'Sedang'
        elif int_result == 2:
            decision = 'Tidak Sehat'
        elif int_result == 3:
            decision = 'Sangat Tidak Sehat'
        else:
            decision = 'Tidak Terdefinisi'

    print('Disease is', decision)

    # Return the output and load result.html
    return render_template('result.html', pm_sepuluh=pm_sepuluh, pm_dualima=pm_dualima, sulfur_dioksida=sulfur_dioksida,
                           karbon_monoksida=karbon_monoksida, ozon=ozon, nitrogen_dioksida=nitrogen_dioksida, status=decision)

if __name__ == "__main__":
    app.run()
