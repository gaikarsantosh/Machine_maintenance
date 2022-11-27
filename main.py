from flask import Flask, render_template, request
import pickle
import numpy as np


app=Flask(__name__)
model= pickle.load(open('RF_model.pkl','rb'))


# @app.route('/')
# def Home():
#     return 'Welcome to home page'

@app.route('/')
def man():
    return render_template('home.html')

@app.route('/Prediction', methods=['GET','POST'])
def prediction():
    data = request.form
    if request.method == 'POST':
        Type = int(data['Type'])
        Air_temp = float(data['Air_temp'])
        Process_temp = float(data['Process_temp'])
        Speed_RPM = int(data['Speed_RPM'])
        Torque_Nm = float(data['Torque_Nm'])
        Tool_wear = int(data['Tool_wear'])

        inputs= [Type,Air_temp,Process_temp,Speed_RPM,Torque_Nm,Tool_wear]
        features=[np.array(inputs)]

        Mc_maint= model.predict(features)

        a=list(Mc_maint.flatten())
        
        return render_template('prediction.html', data=a)

    #     op=''
    #     if a[0]==1:
    #         op+= ('Machine will have failure and potential failure modes will be_\n')
    #     else:
    #         op+= ('Machine will not have failure')
    #     if a[1]==1:
    #         op+=('\nTWF- Tool Wear Failure')
    #     if a[2]==1:
    #         op+=('\nHDF- Heat Desipation Failure')
    #     if a[3]==1:
    #         op+=('\nPWF- Power Failure')
    #     if a[4]==1:
    #         op+=('\nOSF- Overstrain Failure')
    #     if a[5]==1:
    #         op+=('\nRNF- Random Failure')
    # return op
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)