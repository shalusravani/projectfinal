from flask import Flask, request, render_template
from flask_cors import cross_origin
import sklearn
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)
model = pickle.load(open("customer_churn_prediction_abc_gscv.pkl", "rb"))



@app.route("/")
@cross_origin()
def home():
    return render_template("home.html")




@app.route("/predict", methods = ["GET", "POST"])
@cross_origin()
def predict():
    if request.method == "POST":

        # Gender
        gender = request.form["Gender"]
        if(gender=='Yes'):
            gender = 1
        else:
            gender = 0
          
        #Senior Citizen
        senior_citizen = request.form["Senior_Citizen"]
        if(senior_citizen=='Yes'):
            SeniorCitizen = 1
        else:
            SeniorCitizen = 0

        #Partner
        partner = request.form["Partner"]
        if(partner=='Yes'):
            Partner = 1
        else:
            Partner = 0

        #Dependents
        dependents = request.form["Dependents"]
        if(dependents=='Yes'):
            Dependents = 1
        else:
            Dependents = 0


        #Tenure
        tenure = request.form["Tenure"]
        tenure = int(tenure)

        #PhoneService
        phoneservice = request.form["PhoneService"]
        if(phoneservice=='Yes'):
            PhoneService = 1
        else:
            PhoneService = 0

        #Multiple Lines
        multiple_lines = request.form["Multiple_Lines"]
        if(multiple_lines=='Yes'):
            MultipleLines = 1
        else:
            MultipleLines = 0

         #Internet Service
        internet_service = request.form["Internet_Service"]
        if(internet_service=='DSL'):
            InternetService = 1
        elif(internet_service=='Fiber optic'):
            InternetService = 2
        else:
            InternetService = 3
            
        #Online Security
        online_security = request.form["Online_Security"]
        if(online_security=='Yes'):
            OnlineSecurity = 1
        else:
            OnlineSecurity = 0
            

        #Online Backup
        online_backup = request.form["Online_Backup"]
        if(online_backup=='Yes'):
            OnlineBackup = 1
        else:
            OnlineBackup = 0

        #Device Protection
        device_protection = request.form["Device_Protection"]
        if(device_protection=='Yes'):
            DeviceProtection = 1
        else:
            DeviceProtection = 0

        #Tech Support
        tech_support = request.form["Tech_Support"]
        if(tech_support=='Yes'):
            TechSupport = 1
        else:
            TechSupport = 0

        #Streaming TV
        streaming_tV = request.form["Streaming_TV"]
        if(streaming_tV=='Yes'):
            StreamingTV = 1
        else:
            StreamingTV = 0

        #Streaming Movies
        streaming_movies = request.form["Streaming_Movies"]
        if(streaming_movies=='Yes'):
            StreamingMovies = 1
        else:
            StreamingMovies = 0

        #Contract
        contract = request.form["Contract"]
        if(contract=='Month-to-month'):
            Contract = 0
        elif(contract=='One year'):
            Contract = 1
        else:
            Contract = 2

        #Paperless Billing
        paperless_billing = request.form["Paperless_Billing"]
        if(paperless_billing=='Yes'):
            PaperlessBilling = 1
        else:
            PaperlessBilling = 0

        #Payment Method
        payment_method = request.form["Payment_Method"]
        if(payment_method=='Bank Transfer Automatic'):
            Bank_transfer_automatic = 1
            Credit_card_automatic = 0
            Electronic_check = 0
            Mailed_check = 0
        elif(payment_method=='Credit Card Automatic'):
            Bank_transfer_automatic = 0
            Credit_card_automatic = 1
            Electronic_check = 0
            Mailed_check = 0
        elif(payment_method=='Electronic Check'):
            Bank_transfer_automatic = 0
            Credit_card_automatic = 0
            Electronic_check = 1
            Mailed_check = 0
        elif(payment_method=='Mailed Check'):
            Bank_transfer_automatic = 0
            Credit_card_automatic = 0
            Electronic_check = 0
            Mailed_check = 1
        else:
            Bank_transfer_automatic = 0
            Credit_card_automatic = 0
            Electronic_check = 0
            Mailed_check = 0

        #Monthly Charges
        monthly_charges = request.form["Monthly_Charges"]
        MonthlyCharges = float(monthly_charges)

        #Total Charges
        total_charges = request.form["Total_Charges"]
        TotalCharges = float(total_charges)

        
        prediction=model.predict([[
            gender,
            SeniorCitizen,
            Partner,
            Dependents,
            tenure,
            PhoneService,
            MultipleLines,
            InternetService,
            OnlineSecurity,
            OnlineBackup,
            DeviceProtection,
            TechSupport,
            StreamingTV,
            StreamingMovies,
            Contract,
            PaperlessBilling,
            Bank_transfer_automatic,
            Credit_card_automatic,
            Electronic_check,
            Mailed_check,
            MonthlyCharges,
            TotalCharges
        ]])

    prediction=np.argmax(prediction, axis=0)
    if prediction==0:
        prediction="Churned"
    else:
        preds="Not Churned"
        
    return render_template('home.html',prediction_text="Customer Churn Prediction is. {}".format(prediction))


    return render_template("home.html")




if __name__ == "__main__":
    app.run(debug=True)
