from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np

my_app = Flask(__name__,template_folder='templates')

with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

def preprocess_data(input_data):
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    scaler = StandardScaler()
    label_encoder = LabelEncoder()
    object_columns = input_data.select_dtypes(include=['object']).columns

    for column in object_columns:
        input_data[column] = label_encoder.fit_transform(input_data[column])

    scaled_features = scaler.fit_transform(input_data)

    return scaled_features

@my_app.route('/')
def index():
    return render_template('index.html', predictions=None)

@my_app.route('/predict', methods=['POST'])
def predict():
    
    if request.method == 'POST':
        total_f = float(request.form.get('total_f', 0.0))
        funding_rounds = float(request.form.get('funding_rounds', 0.0))
        seed = float(request.form.get('seed', 0.0))
        venture = float(request.form.get('venture', 0.0))
        market_encoded = request.form.get('market_encoded', 0.0)
        debt_financing = float(request.form.get('debt_financing', 0.0))
        country_code_encoded = request.form.get('country_code_encoded', 0.0)
        state_code_encoded = request.form.get('state_code_encoded', 0.0)

        input_data = pd.DataFrame({
            'total_f': [total_f],
            'funding_rounds': [funding_rounds],
            'seed': [seed],
            'venture': [venture],
            'market_encoded': [market_encoded],
            'debt_financing': [debt_financing],
            'country_code_encoded': [country_code_encoded],
            'state_code_encoded': [state_code_encoded]
        })

        
        preprocessed_df = preprocess_data(input_data)
        predictions = model.predict(preprocessed_df)
        status_messages = []


        for prediction in predictions:
            if prediction == 1:
                status_messages.append("Operating Status")
            elif prediction == 2:
                status_messages.append("Acquired Status")
            else:
                status_messages.append("Closed Status")

        # Combine the status messages into a single string
        status_message = ", ".join(status_messages)

        # Return the rendered template
        return render_template('index.html', pred='Prediction Result is [{}] ({})'.format(predictions[0], status_message))


if __name__ == '__main__':
    my_app.run()
