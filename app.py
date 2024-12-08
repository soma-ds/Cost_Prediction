from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the saved model and scaler
model = pickle.load(open("model.pkl", 'rb'))
scaler = pickle.load(open("scaler.pkl", 'rb'))

@app.route('/')
def home():
    return render_template('index.html')  # Save the HTML above as 'templates/index.html'

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        form_data = request.form
        features = np.array([[
            float(form_data['unit_sales']),
            float(form_data['avg_cars_at_home']),
            float(form_data['coffee_bar']),
            float(form_data['video_store']),
            float(form_data['salad_bar']),
            float(form_data['prepared_food']),
            float(form_data['florist']),
            float(form_data['have_car']),
            float(form_data['store_sqft'])
        ]])

        # Scale the features
        features_scaled = scaler.transform(features)

        # Predict using the model
        prediction = model.predict(features_scaled)

        return render_template("index.html",prediction_text="Predicted Cost is $ {:.2f}".format(prediction[0]))
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
