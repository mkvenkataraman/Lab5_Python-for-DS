from flask import Flask, render_template, request
import pickle
import numpy as np

# Create a Flask application
app = Flask(__name__)

# Load the trained model
model = pickle.load(open('random_forest_model.pkl', 'rb'))

# Define the homepage route
@app.route('/')
def home():
    return render_template('index.html')

# Define the predict route for handling form submissions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get values from form fields and convert them to floats
        present_price = float(request.form['Present_Price']) / 100000  # Convert to lakhs if needed
        kms_driven = float(request.form['Kms_Driven'])
        owner = float(request.form['Owner'])
        age_of_the_car = float(request.form['age_of_the_car'])
        fuel_type_diesel = float(request.form['Fuel_Type_Diesel'])
        seller_type_individual = float(request.form['Seller_Type_Individual'])
        transmission_manual = float(request.form['Transmission_Manual'])

        # Create a feature array for model prediction
        final_features = np.array([present_price, kms_driven, owner, age_of_the_car,
                                   fuel_type_diesel, seller_type_individual, transmission_manual]).reshape(1, -1)

        # Check if all inputs are zero
        if np.sum(final_features) == 0:
            predicted_price = 0.0
        else:
            # Make a prediction using the trained model
            prediction = model.predict(final_features)
            predicted_price = round(prediction[0], 2)

        # Return the result to the frontend
        return render_template(
            'index.html',
            prediction_text=f'Predicted Selling Price of the car is ₹{predicted_price} lakhs'
        )

    except Exception as e:
        print(f"Error occurred: {e}")  # Print the error for debugging
        return render_template('index.html', prediction_text='Error in prediction, please check your inputs.')

# Run the Flask application
if __name__ == "__main__":
    app.run(debug=True)
