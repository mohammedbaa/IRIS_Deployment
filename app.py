from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load the pre-trained SVM model
with open('svm_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input values from the form
    sepal_length = float(request.form['sepal_length'])
    sepal_width = float(request.form['sepal_width'])
    petal_length = float(request.form['petal_length'])
    petal_width = float(request.form['petal_width'])

    # Make prediction using the pre-trained model
    prediction = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])

    # Mapping the predicted class index to class labels
    species = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}

    # Get the predicted species label
    predicted_species = species[prediction[0]]

    return render_template('index.html', prediction_text=f'Predicted species: {predicted_species}')

if __name__ == '__main__':
    app.run(debug=True)
