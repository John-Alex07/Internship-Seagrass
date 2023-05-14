from flask import Flask, request, render_template
import joblib

app = Flask(__name__)

# Load the pre-trained model and count vectorizer
model = joblib.load("LR_Best_Model.sav")
vectorizer = joblib.load("TFIDF_TRANSFORM.sav")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the wine review input from the POST request
    review = request.form['review']
    
    # Vectorize the input using the pre-trained count vectorizer
    review_vector = vectorizer.transform([review])
    
    # Make a prediction using the pre-trained model
    variety = model.predict(review_vector)[0]
    
    # Return the predicted wine variety to the user
    return render_template('result.html', variety=variety)

if __name__ == '__main__':
    app.run(debug=True)
