from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    message = data.get('message', '').strip()

    if not message:
        return jsonify({'error': 'No message provided'}), 400

    vec = vectorizer.transform([message])
    prediction = model.predict(vec)[0]
    proba = model.predict_proba(vec)[0]

    label = 'SPAM' if prediction == 1 else 'HAM'
    confidence = round(float(np.max(proba)) * 100, 2)

    return jsonify({
        'label': label,
        'confidence': confidence,
        'is_spam': bool(prediction == 1)
    })

if __name__ == '__main__':
    app.run(debug=True)
