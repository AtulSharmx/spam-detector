from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 64 * 1024  # reject oversized request bodies

# Load model and vectorizer
model = joblib.load(os.path.join(BASE_DIR, 'model.pkl'))
vectorizer = joblib.load(os.path.join(BASE_DIR, 'vectorizer.pkl'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(silent=True)
    message = data.get('message') if isinstance(data, dict) else None

    if not isinstance(message, str) or not message.strip():
        return jsonify({'error': 'No message provided'}), 400

    vec = vectorizer.transform([message.strip()])
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
