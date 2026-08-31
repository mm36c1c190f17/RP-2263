from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
from pathlib import Path

app = Flask(__name__)

def load_models():
    """Загрузка моделей"""
    models_dir = Path("models/")
    encoders_files = list(models_dir.glob("encoders_*.pkl"))
    if not encoders_files:
        return None, None, None
    
    latest = max(encoders_files, key=lambda x: x.stat().st_mtime)
    timestamp = latest.stem.replace("encoders_", "")
    
    encoders = joblib.load(f"models/encoders_{timestamp}.pkl")
    scaler = joblib.load(f"models/scaler_{timestamp}.pkl")
    
    models = {}
    for f in models_dir.glob(f"*_{timestamp}.pkl"):
        name = f.stem.replace(f"_{timestamp}", "")
        if name not in ['encoders', 'scaler', 'feature_info']:
            models[name] = joblib.load(f)
    
    return models, encoders, scaler

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        filler_type = data.get('filler_type')
        silane = float(data.get('silane', 0))
        amount = float(data.get('amount', 40))
        
        models, encoders, scaler = load_models()
        if not models:
            return jsonify({'error': 'Модели не загружены'}), 500
        
        # Подготовка данных
        input_data = {
            'filler_type': filler_type,
            'manufacturer': 'Дубна',
            'rubber_type': 'СКТВ-1',
            'silane_content': silane,
            'filler_amount': amount
        }
        
        X_dict = {}
        for col, encoder in encoders.items():
            if col in input_data:
                try:
                    X_dict[f'{col}_encoded'] = encoder.transform([input_data[col]])[0]
                except:
                    X_dict[f'{col}_encoded'] = encoder.transform([encoder.classes_[0]])[0]
        
        X_dict['silane_content'] = input_data['silane_content']
        X_dict['filler_amount'] = input_data['filler_amount']
        
        X_df = pd.DataFrame([X_dict])
        X_scaled = scaler.transform(X_df)
        
        predictions = {}
        for name, model in models.items():
            predictions[name] = float(model.predict(X_scaled)[0])
        
        return jsonify(predictions)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
