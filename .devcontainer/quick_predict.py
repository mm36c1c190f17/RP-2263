cat > quick_predict.py << 'EOF'
import pandas as pd
import joblib
from pathlib import Path
import sys
import os

def quick_predict(filler_type, amount=30, silane=0):
    """Быстрое предсказание для любого состава"""
    # Загружаем модели
    models_dir = Path('models/')
    encoders_files = list(models_dir.glob('encoders_*.pkl'))
    if not encoders_files:
        return "❌ Модели не найдены! Запустите python train_knn.py"
    
    # Берем САМУЮ СВЕЖУЮ модель (по времени создания)
    latest = max(encoders_files, key=lambda x: x.stat().st_mtime)
    timestamp = latest.stem.replace('encoders_', '')
    
    print(f"📂 Загружаем модель {timestamp}")
    
    encoders = joblib.load(f'models/encoders_{timestamp}.pkl')
    scaler = joblib.load(f'models/scaler_{timestamp}.pkl')
    
    models = {}
    for f in models_dir.glob(f'*_{timestamp}.pkl'):
        name = f.stem.replace(f'_{timestamp}', '')
        if name not in ['encoders', 'scaler', 'feature_info']:
            models[name] = joblib.load(f)
    
    # Подготовка данных
    data = {
        'filler_type': filler_type,
        'manufacturer': 'Дубна',
        'rubber_type': 'СКТВ-1',
        'silane_content': silane,
        'filler_amount': amount
    }
    
    X_dict = {}
    for col, encoder in encoders.items():
        if col in data:
            try:
                X_dict[f'{col}_encoded'] = encoder.transform([data[col]])[0]
            except:
                X_dict[f'{col}_encoded'] = encoder.transform([encoder.classes_[0]])[0]
    
    X_dict['silane_content'] = data['silane_content']
    X_dict['filler_amount'] = data['filler_amount']
    
    X_df = pd.DataFrame([X_dict])
    X_scaled = scaler.transform(X_df)
    
    # Предсказания
    results = {}
    for name, model in models.items():
        results[name] = model.predict(X_scaled)[0]
    
    return results

if __name__ == '__main__':
    if len(sys.argv) > 1:
        filler = sys.argv[1]
        amount = float(sys.argv[2]) if len(sys.argv) > 2 else 30
        silane = float(sys.argv[3]) if len(sys.argv) > 3 else 0
        
        result = quick_predict(filler, amount, silane)
        
        if isinstance(result, str):
            print(result)
        else:
            print(f'\n🔮 {filler} ({amount} phr, силан: {silane} phr)')
            print('=' * 50)
            for key, value in result.items():
                if abs(value) > 1e12:
                    print(f'  {key}: {value:.3e}')
                elif abs(value) < 0.1:
                    print(f'  {key}: {value:.4f}')
                else:
                    print(f'  {key}: {value:.2f}')
    else:
        print('Использование: python quick_predict.py <наполнитель> [дозировка] [силан]')
        print('\nПримеры:')
        print('  python quick_predict.py шпинель 30 0')
        print('  python quick_predict.py волластонит 40 5')
        print('  python quick_predict.py Al2O3 50 0')
EOF