import pandas as pd
import numpy as np
import joblib
import yaml
import argparse
from pathlib import Path
import json
import glob

class RubberPredictor:
    def __init__(self, config_path='config.yaml'):
        self.config = yaml.safe_load(open(config_path, 'r'))
        self.models_dir = Path(self.config['output']['models_dir'])
        self.load_latest_models()
        
    def load_latest_models(self):
        """Загрузка последних обученных моделей"""
        # Находим последнюю версию
        model_files = list(self.models_dir.glob('*.pkl'))
        
        if not model_files:
            raise FileNotFoundError("Нет сохраненных моделей!")
        
        # Определяем последний timestamp
        timestamps = set()
        for f in model_files:
            parts = f.stem.split('_')
            if len(parts) >= 2 and parts[-1].isdigit():
                timestamps.add(parts[-1])
        
        if not timestamps:
            raise ValueError("Не удалось определить версию моделей")
        
        latest = max(timestamps)
        print(f"📂 Загрузка моделей версии: {latest}")
        
        # Загружаем энкодеры
        self.encoders = joblib.load(self.models_dir / f"encoders_{latest}.pkl")
        
        # Загружаем скейлер
        self.scaler = joblib.load(self.models_dir / f"scaler_{latest}.pkl")
        
        # Загружаем информацию о признаках
        feature_info = joblib.load(self.models_dir / f"feature_info_{latest}.pkl")
        self.features = feature_info['features']
        
        # Загружаем модели
        self.models = {}
        for target in self.config['data']['targets']:
            model_file = self.models_dir / f"{target}_{latest}.pkl"
            if model_file.exists():
                self.models[target] = joblib.load(model_file)
        
        print(f"✅ Загружено {len(self.models)} моделей")
        print(f"✅ Признаки: {self.features}")
        
    def predict(self, filler_type, silane_content, filler_amount, manufacturer='Дубна', rubber_type='СКТВ-1'):
        """Предсказание для нового состава"""
        # Создаем DataFrame с новыми данными
        new_data = pd.DataFrame({
            'filler_type': [filler_type],
            'silane_content': [silane_content],
            'filler_amount': [filler_amount],
            'manufacturer': [manufacturer],
            'rubber_type': [rubber_type]
        })
        
        # Кодируем категориальные признаки
        X_encoded = {}
        for col, encoder in self.encoders.items():
            if col in new_data.columns:
                # Проверяем, есть ли такое значение в обучении
                try:
                    encoded_val = encoder.transform(new_data[col])[0]
                    X_encoded[f'{col}_encoded'] = encoded_val
                except ValueError:
                    print(f"⚠️  Неизвестное значение {new_data[col][0]} для {col}")
                    print(f"Доступные значения: {encoder.classes_}")
                    # Используем ближайшее известное значение
                    fallback = encoder.classes_[0]
                    encoded_val = encoder.transform([fallback])[0]
                    X_encoded[f'{col}_encoded'] = encoded_val
                    print(f"💡 Используем {fallback} вместо {new_data[col][0]}")
        
        # Добавляем числовые признаки
        X_encoded['silane_content'] = silane_content
        X_encoded['filler_amount'] = filler_amount
        
        # Создаем DataFrame с правильным порядком признаков
        X_new = pd.DataFrame([X_encoded])[self.features]
        
        # Масштабируем
        X_new_scaled = self.scaler.transform(X_new)
        
        # Предсказываем
        predictions = {}
        for target, model in self.models.items():
            value = model.predict(X_new_scaled)[0]
            predictions[target] = value
        
        return predictions
    
    def predict_batch(self, compositions):
        """Пакетное предсказание для нескольких составов"""
        results = []
        for comp in compositions:
            pred = self.predict(**comp)
            results.append({**comp, **pred})
        return pd.DataFrame(results)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filler', type=str, required=True,
                       help='Тип наполнителя (шпинель, волластонит, Al2O3)')
    parser.add_argument('--silane', type=float, required=True,
                       help='Дозировка силана (масс. частей)')
    parser.add_argument('--amount', type=float, required=True,
                       help='Дозировка наполнителя (масс. частей)')
    parser.add_argument('--manufacturer', type=str, default='Дубна',
                       help='Производитель каучука')
    parser.add_argument('--rubber', type=str, default='СКТВ-1',
                       help='Марка каучука')
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--output', type=str, help='Файл для сохранения результатов')
    parser.add_argument('--batch', type=str, help='CSV файл с пакетными предсказаниями')
    
    args = parser.parse_args()
    
    predictor = RubberPredictor(args.config)
    
    if args.batch:
        # Пакетное предсказание
        batch_data = pd.read_csv(args.batch)
        results = []
        for _, row in batch_data.iterrows():
            pred = predictor.predict(
                filler_type=row['filler_type'],
                silane_content=row['silane_content'],
                filler_amount=row['filler_amount'],
                manufacturer=row.get('manufacturer', 'Дубна'),
                rubber_type=row.get('rubber_type', 'СКТВ-1')
            )
            results.append({**row.to_dict(), **pred})
        
        df_results = pd.DataFrame(results)
        output_file = args.output or "batch_predictions.csv"
        df_results.to_csv(output_file, index=False)
        print(f"\n✅ Пакетные предсказания сохранены в {output_file}")
        
    else:
        # Одиночное предсказание
        predictions = predictor.predict(
            args.filler,
            args.silane,
            args.amount,
            args.manufacturer,
            args.rubber
        )
        
        print("\n🔮 ПРЕДСКАЗАНИЯ:")
        print("=" * 50)
        for param, value in predictions.items():
            print(f"{param:20}: {value:.4e}")
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(predictions, f, indent=2)
            print(f"\n✅ Результаты сохранены в {args.output}")

if __name__ == "__main__":
    main()