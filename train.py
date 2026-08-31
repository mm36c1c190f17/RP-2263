import pandas as pd
import numpy as np
import yaml
import joblib
import json
from datetime import datetime
from pathlib import Path
import argparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class RubberModelTrainer:
    def __init__(self, config_path='config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.models = {}
        self.encoders = {}
        self.scaler = None
        self.results = {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for dir_path in [self.config['output']['models_dir'], 
                        self.config['output']['plots_dir']]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def load_data(self):
        """Загрузка данных из CSV файла"""
        file_path = 'data/raw/experimental_data.csv'
        
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Файл не найден: {file_path}")
        
        self.df = pd.read_csv(file_path)
        print(f"✅ Загружено {len(self.df)} записей для обучения")
        print(f"📊 Колонки: {self.df.columns.tolist()}")
        return self.df
    
    def prepare_features(self):
        """Подготовка признаков"""
        features_config = self.config['data']['features']
        targets = self.config['data']['targets']
        
        # Кодирование категориальных признаков
        self.encoded_features = []
        for col in features_config['categorical']:
            if col in self.df.columns:
                self.encoders[col] = LabelEncoder()
                self.df[f'{col}_encoded'] = self.encoders[col].fit_transform(self.df[col].astype(str))
                self.encoded_features.append(f'{col}_encoded')
        
        # Добавляем числовые признаки
        for col in features_config['numerical']:
            if col in self.df.columns:
                self.encoded_features.append(col)
        
        # Проверяем наличие признаков
        missing = [f for f in self.encoded_features if f not in self.df.columns]
        if missing:
            print(f"⚠️  Отсутствуют признаки: {missing}")
            self.encoded_features = [f for f in self.encoded_features if f in self.df.columns]
        
        # Проверяем целевые переменные
        targets_exist = [t for t in targets if t in self.df.columns]
        if not targets_exist:
            raise ValueError(f"Целевые переменные не найдены. Доступны: {self.df.columns.tolist()}")
        
        self.X = self.df[self.encoded_features]
        self.y = self.df[targets_exist]
        
        # Масштабирование
        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(self.X)
        
        print(f"✅ Признаки: {self.encoded_features}")
        print(f"✅ Целевые переменные: {targets_exist}")
        print(f"✅ Форма X: {self.X.shape}, Форма y: {self.y.shape}")
        
        return self.X, self.y
    
    def train_models(self):
        """Обучение моделей"""
        model_params = self.config['model']['params']
        
        # Если данных мало, используем все для обучения
        if len(self.df) < 10:
            print("⚠️  Мало данных. Используем все для обучения (без теста)")
            for target in self.y.columns:
                print(f"\n🔬 Обучение модели для {target}")
                print("-" * 50)
                
                model = RandomForestRegressor(**model_params)
                model.fit(self.X_scaled, self.y[target])
                self.models[target] = model
                
                pred = model.predict(self.X_scaled)
                r2 = r2_score(self.y[target], pred)
                rmse = np.sqrt(mean_squared_error(self.y[target], pred))
                
                self.results[target] = {
                    'train_r2': r2,
                    'train_rmse': rmse,
                    'n_samples': len(self.df)
                }
                
                print(f"  Train R²: {r2:.4f}, RMSE: {rmse:.4f}")
        else:
            # Обычное обучение с разделением
            test_size = min(0.2, 5/len(self.df))
            X_train, X_test, y_train, y_test = train_test_split(
                self.X_scaled, self.y,
                test_size=test_size,
                random_state=self.config['training']['random_state']
            )
            
            for target in self.y.columns:
                print(f"\n🔬 Обучение модели для {target}")
                print("-" * 50)
                
                model = RandomForestRegressor(**model_params)
                model.fit(X_train, y_train[target])
                self.models[target] = model
                
                train_pred = model.predict(X_train)
                test_pred = model.predict(X_test)
                
                train_r2 = r2_score(y_train[target], train_pred)
                test_r2 = r2_score(y_test[target], test_pred)
                train_rmse = np.sqrt(mean_squared_error(y_train[target], train_pred))
                test_rmse = np.sqrt(mean_squared_error(y_test[target], test_pred))
                
                self.results[target] = {
                    'train_r2': train_r2,
                    'train_rmse': train_rmse,
                    'test_r2': test_r2,
                    'test_rmse': test_rmse,
                    'n_samples': len(self.df)
                }
                
                print(f"  Train R²: {train_r2:.4f}, RMSE: {train_rmse:.4f}")
                print(f"  Test  R²: {test_r2:.4f}, RMSE: {test_rmse:.4f}")
        
        return self.models, self.results
    
    def save_models(self):
        """Сохранение моделей"""
        models_dir = self.config['output']['models_dir']
        
        # Сохраняем модели
        for name, model in self.models.items():
            joblib.dump(model, f"{models_dir}{name}_{self.timestamp}.pkl")
        
        # Сохраняем энкодеры и скейлер
        joblib.dump(self.encoders, f"{models_dir}encoders_{self.timestamp}.pkl")
        joblib.dump(self.scaler, f"{models_dir}scaler_{self.timestamp}.pkl")
        
        # Сохраняем информацию о признаках
        feature_info = {
            'features': self.encoded_features,
            'feature_names': self.X.columns.tolist(),
            'timestamp': self.timestamp
        }
        joblib.dump(feature_info, f"{models_dir}feature_info_{self.timestamp}.pkl")
        
        # Сохраняем результаты
        results = {
            'timestamp': self.timestamp,
            'config': self.config,
            'results': self.results,
            'n_samples': len(self.df),
            'features': self.encoded_features,
            'targets': self.y.columns.tolist()
        }
        
        with open(f"{models_dir}training_results_{self.timestamp}.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✅ Модели сохранены в {models_dir}")
        print(f"   Префикс: {self.timestamp}")
        print(f"   Обучено на {len(self.df)} записях")
    
    def run(self):
        """Запуск полного процесса"""
        print("=" * 60)
        print("ОБУЧЕНИЕ МОДЕЛИ ДЛЯ ПРОГНОЗИРОВАНИЯ СВОЙСТВ РЕЗИН")
        print("=" * 60)
        
        self.load_data()
        self.prepare_features()
        self.train_models()
        self.save_models()
        
        print("\n✅ Обучение завершено!")
        return self.models, self.encoders, self.results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml', 
                       help='Путь к конфигурационному файлу')
    parser.add_argument('--data', type=str, 
                       help='Путь к данным (переопределяет config)')
    parser.add_argument('--model', type=str, 
                       help='Тип модели (переопределяет config)')
    args = parser.parse_args()
    
    trainer = RubberModelTrainer(args.config)
    
    if args.data:
        # Сначала обрабатываем новые данные
        from utils.data_loader import DataLoader
        loader = DataLoader(args.config)
        raw = loader.load_data(args.data)
        loader.preprocess_data()
    
    if args.model:
        trainer.config['model']['type'] = args.model
    
    trainer.run()