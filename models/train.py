# Проверяем, что train.py существует
ls -la train.py

# Если нет, создаем
cat > train.py << 'EOF'
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
        file_path = self.config['data']['raw_file']
        
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Файл не найден: {file_path}")
        
        if file_path.endswith('.csv'):
            self.df = pd.read_csv(file_path)
        elif file_path.endswith('.xlsx'):
            self.df = pd.read_excel(file_path)
        else:
            raise ValueError("Поддерживаются только CSV и Excel")
        
        print(f"✅ Загружено {len(self.df)} записей")
        print(f"📊 Колонки: {self.df.columns.tolist()}")
        return self.df
    
    def prepare_features(self):
        features_config = self.config['data']['features']
        targets = self.config['data']['targets']
        
        self.encoded_features = []
        for col in features_config['categorical']:
            if col in self.df.columns:
                self.encoders[col] = LabelEncoder()
                self.df[f'{col}_encoded'] = self.encoders[col].fit_transform(self.df[col].astype(str))
                self.encoded_features.append(f'{col}_encoded')
        
        for col in features_config['numerical']:
            if col in self.df.columns:
                self.encoded_features.append(col)
        
        # Проверяем наличие признаков
        missing = [f for f in self.encoded_features if f not in self.df.columns]
        if missing:
            print(f"⚠️  Отсутствуют: {missing}")
            self.encoded_features = [f for f in self.encoded_features if f in self.df.columns]
        
        # Проверяем целевые переменные
        targets_exist = [t for t in targets if t in self.df.columns]
        if not targets_exist:
            raise ValueError(f"Целевые переменные не найдены. Доступны: {self.df.columns.tolist()}")
        
        self.X = self.df[self.encoded_features]
        self.y = self.df[targets_exist]
        
        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(self.X)
        
        print(f"✅ Признаки: {self.encoded_features}")
        print(f"✅ Целевые: {targets_exist}")
        print(f"✅ Форма X: {self.X.shape}, y: {self.y.shape}")
        
        return self.X, self.y
    
    def train_models(self):
        model_params = self.config['model']['params']
        
        if len(self.df) < 10:
            print("⚠️  Мало данных. Используем все для обучения")
            for target in self.y.columns:
                print(f"\n🔬 Обучение для {target}")
                model = RandomForestRegressor(**model_params)
                model.fit(self.X_scaled, self.y[target])
                self.models[target] = model
                
                pred = model.predict(self.X_scaled)
                r2 = r2_score(self.y[target], pred)
                rmse = np.sqrt(mean_squared_error(self.y[target], pred))
                self.results[target] = {'train_r2': r2, 'train_rmse': rmse}
                print(f"  Train R²: {r2:.4f}, RMSE: {rmse:.4f}")
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                self.X_scaled, self.y,
                test_size=self.config['training']['test_size'],
                random_state=self.config['training']['random_state']
            )
            
            for target in self.y.columns:
                print(f"\n🔬 Обучение для {target}")
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
                    'test_rmse': test_rmse
                }
                
                print(f"  Train R²: {train_r2:.4f}, RMSE: {train_rmse:.4f}")
                print(f"  Test  R²: {test_r2:.4f}, RMSE: {test_rmse:.4f}")
    
    def save_models(self):
        models_dir = self.config['output']['models_dir']
        
        for name, model in self.models.items():
            joblib.dump(model, f"{models_dir}{name}_{self.timestamp}.pkl")
        
        joblib.dump(self.encoders, f"{models_dir}encoders_{self.timestamp}.pkl")
        joblib.dump(self.scaler, f"{models_dir}scaler_{self.timestamp}.pkl")
        
        info = {
            'timestamp': self.timestamp,
            'features': self.encoded_features,
            'targets': self.y.columns.tolist(),
            'n_samples': len(self.df),
            'results': self.results
        }
        
        with open(f"{models_dir}info_{self.timestamp}.json", 'w') as f:
            json.dump(info, f, indent=2)
        
        print(f"\n✅ Модели сохранены в {models_dir}")
        print(f"   Префикс: {self.timestamp}")
    
    def run(self):
        print("=" * 60)
        print("ОБУЧЕНИЕ МОДЕЛИ")
        print("=" * 60)
        
        self.load_data()
        self.prepare_features()
        self.train_models()
        self.save_models()
        
        print("\n✅ Обучение завершено!")

if __name__ == "__main__":
    trainer = RubberModelTrainer()
    trainer.run()
EOF