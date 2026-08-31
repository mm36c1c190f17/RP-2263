import pandas as pd
import numpy as np
from pathlib import Path
import yaml

class DataLoader:
    def __init__(self, config_path='config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.raw_data = None
        self.processed_data = None
        
    def load_data(self, file_path=None):
        """Загрузка данных из CSV или Excel"""
        if file_path is None:
            file_path = self.config['data']['raw_file']
        
        if file_path.endswith('.csv'):
            self.raw_data = pd.read_csv(file_path)
        elif file_path.endswith('.xlsx'):
            self.raw_data = pd.read_excel(file_path)
        else:
            raise ValueError("Поддерживаются только CSV и Excel файлы")
        
        print(f"✅ Загружено {len(self.raw_data)} записей")
        print(f"📊 Колонки: {self.raw_data.columns.tolist()}")
        print(f"📋 Данные:\n{self.raw_data.head()}")
        
        return self.raw_data
    
    def add_new_data(self, new_data):
        """Добавление новых данных к существующим"""
        if self.raw_data is None:
            self.raw_data = new_data
        else:
            self.raw_data = pd.concat([self.raw_data, new_data], ignore_index=True)
        
        print(f"✅ Добавлено {len(new_data)} записей")
        print(f"📊 Всего записей: {len(self.raw_data)}")
        
        return self.raw_data
    
    def add_data_from_file(self, file_path):
        """Добавление данных из файла"""
        if file_path.endswith('.csv'):
            new_data = pd.read_csv(file_path)
        elif file_path.endswith('.xlsx'):
            new_data = pd.read_excel(file_path)
        else:
            raise ValueError("Поддерживаются только CSV и Excel файлы")
        
        return self.add_new_data(new_data)
    
    def preprocess_data(self):
        """Предобработка данных"""
        if self.raw_data is None:
            raise ValueError("Сначала загрузите данные!")
        
        df = self.raw_data.copy()
        
        # Проверка на пропуски
        if df.isnull().sum().sum() > 0:
            print("⚠️  Обнаружены пропуски в данных:")
            print(df.isnull().sum())
            df = df.dropna()
            print(f"✅ Удалены пропуски. Осталось {len(df)} записей")
        
        # Преобразование типов
        for col in df.columns:
            if df[col].dtype == 'object':
                # Для категориальных признаков
                if col in self.config['data']['features']['categorical']:
                    df[col] = df[col].astype('category')
        
        self.processed_data = df
        self.save_processed_data()
        
        return df
    
    def save_processed_data(self):
        """Сохранение обработанных данных"""
        if self.processed_data is not None:
            output_path = self.config['data']['processed_file']
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            self.processed_data.to_csv(output_path, index=False)
            print(f"💾 Обработанные данные сохранены в {output_path}")

# Если запускаем напрямую
if __name__ == "__main__":
    loader = DataLoader()
    data = loader.load_data()
    processed = loader.preprocess_data()