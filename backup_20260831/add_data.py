import pandas as pd
import argparse
from utils.data_loader import DataLoader
import yaml

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, required=True,
                       help='Файл с новыми данными (CSV или Excel)')
    parser.add_argument('--config', type=str, default='config.yaml')
    args = parser.parse_args()
    
    # Загружаем конфиг
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Создаем загрузчик
    loader = DataLoader(args.config)
    
    # Загружаем существующие данные
    print("📂 Загрузка существующих данных...")
    existing = loader.load_data()
    
    # Загружаем новые данные
    print(f"📂 Загрузка новых данных из {args.file}...")
    if args.file.endswith('.csv'):
        new_data = pd.read_csv(args.file)
    elif args.file.endswith('.xlsx'):
        new_data = pd.read_excel(args.file)
    else:
        raise ValueError("Поддерживаются только CSV и Excel файлы")
    
    # Проверяем, что колонки совпадают
    required_cols = set(existing.columns)
    new_cols = set(new_data.columns)
    
    if not required_cols.issubset(new_cols):
        missing = required_cols - new_cols
        print(f"⚠️  В новых данных отсутствуют колонки: {missing}")
        print("Добавьте эти колонки в файл с данными!")
        return
    
    # Добавляем данные
    updated = loader.add_new_data(new_data)
    
    # Пересохраняем
    loader.preprocess_data()
    
    print(f"\n✅ Данные обновлены! Всего записей: {len(updated)}")
    
    # Показываем новые данные
    print("\n📋 Новые данные:")
    print(updated.tail(5))

if __name__ == "__main__":
    main()