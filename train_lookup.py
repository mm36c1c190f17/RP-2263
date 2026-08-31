import pandas as pd
import joblib
from pathlib import Path

print("=" * 60)
print("СОЗДАНИЕ ТАБЛИЦЫ ПОИСКА")
print("=" * 60)

df = pd.read_csv('data/raw/experimental_data.csv')
print(f"✅ Загружено {len(df)} записей")

lookup_table = {}
for _, row in df.iterrows():
    key = (row['filler_type'], row['filler_content'], row['silane_content'])
    lookup_table[key] = {
        'ceramic_strength': float(row['ceramic_strength']) if pd.notna(row['ceramic_strength']) else None,
        'strength_initial': float(row['strength_initial']) if pd.notna(row['strength_initial']) else None,
        'elongation_initial': float(row['elongation_initial']) if pd.notna(row['elongation_initial']) else None,
        'strength_aged_240h_250C': float(row['strength_aged_240h_250C']) if pd.notna(row['strength_aged_240h_250C']) else None,
        'elongation_aged_240h_250C': float(row['elongation_aged_240h_250C']) if pd.notna(row['elongation_aged_240h_250C']) else None,
        'resistivity': float(row['resistivity']) if pd.notna(row['resistivity']) else None,
    }

models_dir = Path('models/')
models_dir.mkdir(exist_ok=True)
joblib.dump(lookup_table, models_dir / 'lookup_table.pkl')
print(f"✅ Таблица поиска сохранена с {len(lookup_table)} записями")

print("\n📊 Содержимое таблицы:")
for key, values in lookup_table.items():
    filler, content, silane = key
    ceramic = values.get('ceramic_strength')
    if ceramic is not None:
        print(f"  {filler} {content} phr + {silane} phr силан: {ceramic:.1f}")
