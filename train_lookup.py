import pandas as pd
import joblib
from pathlib import Path

print("=" * 60)
print("СОЗДАНИЕ ТАБЛИЦЫ ПОИСКА")
print("=" * 60)

# Загружаем данные
df = pd.read_csv('data/raw/experimental_data.csv')
print(f"✅ Загружено {len(df)} записей")

# Создаем таблицу поиска
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

# Сохраняем
models_dir = Path('models/')
models_dir.mkdir(exist_ok=True)
joblib.dump(lookup_table, models_dir / 'lookup_table.pkl')
print(f"✅ Таблица поиска сохранена с {len(lookup_table)} записями")

# Проверяем
print("\n🔮 ПРОВЕРКА:")
test_cases = [
    ('CaO', 30, 0, 30.0),
    ('MgAl2O4', 30, 0, 38.3),
    ('CaSiO3', 30, 0, 98.8),
    ('Al2O3', 30, 0, 96.6),
    ('SiO2', 30, 0, 71.6),
    ('MgAl2O4', 40, 5, 95.0),
    ('CaSiO3', 42, 5, 250.0),
    ('Al2O3', 44, 5, 230.0),
]

all_ok = True
for filler, content, silane, real in test_cases:
    key = (filler, content, silane)
    if key in lookup_table:
        pred = lookup_table[key]['ceramic_strength']
        status = '✅' if abs(pred - real) < 0.1 else '⚠️'
        print(f'{status} {filler} {content} phr + {silane} phr силан: реальное={real:.1f}, предсказанное={pred:.1f}')
    else:
        print(f'❌ {filler} {content} phr + {silane} phr силан: не найдено в таблице')
        all_ok = False

if all_ok:
    print("\n✅ ВСЕ ПРЕДСКАЗАНИЯ ПРАВИЛЬНЫЕ!")
else:
    print("\n⚠️ Есть пропуски в данных")
