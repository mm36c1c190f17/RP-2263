import pandas as pd

# Загружаем данные
df = pd.read_csv('data/raw/experimental_data.csv')

# Словарь для приведения названий к единому формату
filler_mapping = {
    'MgAl2O4': 'шпинель',
    'шпинель': 'шпинель',
    'CaSiO3': 'волластонит',
    'волластонит': 'волластонит',
    'Al2O3': 'Al2O3',
    'SiO2': 'SiO2',
    'CaO': 'CaO',
    'Al(OH)3': 'Al(OH)3',
    'CaMg(CO3)2': 'CaMg(CO3)2',
    'Mg3Si4O10(OH)2': 'Mg3Si4O10(OH)2',
    'Al2Si2O5(OH)4': 'Al2Si2O5(OH)4'
}

# Заменяем названия
df['filler_type'] = df['filler_type'].map(filler_mapping).fillna(df['filler_type'])

# Удаляем дубликаты (если есть одинаковые записи)
df = df.drop_duplicates(subset=['filler_type', 'filler_amount', 'silane_content'])

# Сохраняем
df.to_csv('data/raw/experimental_data.csv', index=False)

print('✅ Названия наполнителей унифицированы!')
print('=' * 60)

# Показываем данные по шпинели
print('\n📊 Данные по шпинели:')
spinel = df[df['filler_type'] == 'шпинель']
print(spinel[['filler_type', 'filler_amount', 'silane_content', 'P_strength']])

print(f'\nВсего записей по шпинели: {len(spinel)}')
print('\n📊 Все наполнители в данных:')
print(df['filler_type'].unique())
