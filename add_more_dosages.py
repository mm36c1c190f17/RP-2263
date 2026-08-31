import pandas as pd

# Добавляем больше данных с разными дозировками
# (это пример - вы должны добавить свои реальные данные)
new_data = [
    # Шпинель - разные дозировки
    {'filler_type': 'шпинель', 'filler_amount': 20, 'silane_content': 0,
     'fp_before': 8.7, 'ep_before': 380, 'fp_after': 4.8, 'ep_after': 120,
     'rho_before': 4.3e13, 'rho_after': 3.8e13, 'P_strength': 25.0,
     'manufacturer': 'Дубна', 'rubber_type': 'СКТВ-1'},
     
    {'filler_type': 'шпинель', 'filler_amount': 50, 'silane_content': 0,
     'fp_before': 7.8, 'ep_before': 320, 'fp_after': 3.9, 'ep_after': 85,
     'rho_before': 4.1e13, 'rho_after': 3.0e13, 'P_strength': 55.0,
     'manufacturer': 'Дубна', 'rubber_type': 'СКТВ-1'},
     
    # Волластонит - разные дозировки
    {'filler_type': 'волластонит', 'filler_amount': 20, 'silane_content': 0,
     'fp_before': 9.5, 'ep_before': 270, 'fp_after': 7.5, 'ep_after': 180,
     'rho_before': 4.7e13, 'rho_after': 4.0e13, 'P_strength': 65.0,
     'manufacturer': 'Дубна', 'rubber_type': 'СКТВ-1'},
     
    {'filler_type': 'волластонит', 'filler_amount': 50, 'silane_content': 0,
     'fp_before': 8.5, 'ep_before': 200, 'fp_after': 6.5, 'ep_after': 130,
     'rho_before': 4.5e13, 'rho_after': 3.4e13, 'P_strength': 150.0,
     'manufacturer': 'Дубна', 'rubber_type': 'СКТВ-1'},
]

# Загружаем существующие данные
df = pd.read_csv('data/raw/experimental_data.csv')

# Добавляем новые
for record in new_data:
    exists = df[
        (df['filler_type'] == record['filler_type']) & 
        (df['filler_amount'] == record['filler_amount']) & 
        (df['silane_content'] == record['silane_content'])
    ]
    if len(exists) == 0:
        df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)

# Сохраняем
df.to_csv('data/raw/experimental_data.csv', index=False)

print(f'✅ Добавлены данные с разными дозировками')
print(f'📊 Всего записей: {len(df)}')
print('\n📊 Данные по шпинели:')
print(df[df['filler_type'] == 'шпинель'][['filler_amount', 'P_strength']])
