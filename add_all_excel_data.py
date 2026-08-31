import pandas as pd
import numpy as np

# Загружаем данные из Excel
excel_data = pd.read_csv('data/raw/rubber_data_extended.csv')

print('📊 Данные из Excel:')
print(excel_data[['filler_type', 'filler_content', 'ceramic_strength']])

# Преобразуем в формат P_strength
new_records = []
for _, row in excel_data.iterrows():
    new_records.append({
        'filler_type': row['filler_type'],
        'silane_content': 0,  # В Excel нет силана
        'filler_amount': row['filler_content'],
        'fp_before': row['strength_initial'],
        'ep_before': row['elongation_initial'],
        'fp_after': row['strength_aged_240h_250C'],
        'ep_after': row['elongation_aged_240h_250C'],
        'rho_before': row['resistivity'],
        'rho_after': row['resistivity'] * 0.8,  # приближение
        'P_strength': row['ceramic_strength'],  # ВАЖНО: используем ceramic_strength
        'manufacturer': 'Дубна',
        'rubber_type': 'СКТВ-1'
    })

# Загружаем существующие данные
df = pd.read_csv('data/raw/experimental_data.csv')

# Добавляем все новые записи
for record in new_records:
    # Проверяем, есть ли уже такая запись
    exists = df[
        (df['filler_type'] == record['filler_type']) & 
        (df['filler_amount'] == record['filler_amount']) & 
        (df['silane_content'] == record['silane_content'])
    ]
    if len(exists) == 0:
        df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)

# Сохраняем
df.to_csv('data/raw/experimental_data.csv', index=False)

print(f'\n✅ Добавлены все данные из Excel')
print(f'📊 Всего записей: {len(df)}')

# Показываем статистику
print('\n📊 Данные по P_strength (ceramic_strength) для разных наполнителей:')
print(df[['filler_type', 'filler_amount', 'P_strength']].drop_duplicates().sort_values('filler_type'))
