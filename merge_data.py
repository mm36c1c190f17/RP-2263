import pandas as pd
import numpy as np

print("=" * 60)
print("ОБЪЕДИНЕНИЕ ДАННЫХ ИЗ РАЗНЫХ ИСТОЧНИКОВ")
print("=" * 60)

# 1. Загружаем старые данные (6 записей)
print("\n📂 Загружаем старые данные...")
old_data = pd.read_csv('data/raw/experimental_data.csv')
print(f"   Старых записей: {len(old_data)}")
print(f"   Колонки: {old_data.columns.tolist()}")

# 2. Загружаем новые данные (9 записей)
print("\n📂 Загружаем новые данные из Excel...")
new_data = pd.read_csv('data/raw/rubber_data_extended.csv')
print(f"   Новых записей: {len(new_data)}")
print(f"   Колонки: {new_data.columns.tolist()}")

# 3. Приводим новые данные к старому формату
print("\n🔄 Приводим новые данные к старому формату...")

# Маппинг колонок
column_mapping = {
    'filler_type': 'filler_type',
    'silane_content': 'silane_content',  # В новых данных нет, добавим 0
    'filler_amount': 'filler_content',
    'fp_before': 'strength_initial',
    'ep_before': 'elongation_initial',
    'fp_after': 'strength_aged_240h_250C',
    'ep_after': 'elongation_aged_240h_250C',
    'rho_before': 'resistivity',
    'rho_after': 'resistivity',  # В новых данных нет отдельного rho_after
    'P_strength': 'ceramic_strength',
    'manufacturer': 'base_manufacturer',
    'rubber_type': 'base_type'
}

# Создаем DataFrame в старом формате
old_format_data = pd.DataFrame()

# Копируем существующие колонки
old_format_data['filler_type'] = new_data['filler_type']
old_format_data['silane_content'] = 0  # В новых данных нет силана
old_format_data['filler_amount'] = new_data['filler_content']
old_format_data['fp_before'] = new_data['strength_initial']
old_format_data['ep_before'] = new_data['elongation_initial']
old_format_data['fp_after'] = new_data['strength_aged_240h_250C']
old_format_data['ep_after'] = new_data['elongation_aged_240h_250C']
old_format_data['rho_before'] = new_data['resistivity']
old_format_data['rho_after'] = new_data['resistivity'] * 0.8  # Примерное преобразование
old_format_data['P_strength'] = new_data['ceramic_strength']
old_format_data['manufacturer'] = 'Дубна'  # Значение по умолчанию
old_format_data['rubber_type'] = 'СКТВ-1'  # Значение по умолчанию

print(f"   Преобразовано {len(old_format_data)} записей")

# 4. Объединяем данные
print("\n📊 Объединяем данные...")
combined_data = pd.concat([old_data, old_format_data], ignore_index=True)
print(f"   Всего записей: {len(combined_data)}")

# 5. Сохраняем объединенные данные
combined_data.to_csv('data/raw/experimental_data.csv', index=False)
print("\n✅ Объединенные данные сохранены в data/raw/experimental_data.csv")

# 6. Показываем статистику
print("\n📊 Статистика объединенных данных:")
print("=" * 60)
print(f"Всего записей: {len(combined_data)}")
print(f"\nНаполнители:")
print(combined_data['filler_type'].value_counts())

print(f"\nДиапазоны значений:")
print(f"fp_before: {combined_data['fp_before'].min():.1f} - {combined_data['fp_before'].max():.1f} МПа")
print(f"ep_before: {combined_data['ep_before'].min():.0f} - {combined_data['ep_before'].max():.0f} %")
print(f"P_strength: {combined_data['P_strength'].min():.0f} - {combined_data['P_strength'].max():.0f} Н/м²")

print("\n✅ Готово! Теперь запустите python train.py для переобучения модели")
