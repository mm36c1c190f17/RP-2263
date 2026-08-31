import joblib

# Загружаем таблицу поиска
lookup = joblib.load('models/lookup_table.pkl')

print('🔮 БЫСТРЫЙ ПОИСК СВОЙСТВ')
print('=' * 60)

# Проверяем все известные составы
for key, values in lookup.items():
    filler, content, silane = key
    ceramic = values.get('ceramic_strength')
    if ceramic is not None:
        print(f'{filler} {content} phr + {silane} phr силан: {ceramic:.1f} Н/м²')
