import pandas as pd
import numpy as np
from catboost import CatBoostRegressor

print("=" * 60)
print("ПРЕДСКАЗАНИЕ СВОЙСТВ ДЛЯ НОВЫХ РЕЦЕПТУР")
print("=" * 60)

# 1. Загружаем обученные модели
print("\n1. Загружаем модели...")
model_strength = CatBoostRegressor()
model_strength.load_model('models/catboost_strength.cbm')

model_elongation = CatBoostRegressor()
model_elongation.load_model('models/catboost_elongation.cbm')
print("   ✅ Модели загружены успешно!")

# 2. Создаём 3 новые рецептуры (которых нет в обучающих данных)
print("\n2. Создаём 3 новые рецептуры для тестирования...")

new_recipes = pd.DataFrame({
    'ingredient_a': [33, 14, 26],
    'ingredient_b': [42, 20, 36],
    'ingredient_c': [17, 5, 12],
    'temperature': [195, 148, 168],
    'time': [54, 28, 40]
})

print("\n   Рецептура #1:")
print(f"     ingredient_a: {new_recipes.iloc[0]['ingredient_a']}")
print(f"     ingredient_b: {new_recipes.iloc[0]['ingredient_b']}")
print(f"     ingredient_c: {new_recipes.iloc[0]['ingredient_c']}")
print(f"     температура: {new_recipes.iloc[0]['temperature']}°C")
print(f"     время: {new_recipes.iloc[0]['time']} мин")

print("\n   Рецептура #2:")
print(f"     ingredient_a: {new_recipes.iloc[1]['ingredient_a']}")
print(f"     ingredient_b: {new_recipes.iloc[1]['ingredient_b']}")
print(f"     ingredient_c: {new_recipes.iloc[1]['ingredient_c']}")
print(f"     температура: {new_recipes.iloc[1]['temperature']}°C")
print(f"     время: {new_recipes.iloc[1]['time']} мин")

print("\n   Рецептура #3:")
print(f"     ingredient_a: {new_recipes.iloc[2]['ingredient_a']}")
print(f"     ingredient_b: {new_recipes.iloc[2]['ingredient_b']}")
print(f"     ingredient_c: {new_recipes.iloc[2]['ingredient_c']}")
print(f"     температура: {new_recipes.iloc[2]['temperature']}°C")
print(f"     время: {new_recipes.iloc[2]['time']} мин")

# 3. Делаем предсказания
print("\n3. Предсказываем свойства...")

pred_strength = model_strength.predict(new_recipes)
pred_elongation = model_elongation.predict(new_recipes)

# 4. Выводим результаты в виде таблицы
print("\n" + "=" * 60)
print("РЕЗУЛЬТАТЫ ПРЕДСКАЗАНИЙ")
print("=" * 60)

print("\n┌────────────┬─────────────────┬────────────────────┐")
print("│ Рецептура  │ Прочность (ед.) │ Эластичность (ед.) │")
print("├────────────┼─────────────────┼────────────────────┤")

for i in range(3):
    print(f"│ #{i+1}        │ {pred_strength[i]:>13.1f}  │ {pred_elongation[i]:>16.1f}  │")

print("└────────────┴─────────────────┴────────────────────┘")

print("\n" + "=" * 60)
print("ВЫВОДЫ:")
print("=" * 60)

# Анализируем результаты
best_strength = np.argmax(pred_strength)
best_elongation = np.argmax(pred_elongation)

print(f"\n✅ Лучшая прочность у рецептуры #{best_strength+1}: {pred_strength[best_strength]:.1f} ед.")
print(f"✅ Лучшая эластичность у рецептуры #{best_elongation+1}: {pred_elongation[best_elongation]:.1f} ед.")

if best_strength == best_elongation:
    print(f"\n🏆 Рецептура #{best_strength+1} - ОПТИМАЛЬНАЯ (лучшая и по прочности, и по эластичности)!")

print("\n" + "=" * 60)
print("Примечание: Это предсказания на основе 15 рецептур.")
print("Для повышения точности добавьте больше реальных данных.")
print("=" * 60)