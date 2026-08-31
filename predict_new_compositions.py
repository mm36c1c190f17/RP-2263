import pandas as pd
import numpy as np
import joblib

# Создаем базу данных из экспериментальных значений
experimental_data = {
    'шпинель': {
        'без_силана': {
            'fp_до': 8.0, 'ep_до': 350,
            'fp_после': 4.30, 'ep_после': 100,
            'rho_до': 4.2e13, 'rho_после': 6.7e12,
            'P': 95
        },
        'с_силаном': {
            'fp_до': 6.2, 'ep_до': 400,
            'fp_после': 4.0, 'ep_после': 90,
            'rho_до': 4.0e13, 'rho_после': 6.6e12,
            'P': 65
        }
    },
    'волластонит': {
        'без_силана': {
            'fp_до': 8.0, 'ep_до': 350,
            'fp_после': 4.3, 'ep_после': 84,
            'rho_до': 3.8e13, 'rho_после': 1.7e12,
            'P': 250
        },
        'с_силаном': {
            'fp_до': 7.1, 'ep_до': 440,
            'fp_после': 4.1, 'ep_после': 78,
            'rho_до': 3.9e13, 'rho_после': 1.5e12,
            'P': 215
        }
    },
    'Al2O3': {
        'без_силана': {
            'fp_до': 5.7, 'ep_до': 230,
            'fp_после': 5.4, 'ep_после': 140,
            'rho_до': 4.0e13, 'rho_после': 0.8e12,
            'P': 230
        },
        'с_силаном': {
            'fp_до': 4.9, 'ep_до': 280,
            'fp_после': 5.3, 'ep_после': 110,
            'rho_до': 4.2e13, 'rho_после': 1.0e12,
            'P': 170
        }
    }
}

def predict_composition(filler_type, silane_present):
    """
    Предсказание свойств на основе экспериментальных данных
    filler_type: 'шпинель', 'волластонит', 'Al2O3'
    silane_present: True/False
    """
    silane_key = 'с_силаном' if silane_present else 'без_силана'
    
    if filler_type not in experimental_data:
        raise ValueError(f"Неизвестный наполнитель: {filler_type}")
    
    if silane_key not in experimental_data[filler_type]:
        raise ValueError(f"Неизвестный вариант силана: {silane_key}")
    
    return experimental_data[filler_type][silane_key]

def predict_with_uncertainty(filler_type, silane_present, uncertainty_factor=0.1):
    """
    Предсказание с учетом погрешности
    uncertainty_factor - доля погрешности от значения
    """
    base_pred = predict_composition(filler_type, silane_present)
    
    # Добавляем случайную погрешность в пределах ±uncertainty_factor
    np.random.seed(42)  # для воспроизводимости
    result = {}
    for key, value in base_pred.items():
        error = value * uncertainty_factor * np.random.uniform(-1, 1)
        result[key] = value + error
    
    return result

# Функция для сравнения составов
def compare_compositions(filler1, silane1, filler2, silane2):
    """Сравнение двух составов"""
    pred1 = predict_composition(filler1, silane1)
    pred2 = predict_composition(filler2, silane2)
    
    print(f"\n{'='*60}")
    print(f"Сравнение: {filler1} ({'с силаном' if silane1 else 'без силана'}) vs {filler2} ({'с силаном' if silane2 else 'без силана'})")
    print(f"{'='*60}")
    
    for key in pred1.keys():
        val1 = pred1[key]
        val2 = pred2[key]
        diff = ((val2 - val1) / val1) * 100
        print(f"{key:15} | {val1:>12.4e} | {val2:>12.4e} | Изменение: {diff:>6.1f}%")

# Анализ влияния силана
def analyze_silane_effect():
    """Анализ влияния силана на свойства"""
    fillers = ['шпинель', 'волластонит', 'Al2O3']
    
    print(f"\n{'='*70}")
    print("Анализ влияния силана А-1120 на свойства")
    print(f"{'='*70}")
    
    for filler in fillers:
        print(f"\n{filler.upper()}:")
        print("-" * 50)
        
        no_silane = predict_composition(filler, False)
        with_silane = predict_composition(filler, True)
        
        print(f"{'Параметр':<20} | {'без силана':>12} | {'с силаном':>12} | Изменение")
        for key in no_silane.keys():
            val_no = no_silane[key]
            val_with = with_silane[key]
            change = ((val_with - val_no) / val_no) * 100
            print(f"{key:<20} | {val_no:>12.4e} | {val_with:>12.4e} | {change:>6.1f}%")

# Анализ сравнения наполнителей
def analyze_fillers():
    """Сравнение разных наполнителей"""
    fillers = ['шпинель', 'волластонит', 'Al2O3']
    silane_options = [False, True]
    
    print(f"\n{'='*70}")
    print("Сравнение наполнителей (без силана)")
    print(f"{'='*70}")
    
    for silane in silane_options:
        silane_text = "с силаном" if silane else "без силана"
        print(f"\n{silane_text}:")
        print("-" * 70)
        
        # Собираем данные для всех наполнителей
        data = {}
        for filler in fillers:
            data[filler] = predict_composition(filler, silane)
        
        # Выводим в виде таблицы
        print(f"{'Параметр':<20}", end="")
        for filler in fillers:
            print(f" | {filler:>12}", end="")
        print()
        print("-" * 70)
        
        for param in data[fillers[0]].keys():
            print(f"{param:<20}", end="")
            for filler in fillers:
                print(f" | {data[filler][param]:>12.4e}", end="")
            print()

# Поиск оптимального состава по критериям
def find_best_composition(criteria):
    """
    Поиск состава, оптимизирующего заданные критерии
    criteria: словарь {параметр: 'max' или 'min'}
    """
    fillers = ['шпинель', 'волластонит', 'Al2O3']
    silane_options = [True, False]
    
    best_score = -np.inf
    best_composition = None
    
    for filler in fillers:
        for silane in silane_options:
            props = predict_composition(filler, silane)
            
            # Вычисляем общий балл
            score = 0
            for param, direction in criteria.items():
                value = props.get(param, 0)
                if direction == 'max':
                    score += value
                elif direction == 'min':
                    score -= value
            
            if score > best_score:
                best_score = score
                best_composition = {
                    'наполнитель': filler,
                    'силан': 'с силаном' if silane else 'без силана',
                    'свойства': props,
                    'score': score
                }
    
    return best_composition

# Основная программа
if __name__ == "__main__":
    print("=" * 70)
    print("СИСТЕМА ПРОГНОЗИРОВАНИЯ СВОЙСТВ РЕЗИН НА ОСНОВЕ СИЛИКОНОВОГО КАУЧУКА")
    print("=" * 70)
    
    # 1. Показываем влияние силана
    analyze_silane_effect()
    
    # 2. Сравниваем наполнители
    analyze_fillers()
    
    # 3. Поиск оптимального состава для максимальной прочности
    print("\n" + "=" * 70)
    print("ПОИСК ОПТИМАЛЬНОГО СОСТАВА")
    print("=" * 70)
    
    # Максимизируем прочность при сохранении эластичности
    criteria = {
        'P': 'max',  # максимальная прочность
        'fp_до': 'max',  # максимальная прочность до старения
        'fp_после': 'max',  # максимальная прочность после старения
        'ep_до': 'max'  # максимальная эластичность
    }
    
    best = find_best_composition(criteria)
    print(f"\nОптимальный состав по критериям:")
    print(f"Наполнитель: {best['наполнитель']}")
    print(f"Силан: {best['силан']}")
    print(f"Score: {best['score']:.1f}")
    print("\nПрогнозируемые свойства:")
    for param, value in best['свойства'].items():
        print(f"  {param}: {value:.4e}")
    
    # 4. Сравнение конкретных рецептур
    print("\n" + "=" * 70)
    print("СРАВНЕНИЕ КОНКРЕТНЫХ РЕЦЕПТУР")
    print("=" * 70)
    
    compare_compositions('шпинель', True, 'волластонит', True)
    compare_compositions('шпинель', True, 'Al2O3', True)
    compare_compositions('волластонит', True, 'Al2O3', True)
    
    # 5. Создаем рекомендации
    print("\n" + "=" * 70)
    print("РЕКОМЕНДАЦИИ ПО ВЫБОРУ НАПОЛНИТЕЛЯ")
    print("=" * 70)
    
    recommendations = {
        'Высокая прочность': 'Волластонит (даже без силана дает P=250 Н/м²)',
        'Высокая эластичность': 'Волластонит с силаном (ε=440%)',
        'Диэлектрические свойства': 'Шпинель (ρv=4.2·10¹³ Ом·м)',
        'Стабильность после старения': 'Al₂O₃ (наименьшее падение свойств)',
        'Водостойкость': 'Шпинель (наименьшее падение ρv после воды)'
    }
    
    for criteria, rec in recommendations.items():
        print(f"\n{criteria}: {rec}")
    
    print("\n" + "=" * 70)
    print("Рекомендуемый универсальный состав: ВОЛЛАСТОНИТ С СИЛАНОМ")
    print("Обеспечивает баланс прочности (P=215), эластичности (ε=440%)")
    print("и диэлектрических свойств")
    print("=" * 70)