import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor

# ==================== НАСТРОЙКА СТРАНИЦЫ ====================
st.set_page_config(
    page_title="Резиновый калькулятор",
    page_icon="🔬",
    layout="centered"
)

st.title("🔬 Предсказание свойств резиновых смесей")
st.markdown("Загрузи свою модель и получи предсказание прочности и эластичности")

# ==================== ЗАГРУЗКА МОДЕЛЕЙ ====================
@st.cache_resource
def load_models():
    """Загружает сохранённые модели"""
    try:
        model_strength = CatBoostRegressor()
        model_strength.load_model('models/catboost_strength.cbm')
        
        model_elongation = CatBoostRegressor()
        model_elongation.load_model('models/catboost_elongation.cbm')
        
        return model_strength, model_elongation, True
    except Exception as e:
        return None, None, False

model_strength, model_elongation, models_loaded = load_models()

if not models_loaded:
    st.error("❌ Не удалось загрузить модели. Убедись, что файлы .cbm лежат в папке models/")
    st.stop()

st.success("✅ Модели загружены успешно!")

# ==================== ВВОД ДАННЫХ ====================
st.subheader("📝 Введите параметры рецептуры")

# Два способа ввода: ручной или загрузка из файла
input_method = st.radio(
    "Выберите способ ввода данных:",
    ["Ввести вручную", "Загрузить Excel-файл"]
)

if input_method == "Ввести вручную":
    col1, col2 = st.columns(2)
    
    with col1:
        ingredient_a = st.number_input("Ингредиент A", min_value=0.0, max_value=100.0, value=33.0, step=1.0)
        ingredient_b = st.number_input("Ингредиент B", min_value=0.0, max_value=100.0, value=42.0, step=1.0)
        ingredient_c = st.number_input("Ингредиент C", min_value=0.0, max_value=100.0, value=17.0, step=1.0)
    
    with col2:
        temperature = st.number_input("Температура вулканизации, °C", min_value=100, max_value=250, value=195, step=1)
        time = st.number_input("Время вулканизации, мин", min_value=1, max_value=120, value=54, step=1)
    
    # Кнопка для расчёта
    if st.button("🔮 Предсказать свойства", type="primary"):
        # Собираем данные для модели
        input_data = pd.DataFrame({
            'ingredient_a': [ingredient_a],
            'ingredient_b': [ingredient_b],
            'ingredient_c': [ingredient_c],
            'temperature': [temperature],
            'time': [time]
        })
        
        # Предсказываем
        with st.spinner("Идёт расчёт..."):
            pred_strength = model_strength.predict(input_data)[0]
            pred_elongation = model_elongation.predict(input_data)[0]
        
        # Показываем результаты
        st.success("✅ Расчёт завершён!")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("📊 Прочность", f"{pred_strength:.1f} МПа")
        with col2:
            st.metric("📊 Эластичность", f"{pred_elongation:.1f} %")

else:  # Загрузка файла
    st.info("📂 Загрузите Excel-файл с рецептурами (первые строки должны содержать ингредиенты)")
    
    uploaded_file = st.file_uploader("Выберите Excel-файл", type=['xlsx', 'csv'])
    
    if uploaded_file is not None:
        try:
            # Определяем тип файла
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.write("📋 Загруженные данные:")
            st.dataframe(df.head())
            
            # Проверяем, что есть нужные колонки
            required_cols = ['ingredient_a', 'ingredient_b', 'ingredient_c', 'temperature', 'time']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                st.error(f"❌ В файле нет столбцов: {missing_cols}")
                st.info("Доступные столбцы: " + ", ".join(df.columns))
            else:
                if st.button("🔮 Предсказать для всех строк", type="primary"):
                    with st.spinner("Идёт расчёт..."):
                        df['pred_strength'] = model_strength.predict(df[required_cols])
                        df['pred_elongation'] = model_elongation.predict(df[required_cols])
                    
                    st.success("✅ Расчёт завершён!")
                    
                    # Показываем результат
                    result_cols = required_cols + ['pred_strength', 'pred_elongation']
                    st.dataframe(df[result_cols])
                    
                    # Кнопка скачивания
                    csv = df[result_cols].to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Скачать результат в CSV",
                        data=csv,
                        file_name="predictions.csv",
                        mime="text/csv"
                    )
        except Exception as e:
            st.error(f"❌ Ошибка при загрузке файла: {e}")

# ==================== ИНФОРМАЦИЯ О ПРОЕКТЕ ====================
with st.expander("ℹ️ О модели"):
    st.markdown("""
    **Модель обучена на данных о резиновых смесях.**
    
    - **Алгоритм:** CatBoost Regressor
    - **Предсказываемые свойства:** прочность (МПа) и эластичность (%)
    - **Входные данные:** ингредиенты (A, B, C), температура и время вулканизации
    
    ⚠️ **Важно:** Модель работает только в диапазоне данных, на которых была обучена.
    """
    )