import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor

# ============================================================
#  НАСТРОЙКА СТРАНИЦЫ
# ============================================================
st.set_page_config(
    page_title="Расчёт характеристик эластомерных композиций",
    page_icon="⚙️",
    layout="centered"
)

# ============================================================
#  CSS ДЛЯ СТРОГОГО КОРПОРАТИВНОГО ДИЗАЙНА
# ============================================================
st.markdown("""
<style>
    /* Основной фон и шрифт */
    .main {
        background-color: #ffffff;
    }
    
    /* Заголовок страницы */
    h1 {
        font-weight: 400 !important;
        font-size: 1.6rem !important;
        color: #1a1a2e !important;
        border-bottom: 2px solid #e8e8e8;
        padding-bottom: 0.8rem;
        margin-bottom: 1.5rem !important;
        letter-spacing: 0.3px;
    }
    
    /* Подзаголовки */
    h2, h3 {
        font-weight: 400 !important;
        color: #2d3436 !important;
    }
    
    h3 {
        font-size: 1.0rem !important;
        margin-bottom: 0.5rem !important;
    }
    
    /* Кнопка */
    .stButton > button {
        background-color: #2d3436 !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 3px !important;
        padding: 0.6rem 2rem !important;
        font-weight: 400 !important;
        font-size: 0.9rem !important;
        width: 100%;
        transition: background-color 0.2s;
    }
    .stButton > button:hover {
        background-color: #1a1a2e !important;
        color: #ffffff !important;
    }
    
    /* Сообщения об успехе/ошибке */
    .stAlert {
        background-color: #f8f9fa !important;
        border-left: 3px solid #2d3436 !important;
        color: #2d3436 !important;
        padding: 0.6rem 1rem !important;
    }
    .stAlert svg {
        display: none !important;
    }
    
    /* Метрики */
    [data-testid="metric-container"] {
        background-color: #f8f9fa;
        border: 1px solid #e8e8e8;
        border-radius: 3px;
        padding: 1.2rem 0.5rem;
        text-align: center;
    }
    [data-testid="metric-container"] label {
        font-weight: 400 !important;
        color: #555 !important;
        font-size: 0.85rem !important;
    }
    [data-testid="metric-container"] .stMetricValue {
        font-size: 1.6rem !important;
        font-weight: 400 !important;
        color: #1a1a2e !important;
    }
    
    /* Разделитель */
    hr {
        margin: 1.8rem 0 !important;
        border-color: #e8e8e8 !important;
    }
    
    /* Радио-кнопки */
    .stRadio > div {
        gap: 1.5rem !important;
    }
    .stRadio label {
        font-size: 0.9rem !important;
        color: #555 !important;
    }
    
    /* Поля ввода */
    .stNumberInput label, .stFileUploader label {
        font-size: 0.85rem !important;
        color: #555 !important;
        font-weight: 400 !important;
    }
    
    /* Подвал */
    .footer {
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #e8e8e8;
        font-size: 0.75rem;
        color: #999;
        text-align: center;
        letter-spacing: 0.3px;
    }
    .footer a {
        color: #999;
        text-decoration: none;
    }
    .footer a:hover {
        color: #555;
        text-decoration: underline;
    }
    
    /* Скрыть лишние элементы */
    .stSpinner > div {
        border-color: #2d3436 !important;
    }
    .stCaption {
        font-size: 0.8rem !important;
        color: #888 !important;
    }
    
    /* Контейнер для данных */
    .data-container {
        background-color: #f8f9fa;
        padding: 1.2rem;
        border-radius: 3px;
        border: 1px solid #e8e8e8;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
#  ЗАГОЛОВОК
# ============================================================
st.markdown("""
<h1 style="font-weight: 300; font-size: 1.0rem; color: #888; border-bottom: none; margin-bottom: 0; padding-bottom: 0;">
    Модель машинного обучения дя пронозирования свойств эластомерных материалов
</h1>
<h1>
    Характеристики эластомерных материалов
</h1>
""", unsafe_allow_html=True)

st.caption("Модель прогнозирования на основе рецептуры")

# ============================================================
#  ЗАГРУЗКА МОДЕЛЕЙ
# ============================================================
@st.cache_resource
def load_models():
    try:
        model_strength = CatBoostRegressor()
        model_strength.load_model('models/catboost_strength.cbm')
        model_elongation = CatBoostRegressor()
        model_elongation.load_model('models/catboost_elongation.cbm')
        return model_strength, model_elongation, True
    except Exception:
        return None, None, False

model_strength, model_elongation, models_loaded = load_models()

if not models_loaded:
    st.error("Не удалось загрузить модели. Проверьте наличие файлов в каталоге models/")
    st.stop()

# ============================================================
#  ИНТЕРФЕЙС ВВОДА
# ============================================================
st.markdown("---")

input_mode = st.radio(
    "Способ ввода данных",
    ["Ручной ввод", "Загрузка из файла"],
    horizontal=True,
    index=0
)

st.markdown("---")

# ---- Режим ручного ввода ----
if input_mode == "Ручной ввод":
    
    st.markdown("##### Состав композиции")
    col1, col2, col3 = st.columns(3)
    with col1:
        component_a = st.number_input(
            "Компонент A",
            min_value=0.0,
            max_value=200.0,
            value=33.0,
            step=1.0,
            help="Содержание в частях на 100 частей каучука (phr)"
        )
    with col2:
        component_b = st.number_input(
            "Компонент B",
            min_value=0.0,
            max_value=200.0,
            value=42.0,
            step=1.0
        )
    with col3:
        component_c = st.number_input(
            "Компонент C",
            min_value=0.0,
            max_value=200.0,
            value=17.0,
            step=1.0
        )
    
    st.markdown("##### Режим вулканизации")
    col1, col2 = st.columns(2)
    with col1:
        temperature = st.number_input(
            "Температура",
            min_value=100,
            max_value=250,
            value=195,
            step=1,
            help="Температура вулканизации, °C"
        )
    with col2:
        time = st.number_input(
            "Продолжительность",
            min_value=1,
            max_value=120,
            value=54,
            step=1,
            help="Продолжительность вулканизации, мин"
        )
    
    st.markdown("---")
    
    if st.button("Выполнить расчёт", type="primary"):
        input_data = pd.DataFrame({
            'ingredient_a': [component_a],
            'ingredient_b': [component_b],
            'ingredient_c': [component_c],
            'temperature': [temperature],
            'time': [time]
        })
        
        with st.spinner("Расчёт..."):
            pred_strength = model_strength.predict(input_data)[0]
            pred_elongation = model_elongation.predict(input_data)[0]
        
        st.markdown("---")
        st.markdown("##### Результаты расчёта")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                label="Прочность при растяжении",
                value=f"{pred_strength:.1f} МПа"
            )
        with col2:
            st.metric(
                label="Относительное удлинение",
                value=f"{pred_elongation:.1f} %"
            )
        
        st.caption("Прогноз получен на основе модели Podlesskiy")

# ---- Режим загрузки файла ----
else:
    st.caption("Поддерживаются форматы .xlsx, .xls, .csv")
    
    uploaded_file = st.file_uploader(
        "Выберите файл",
        type=['xlsx', 'xls', 'csv'],
        label_visibility="collapsed"
    )
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            required = ['ingredient_a', 'ingredient_b', 'ingredient_c', 'temperature', 'time']
            missing = [c for c in required if c not in df.columns]
            
            if missing:
                st.error(f"Отсутствуют столбцы: {', '.join(missing)}")
                st.info(f"Доступные столбцы: {', '.join(df.columns)}")
            else:
                st.markdown("##### Данные для расчёта")
                st.dataframe(df[required], use_container_width=True, hide_index=True)
                
                if st.button("Выполнить расчёт для всех записей", type="primary"):
                    with st.spinner("Расчёт..."):
                        df['Прочность_МПа'] = model_strength.predict(df[required])
                        df['Удлинение_процент'] = model_elongation.predict(df[required])
                    
                    st.success("Расчёт завершён")
                    
                    result_cols = required + ['Прочность_МПа', 'Удлинение_процент']
                    st.dataframe(
                        df[result_cols],
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Прочность_МПа": st.column_config.NumberColumn("Прочность, МПа", format="%.1f"),
                            "Удлинение_процент": st.column_config.NumberColumn("Удлинение, %", format="%.1f")
                        }
                    )
                    
                    csv_data = df[result_cols].to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Скачать результаты (CSV)",
                        data=csv_data,
                        file_name="results.csv",
                        mime="text/csv"
                    )
        except Exception as e:
            st.error(f"Ошибка: {e}")

# ============================================================
#  ПОДВАЛ
# ============================================================
st.markdown("""
<div class="footer">
    <span>Версия 1.0 &bull; Алгоритм: CatBoost &bull; 
    <a href="#">Документация</a> &bull; 
    <a href="#">Поддержка</a></span>
</div>
""", unsafe_allow_html=True)