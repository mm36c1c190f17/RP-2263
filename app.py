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
# === ЛОГОТИП ===
col1, col2, col3 = st.columns([1, 10, 1])  # создаём колонки
with col1:
    st.image('static/logo.png', width=160)  # ← путь к файлу и размер
# ============================================================
#  CSS ДЛЯ СТРОГОГО ДИЗАЙНА
# ============================================================
st.markdown("""
<style>
    .main { background-color: #ffffff; }
    h1 {
        font-weight: 400 !important;
        font-size: 1.6rem !important;
        color: #1a1a2e !important;
        border-bottom: 2px solid #e8e8e8;
        padding-bottom: 0.8rem;
        margin-bottom: 1.5rem !important;
    }
    .stButton > button {
        background-color: #2d3436 !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 3px !important;
        padding: 0.6rem 2rem !important;
        font-weight: 400 !important;
        width: 100%;
    }
    .stButton > button:hover {
        background-color: #1a1a2e !important;
    }
    [data-testid="metric-container"] {
        background-color: #f8f9fa;
        border: 1px solid #e8e8e8;
        border-radius: 3px;
        padding: 1rem 0.5rem;
        text-align: center;
    }
    [data-testid="metric-container"] label {
        font-weight: 400 !important;
        color: #555 !important;
        font-size: 0.8rem !important;
    }
    .footer {
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #e8e8e8;
        font-size: 0.75rem;
        color: #999;
        text-align: center;
    }
    .stAlert {
        background-color: #f8f9fa !important;
        border-left: 3px solid #2d3436 !important;
        color: #2d3436 !important;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
#  ЗАГОЛОВОК
# ============================================================
st.markdown("""
<h1 style="font-weight: 300; font-size: 1.0rem; color: #888; border-bottom: none; margin-bottom: 0; padding-bottom: 0;">
    Расчет характеристик эластомерных материалов
</h1>
""", unsafe_allow_html=True)

st.caption("Модель прогнозирования на основе состава и режима вулканизации")

# ============================================================
#  ЗАГРУЗКА МОДЕЛЕЙ
# ============================================================
@st.cache_resource
def load_models():
    models = {}
    model_files = {
        'strength_initial': 'catboost_strength_initial.cbm',
        'elongation_initial': 'catboost_elongation_initial.cbm'
    }
    
    for name, filename in model_files.items():
        try:
            model = CatBoostRegressor()
            model.load_model(f'models/{filename}')
            models[name] = model
        except Exception as e:
            st.warning(f"Не удалось загрузить {name}: {e}")
    
    return models

models = load_models()

if len(models) == 0:
    st.error("❌ Не удалось загрузить модели. Проверьте наличие файлов в папке models/")
    st.stop()

st.success(f"✅ Загружено {len(models)} моделей")

# ============================================================
#  ИНТЕРФЕЙС ВВОДА
# ============================================================
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.markdown("##### Состав композиции (мас. ч.)")
    component_a = st.number_input("Компонент A", min_value=0.0, max_value=200.0, value=0.0, step=1.0)
    component_b = st.number_input("Компонент B", min_value=0.0, max_value=200.0, value=0.0, step=1.0)
    component_c = st.number_input("Компонент C", min_value=0.0, max_value=200.0, value=0.0, step=1.0)

with col2:
    st.markdown("##### Режим вулканизации")
    temperature = st.number_input("Температура, °C", min_value=0, max_value=250, value=0, step=1)
    time = st.number_input("Продолжительность, мин", min_value=0, max_value=180, value=0, step=1)

st.markdown("---")

if st.button("Выполнить расчёт", type="primary"):
    input_data = pd.DataFrame({
        'component_a': [component_a],
        'component_b': [component_b],
        'component_c': [component_c],
        'temp': [temperature],
        'time': [time]
    })
    
    with st.spinner("Расчёт..."):
        results = {}
        for name, model in models.items():
            try:
                results[name] = model.predict(input_data)[0]
            except Exception:
                results[name] = None
    
    st.markdown("---")
    st.markdown("##### Результаты расчёта")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'strength_initial' in results and results['strength_initial'] is not None:
            st.metric("Прочность при растяжении", f"{results['strength_initial']:.1f} МПа")
    with col2:
        if 'elongation_initial' in results and results['elongation_initial'] is not None:
            st.metric("Относительное удлинение", f"{results['elongation_initial']:.1f} %")
    
    st.caption("Прогноз получен на основе модели Podlesskiy")

# ============================================================
#  ПОДВАЛ
# ============================================================
st.markdown("""
<div class="footer">
    <span>Версия 1.0.1 &bull; Алгоритм: CatBoost &bull; 
    <a href="#" style="color: #999; text-decoration: none;">Документация</a></span>
</div>
""", unsafe_allow_html=True)