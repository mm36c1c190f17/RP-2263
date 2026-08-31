import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Прогнозирование свойств эластомерных материалов",
    page_icon="⚙️",
    layout="wide"
)

st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    h1 { font-weight: 300; font-size: 2rem; color: #1a1a2e; border-bottom: 3px solid #4361ee; padding-bottom: 0.5rem; }
    .stButton > button { background-color: #4361ee; color: #ffffff; border: none; border-radius: 4px; padding: 0.6rem 2rem; font-weight: 500; width: 100%; }
    .stButton > button:hover { background-color: #3a56d4; box-shadow: 0 4px 12px rgba(67, 97, 238, 0.3); }
    [data-testid="metric-container"] { background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%); border: 1px solid #e9ecef; border-radius: 8px; padding: 1.2rem 0.5rem; }
    .footer { margin-top: 3rem; padding-top: 1rem; border-top: 2px solid #4361ee; font-size: 0.75rem; color: #6c757d; text-align: center; }
    .section-header { color: #1a1a2e; font-weight: 500; font-size: 1.3rem; border-bottom: 2px solid #4361ee; padding-bottom: 0.3rem; margin-bottom: 1.5rem; }
</style>
""", unsafe_allow_html=True)

st.title("Прогнозирование свойств эластомерных материалов")
st.caption("Система прогнозирования на основе машинного обучения")

@st.cache_resource
def load_model():
    try:
        model = CatBoostRegressor()
        model.load_model('models/catboost_predictor.cbm')
        feature_names = joblib.load('models/feature_names.pkl')
        cat_features = joblib.load('models/cat_features.pkl')
        df = pd.read_csv('data/raw/experimental_data.csv')
        return model, feature_names, cat_features, df
    except Exception as e:
        st.error(f"Ошибка загрузки модели: {e}")
        return None, None, None, None

model, feature_names, cat_features, data_df = load_model()

if model is None:
    st.warning("Модель не загружена")
    st.stop()

st.sidebar.markdown("### Управление")
st.sidebar.write(f"**Записей для обучения:** {len(data_df)}")
st.sidebar.write(f"**Наполнителей:** {len(data_df['filler_type'].unique())}")

tab1, tab2 = st.tabs(["Прогнозирование", "Данные"])

with tab1:
    st.markdown('<p class="section-header">Параметры состава</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fillers = sorted(data_df['filler_type'].unique())
        filler_type = st.selectbox("Тип наполнителя", fillers)
        filler_content = st.number_input("Дозировка наполнителя (phr)", min_value=10, max_value=50, value=30, step=1)
        silane_content = st.number_input("Содержание силана (phr)", min_value=0, max_value=10, value=0, step=1)
    
    with col2:
        st.markdown("### Состав композиции")
        st.info(f"**Каучук:** VMQ (Xiameter, твердость 70)")
        st.info(f"**Наполнитель:** {filler_type} ({filler_content} phr)")
        st.info(f"**Силан:** {silane_content} phr")
        
        if st.button("Рассчитать свойства", use_container_width=True):
            with st.spinner("Выполняется расчет..."):
                try:
                    input_data = pd.DataFrame([{
                        'base_type': 'VMQ',
                        'base_manufacturer': 'Xiameter',
                        'filler_type': filler_type,
                        'filler_manufacturer': 'JSC_Vostochnye_Ogneupory',
                        'silane_type': 'A-1120' if silane_content > 0 else '0',
                        'base_hardness': 70,
                        'filler_content': filler_content,
                        'silane_content': silane_content,
                        'temp': 115,
                        'time': 15
                    }])
                    
                    input_data = input_data[feature_names]
                    pred = model.predict(input_data)[0]
                    
                    st.success("✅ Расчет выполнен")
                    st.metric("Прочность керамического остатка", f"{pred:.1f} Н/м²")
                    
                    # Дополнительная информация
                    st.caption(f"Модель обучена на {len(data_df)} экспериментальных образцах")
                    
                except Exception as e:
                    st.error(f"Ошибка расчета: {e}")

with tab2:
    st.markdown('<p class="section-header">Экспериментальные данные</p>', unsafe_allow_html=True)
    st.dataframe(data_df, use_container_width=True)

st.markdown("""
<div class="footer">
    Система прогнозирования свойств эластомерных материалов &bull; Алгоритм: CatBoost
</div>
""", unsafe_allow_html=True)
