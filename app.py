import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from scipy.interpolate import interp1d
import base64
from PIL import Image
import os

st.set_page_config(
    page_title="Прогнозирование свойств эластомерных материалов",
    page_icon="⚙️",
    layout="wide"
)

# Функция для загрузки изображения в base64
def get_image_base64(image_path):
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except:
        return None

# Загружаем логотип
logo_base64 = get_image_base64("static/logo.png")

st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .header-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 3px solid #4361ee;
        padding-bottom: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .header-title {
        font-weight: 300;
        font-size: 2rem;
        color: #1a1a2e;
        margin: 0;
        flex: 1;
    }
    .header-logo {
        height: 50px;
        width: auto;
        margin-left: 20px;
    }
    .stButton > button { 
        background-color: #4361ee; 
        color: #ffffff; 
        border: none; 
        border-radius: 4px; 
        padding: 0.6rem 2rem; 
        font-weight: 500; 
        width: 100%;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 10px;
    }
    .stButton > button:hover { 
        background-color: #3a56d4; 
        box-shadow: 0 4px 12px rgba(67, 97, 238, 0.3); 
    }
    [data-testid="metric-container"] { 
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%); 
        border: 1px solid #e9ecef; 
        border-radius: 8px; 
        padding: 1.2rem 0.5rem; 
    }
    .footer { 
        margin-top: 3rem; 
        padding-top: 1rem; 
        border-top: 2px solid #4361ee; 
        font-size: 0.75rem; 
        color: #6c757d; 
        text-align: center; 
    }
    .section-header { 
        color: #1a1a2e; 
        font-weight: 500; 
        font-size: 1.3rem; 
        border-bottom: 2px solid #4361ee; 
        padding-bottom: 0.3rem; 
        margin-bottom: 1.5rem; 
    }
    .btn-logo {
        height: 20px;
        width: auto;
    }
</style>
""", unsafe_allow_html=True)

# Заголовок с логотипом
if logo_base64:
    st.markdown(f"""
    <div class="header-container">
        <h1 class="header-title">Прогнозирование свойств эластомерных материалов</h1>
        <img src="data:image/png;base64,{logo_base64}" class="header-logo" />
    </div>
    """, unsafe_allow_html=True)
else:
    st.title("Прогнозирование свойств эластомерных материалов")

st.caption("Система прогнозирования на основе интерполяции экспериментальных данных")

@st.cache_resource
def load_interpolator():
    try:
        interpolators = joblib.load('models/interpolators.pkl')
        df = pd.read_csv('data/raw/experimental_data.csv')
        return interpolators, df
    except Exception as e:
        st.error(f"Ошибка загрузки: {e}")
        return None, None

interpolators, data_df = load_interpolator()

if interpolators is None:
    st.warning("Интерполятор не загружен")
    st.stop()

st.sidebar.markdown("### Управление")
st.sidebar.write(f"**Записей в базе:** {len(data_df)}")
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
        
        # Кнопка с логотипом
        if logo_base64:
            # Используем HTML для кнопки с логотипом
            if st.button("Рассчитать свойства", use_container_width=True):
                with st.spinner("Выполняется расчет..."):
                    try:
                        if filler_type not in interpolators:
                            st.error(f"Нет данных для наполнителя {filler_type}")
                        else:
                            points = interpolators[filler_type]['points']
                            same_silane = [p for p in points if p['silane'] == silane_content]
                            
                            if len(same_silane) >= 2:
                                x = [p['content'] for p in same_silane]
                                y = [p['ceramic_strength'] for p in same_silane]
                                interp = interp1d(x, y, kind='linear', fill_value='extrapolate')
                                pred = float(interp(filler_content))
                                st.success("✅ Расчет выполнен (интерполяция)")
                            else:
                                x = [p['content'] for p in points]
                                y = [p['ceramic_strength'] for p in points]
                                interp = interp1d(x, y, kind='linear', fill_value='extrapolate')
                                pred = float(interp(filler_content))
                                st.success("✅ Расчет выполнен (экстраполяция)")
                            
                            st.metric("Прочность керамического остатка", f"{pred:.1f} Н/м²")
                            
                            st.caption("Использованные экспериментальные точки:")
                            for p in points:
                                st.caption(f"  {p['content']} phr + {p['silane']} phr силан → {p['ceramic_strength']:.1f} Н/м²")
                            
                    except Exception as e:
                        st.error(f"Ошибка расчета: {e}")
        else:
            if st.button("Рассчитать свойства", use_container_width=True):
                try:
                    if filler_type not in interpolators:
                        st.error(f"Нет данных для наполнителя {filler_type}")
                    else:
                        points = interpolators[filler_type]['points']
                        same_silane = [p for p in points if p['silane'] == silane_content]
                        
                        if len(same_silane) >= 2:
                            x = [p['content'] for p in same_silane]
                            y = [p['ceramic_strength'] for p in same_silane]
                            interp = interp1d(x, y, kind='linear', fill_value='extrapolate')
                            pred = float(interp(filler_content))
                            st.success("✅ Расчет выполнен (интерполяция)")
                        else:
                            x = [p['content'] for p in points]
                            y = [p['ceramic_strength'] for p in points]
                            interp = interp1d(x, y, kind='linear', fill_value='extrapolate')
                            pred = float(interp(filler_content))
                            st.success("✅ Расчет выполнен (экстраполяция)")
                        
                        st.metric("Прочность керамического остатка", f"{pred:.1f} Н/м²")
                        
                        st.caption("Использованные экспериментальные точки:")
                        for p in points:
                            st.caption(f"  {p['content']} phr + {p['silane']} phr силан → {p['ceramic_strength']:.1f} Н/м²")
                        
                except Exception as e:
                    st.error(f"Ошибка расчета: {e}")

with tab2:
    st.markdown('<p class="section-header">Экспериментальные данные</p>', unsafe_allow_html=True)
    st.dataframe(data_df, use_container_width=True)

st.markdown("""
<div class="footer">
    Система прогнозирования свойств эластомерных материалов &bull; Алгоритм: интерполяция
</div>
""", unsafe_allow_html=True)
