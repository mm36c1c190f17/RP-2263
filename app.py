import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from scipy.interpolate import interp1d
import base64

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
    .property-card {
        background: white;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.5rem;
        border-left: 3px solid #4361ee;
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
        interpolators = joblib.load('models/interpolators_all.pkl')
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
        
        if st.button("Рассчитать свойства", use_container_width=True):
            with st.spinner("Выполняется расчет..."):
                try:
                    if filler_type not in interpolators:
                        st.error(f"Нет данных для наполнителя {filler_type}")
                    else:
                        results = {}
                        filler_interp = interpolators[filler_type]
                        
                        for prop, interp_data in filler_interp.items():
                            points = interp_data['points']
                            
                            # Ищем точки с таким же силаном
                            same_silane = [p for p in points if p['silane'] == silane_content]
                            
                            if len(same_silane) >= 2:
                                x = [p['content'] for p in same_silane]
                                y = [p['value'] for p in same_silane]
                                interp = interp1d(x, y, kind='linear', fill_value='extrapolate')
                                results[prop] = float(interp(filler_content))
                            elif len(points) >= 2:
                                x = [p['content'] for p in points]
                                y = [p['value'] for p in points]
                                interp = interp1d(x, y, kind='linear', fill_value='extrapolate')
                                results[prop] = float(interp(filler_content))
                            else:
                                results[prop] = None
                        
                        st.success("✅ Расчет выполнен")
                        
                        # Отображение результатов
                        st.subheader("Результаты прогнозирования")
                        
                        # Группируем характеристики
                        props_mechanical = ['strength_initial', 'elongation_initial', 
                                           'strength_aged_240h_250C', 'elongation_aged_240h_250C',
                                           'strength_aged_72h_250C', 'elongation_aged_72h_250C']
                        props_electrical = ['resistivity', 'permittivity', 'tan_delta', 'dielectric_strength']
                        props_ceramic = ['ceramic_strength']
                        
                        labels = {
                            'strength_initial': 'Прочность начальная (МПа)',
                            'elongation_initial': 'Удлинение начальное (%)',
                            'strength_aged_240h_250C': 'Прочность 240ч/250°C (МПа)',
                            'elongation_aged_240h_250C': 'Удлинение 240ч/250°C (%)',
                            'strength_aged_72h_250C': 'Прочность 72ч/250°C (МПа)',
                            'elongation_aged_72h_250C': 'Удлинение 72ч/250°C (%)',
                            'resistivity': 'Удельное сопротивление (Ом·м)',
                            'permittivity': 'Диэлектрическая проницаемость',
                            'tan_delta': 'Тангенс угла диэлектрических потерь',
                            'dielectric_strength': 'Диэлектрическая прочность (кВ/мм)',
                            'ceramic_strength': 'Прочность керамического остатка (Н/м²)'
                        }
                        
                        formats = {
                            'strength_initial': '{:.2f}',
                            'elongation_initial': '{:.0f}',
                            'strength_aged_240h_250C': '{:.2f}',
                            'elongation_aged_240h_250C': '{:.0f}',
                            'strength_aged_72h_250C': '{:.2f}',
                            'elongation_aged_72h_250C': '{:.0f}',
                            'resistivity': '{:.2e}',
                            'permittivity': '{:.3f}',
                            'tan_delta': '{:.4f}',
                            'dielectric_strength': '{:.1f}',
                            'ceramic_strength': '{:.1f}'
                        }
                        
                        # Механические свойства
                        st.markdown("##### Механические свойства")
                        cols = st.columns(3)
                        for i, prop in enumerate(props_mechanical):
                            if prop in results and results[prop] is not None:
                                val = results[prop]
                                fmt = formats.get(prop, '{:.2f}')
                                with cols[i % 3]:
                                    st.metric(labels.get(prop, prop), fmt.format(val))
                        
                        # Электрические свойства
                        st.markdown("##### Электрические свойства")
                        cols = st.columns(3)
                        for i, prop in enumerate(props_electrical):
                            if prop in results and results[prop] is not None:
                                val = results[prop]
                                fmt = formats.get(prop, '{:.2f}')
                                with cols[i % 3]:
                                    st.metric(labels.get(prop, prop), fmt.format(val))
                        
                        # Керамическая прочность
                        st.markdown("##### Керамическая прочность")
                        if 'ceramic_strength' in results and results['ceramic_strength'] is not None:
                            st.metric(labels['ceramic_strength'], f"{results['ceramic_strength']:.1f} Н/м²")
                        
                        # Показываем использованные точки
                        with st.expander("Показать использованные экспериментальные точки"):
                            for prop, interp_data in filler_interp.items():
                                if interp_data['points']:
                                    st.caption(f"**{labels.get(prop, prop)}:**")
                                    for p in interp_data['points']:
                                        st.caption(f"  {p['content']} phr + {p['silane']} phr силан → {p['value']:.2f}")
                        
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
