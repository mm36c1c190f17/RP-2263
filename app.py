import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from scipy.interpolate import interp1d
import base64
import time

st.set_page_config(
    page_title="Прогнозирование свойств эластомерных материалов",
    page_icon="⚙️",
    layout="wide"
)

def get_image_base64(image_path):
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except:
        return None

logo_base64 = get_image_base64("static/logo.png")

st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #e0e7ff 0%, #f0e6ff 50%, #fce4ec 100%);
        background-attachment: fixed;
    }
    .main { background: transparent; }
    .header-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.2rem 2rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
    }
    .header-title {
        font-weight: 600;
        font-size: 1.8rem;
        color: #ffffff;
        margin: 0;
        flex: 1;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .header-subtitle {
        color: rgba(255,255,255,0.9);
        font-weight: 300;
        font-size: 0.9rem;
    }
    .header-logo {
        height: 55px;
        width: auto;
        margin-left: 20px;
        filter: drop-shadow(0 2px 8px rgba(0,0,0,0.2));
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: #ffffff;
        border: none;
        border-radius: 12px;
        padding: 0.8rem 2rem;
        font-weight: 600;
        font-size: 1.1rem;
        width: 100%;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6);
    }
    [data-testid="metric-container"] {
        background: rgba(255, 255, 255, 0.85);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 1.5rem 1rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.06);
        border: 1px solid rgba(255, 255, 255, 0.3);
    }
    [data-testid="metric-container"]:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 30px rgba(102, 126, 234, 0.15);
    }
    [data-testid="metric-container"] label {
        font-weight: 500;
        color: #4a5568;
        font-size: 0.85rem;
    }
    .section-header {
        color: #2d3748;
        font-weight: 600;
        font-size: 1.4rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        border-bottom: 3px solid #667eea;
        padding-bottom: 0.3rem;
        margin-bottom: 1.5rem;
    }
    .footer {
        margin-top: 3rem;
        padding-top: 1.5rem;
        border-top: 2px solid rgba(102, 126, 234, 0.3);
        font-size: 0.8rem;
        color: #4a5568;
        text-align: center;
        background: rgba(255, 255, 255, 0.5);
        backdrop-filter: blur(10px);
        border-radius: 12px;
        padding: 1.5rem;
    }
    .stTabs {
        background: rgba(255, 255, 255, 0.3);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 8px 12px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.05);
        border: 1px solid rgba(255, 255, 255, 0.5);
        margin-bottom: 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background: transparent;
        border-radius: 12px;
        padding: 4px;
        display: flex;
        flex-wrap: nowrap;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 12px;
        padding: 12px 48px !important;
        font-weight: 500;
        font-size: 1.05rem;
        color: #4a5568;
        transition: all 0.3s ease;
        background: transparent;
        border: none;
        margin: 0;
        min-width: 0;
        flex: 1 1 auto;
        text-align: center;
        white-space: nowrap;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(255, 255, 255, 0.5);
        color: #2d3748;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: #ffffff !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        border-radius: 12px;
    }
    .stTabs [data-baseweb="tab-highlight"] {
        display: none !important;
    }
    .stTabs [data-baseweb="tab-border"] {
        display: none !important;
    }
    .stTabs [role="tabpanel"] {
        padding: 0.5rem 0.5rem 0.5rem 0.5rem;
        background: rgba(255, 255, 255, 0.2);
        border-radius: 12px;
        margin-top: 0.5rem;
    }
    .stSpinner > div {
        border-color: #667eea !important;
    }
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    .fade-in {
        animation: fadeInUp 0.6s ease forwards;
    }
</style>
""", unsafe_allow_html=True)

if logo_base64:
    st.markdown(f"""
    <div class="header-container">
        <div>
            <div class="header-title">Прогнозирование свойств эластомерных материалов</div>
            <div class="header-subtitle">Система прогнозирования на основе интерполяции экспериментальных данных</div>
        </div>
        <img src="data:image/png;base64,{logo_base64}" class="header-logo" />
    </div>
    """, unsafe_allow_html=True)
else:
    st.title("Прогнозирование свойств эластомерных материалов")
    st.caption("Система прогнозирования на основе интерполяции экспериментальных данных")

@st.cache_resource
def load_data():
    try:
        interpolators = joblib.load('models/interpolators_all.pkl')
        df = pd.read_csv('data/raw/experimental_data.csv')
        return interpolators, df
    except Exception as e:
        return None, None

interpolators, data_df = load_data()

if interpolators is None:
    st.error("Не удалось загрузить данные. Проверьте файлы models/interpolators_all.pkl")
    st.stop()

st.sidebar.markdown("Управление")
st.sidebar.write(f"Записей в базе: {len(data_df)}")
st.sidebar.write(f"Наполнителей: {len(data_df['filler_type'].unique())}")
st.sidebar.write(f"Доступно интерполяторов: {len(interpolators)}")
st.sidebar.markdown("---")

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
                time.sleep(0.5)
                
                try:
                    # Проверяем наличие данных для наполнителя
                    if filler_type not in interpolators:
                        st.error(f"Нет данных для наполнителя {filler_type}")
                    else:
                        filler_data = interpolators[filler_type]
                        
                        if not filler_data:
                            st.warning(f"Для {filler_type} нет данных для интерполяции")
                        else:
                            results = {}
                            
                            # Для каждого свойства
                            for prop, prop_data in filler_data.items():
                                points = prop_data['points']
                                
                                if len(points) >= 2:
                                    # Фильтруем по силану
                                    same_silane = [p for p in points if p['silane'] == silane_content]
                                    use_points = same_silane if len(same_silane) >= 2 else points
                                    
                                    try:
                                        x = [p['content'] for p in use_points]
                                        y = [p['value'] for p in use_points]
                                        interp = interp1d(x, y, kind='linear', fill_value='extrapolate')
                                        results[prop] = float(interp(filler_content))
                                    except Exception as e:
                                        results[prop] = None
                                else:
                                    results[prop] = None
                            
                            # Проверяем есть ли результаты
                            has_results = False
                            for prop, val in results.items():
                                if val is not None:
                                    has_results = True
                                    break
                            
                            if not has_results:
                                st.warning(f"Недостаточно данных для расчета {filler_type} при {filler_content} phr")
                            else:
                                st.success("Расчет выполнен успешно")
                                
                                st.markdown('<div class="fade-in">', unsafe_allow_html=True)
                                st.subheader("Результаты прогнозирования")
                                
                                labels = {
                                    'strength_initial': 'Прочность начальная',
                                    'elongation_initial': 'Удлинение начальное',
                                    'strength_aged_240h_250C': 'Прочность 240ч/250°C',
                                    'elongation_aged_240h_250C': 'Удлинение 240ч/250°C',
                                    'strength_aged_72h_250C': 'Прочность 72ч/250°C',
                                    'elongation_aged_72h_250C': 'Удлинение 72ч/250°C',
                                    'resistivity': 'Удельное сопротивление',
                                    'permittivity': 'Диэлектрическая проницаемость',
                                    'tan_delta': 'Тангенс угла потерь',
                                    'dielectric_strength': 'Диэлектрическая прочность',
                                    'ceramic_strength': 'Прочность керамического остатка'
                                }
                                
                                units = {
                                    'strength_initial': 'МПа',
                                    'elongation_initial': '%',
                                    'strength_aged_240h_250C': 'МПа',
                                    'elongation_aged_240h_250C': '%',
                                    'strength_aged_72h_250C': 'МПа',
                                    'elongation_aged_72h_250C': '%',
                                    'resistivity': 'Ом·м',
                                    'permittivity': '',
                                    'tan_delta': '',
                                    'dielectric_strength': 'кВ/мм',
                                    'ceramic_strength': 'Н/м²'
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
                                
                                # Отображаем все результаты
                                for prop, val in results.items():
                                    if val is not None and prop in labels:
                                        fmt = formats.get(prop, '{:.2f}')
                                        unit = units.get(prop, '')
                                        st.metric(labels[prop], f"{fmt.format(val)} {unit}")
                                
                                st.markdown('</div>', unsafe_allow_html=True)
                                
                                with st.expander("Показать использованные экспериментальные точки"):
                                    for prop, prop_data in filler_data.items():
                                        if prop_data['points']:
                                            st.caption(f"**{labels.get(prop, prop)}:**")
                                            for p in prop_data['points']:
                                                st.caption(f"  {p['content']} phr + {p['silane']} phr силан → {p['value']:.2f}")
                        
                except Exception as e:
                    st.error(f"Ошибка расчета: {e}")
                    st.info("Проверьте правильность введенных данных")

with tab2:
    st.markdown('<p class="section-header">Экспериментальные данные</p>', unsafe_allow_html=True)
    st.dataframe(data_df, use_container_width=True)
    
    csv = data_df.to_csv(index=False)
    st.download_button(
        label="Скачать данные (CSV)",
        data=csv,
        file_name="experimental_data.csv",
        mime="text/csv"
    )

st.markdown("""
<div class="footer">
    Система прогнозирования свойств эластомерных материалов &bull; Версия 3.1
</div>
""", unsafe_allow_html=True)
