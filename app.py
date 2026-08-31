import streamlit as st
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import base64
import matplotlib.pyplot as plt
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
    .stApp { background: linear-gradient(135deg, #e0e7ff 0%, #f0e6ff 50%, #fce4ec 100%); }
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
        font-size: 0.9rem;
        font-weight: 300;
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
        font-size: 1.3rem;
        width: 100%;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        letter-spacing: 0.5px;
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
        border: 1px solid rgba(255, 255, 255, 0.3);
        box-shadow: 0 4px 20px rgba(0,0,0,0.06);
        margin-bottom: 0.5rem;
        animation: fadeInUp 0.8s ease forwards;
        opacity: 0;
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
    
    .footer { 
        margin-top: 3rem; 
        padding-top: 1.5rem; 
        border-top: 2px solid rgba(102, 126, 234, 0.3); 
        text-align: center;
        font-size: 0.8rem;
        color: #4a5568;
    }
    
    /* Вкладки - чуть уже, буквы крупнее */
    .stTabs {
        background: rgba(255, 255, 255, 0.3);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 8px 12px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.05);
        border: 1px solid rgba(255, 255, 255, 0.5);
        margin-bottom: 1rem;
        width: 100%;
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
        padding: 12px 36px !important;
        font-weight: 600;
        font-size: 1.15rem;
        color: #4a5568;
        transition: all 0.3s ease;
        background: transparent;
        border: none;
        margin: 0;
        min-width: 0;
        flex: 1 1 auto;
        text-align: center;
        white-space: nowrap;
        letter-spacing: 0.3px;
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
        padding: 12px 36px !important;
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
    
    .stSelectbox label, .stNumberInput label {
        font-weight: 500;
        color: #2d3748;
    }
    
    .stInfo {
        background: rgba(255, 255, 255, 0.5) !important;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.3);
        border-radius: 12px;
    }
    
    /* Анимация появления */
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
    
    .fade-in-delay-1 { animation-delay: 0.2s; }
    .fade-in-delay-2 { animation-delay: 0.4s; }
    .fade-in-delay-3 { animation-delay: 0.6s; }
    .fade-in-delay-4 { animation-delay: 0.8s; }
    
    /* Анимация для графика */
    .chart-container {
        animation: fadeInUp 0.8s ease forwards;
        opacity: 0;
        animation-delay: 0.5s;
    }
    
    /* Стиль для спиннера */
    .stSpinner > div {
        border-color: #667eea !important;
        border-width: 3px !important;
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
        df = pd.read_csv('data/raw/experimental_data.csv')
        return df
    except Exception as e:
        st.error(f"Ошибка загрузки данных: {e}")
        return None

data_df = load_data()

if data_df is None:
    st.error("Не удалось загрузить данные")
    st.stop()

st.sidebar.write(f"Записей в базе: {len(data_df)}")
st.sidebar.write(f"Наполнителей: {len(data_df['filler_type'].unique())}")

tab1, tab2 = st.tabs(["Прогнозирование", "Данные"])

with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        fillers = sorted(data_df['filler_type'].unique())
        filler_type = st.selectbox("Тип наполнителя", fillers)
        filler_content = st.number_input("Дозировка (phr)", min_value=10, max_value=50, value=35, step=1)
        silane_content = st.number_input("Силан (phr)", min_value=0, max_value=10, value=0, step=1)
    
    with col2:
        st.info(f"**Каучук:** VMQ (Xiameter, 70)")
        st.info(f"**Наполнитель:** {filler_type} ({filler_content} phr)")
        st.info(f"**Силан:** {silane_content} phr")
        
        if st.button("Рассчитать", use_container_width=True):
            with st.spinner("Выполняется расчет..."):
                # Имитация загрузки для анимации
                time.sleep(0.8)
                
                filler_data = data_df[data_df['filler_type'] == filler_type]
                
                if len(filler_data) == 0:
                    st.error(f"Нет данных для {filler_type}")
                else:
                    same_silane = filler_data[filler_data['silane_content'] == silane_content]
                    
                    if len(same_silane) >= 2:
                        use_data = same_silane
                    else:
                        use_data = filler_data
                    
                    use_data = use_data.sort_values('filler_content')
                    
                    if len(use_data) >= 2:
                        x = use_data['filler_content'].values
                        y = use_data['ceramic_strength'].values
                        
                        try:
                            interp = interp1d(x, y, kind='linear', fill_value='extrapolate')
                            predicted_value = float(interp(filler_content))
                            
                            # Появляемся с анимацией
                            st.markdown('<div class="fade-in">', unsafe_allow_html=True)
                            st.success(f"Расчет выполнен (интерполяция между {x.min():.0f} и {x.max():.0f} phr)")
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                            # Метрика с анимацией
                            st.markdown('<div class="fade-in fade-in-delay-1">', unsafe_allow_html=True)
                            st.metric("Прочность керамического остатка", f"{predicted_value:.1f} Н/м²")
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                            # Точки с анимацией
                            st.markdown('<div class="fade-in fade-in-delay-2">', unsafe_allow_html=True)
                            st.caption("Использованные экспериментальные точки:")
                            for i in range(len(use_data)):
                                row = use_data.iloc[i]
                                st.caption(f"  {row['filler_content']:.0f} phr + {row['silane_content']:.0f} phr силан → {row['ceramic_strength']:.1f} Н/м²")
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                            # График с анимацией
                            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                            fig, ax = plt.subplots(figsize=(8, 4))
                            ax.plot(x, y, 'o-', color='#667eea', linewidth=2, markersize=8)
                            ax.axvline(x=filler_content, color='#e53e3e', linestyle='--', linewidth=2, label=f'Ваша дозировка: {filler_content} phr')
                            ax.axhline(y=predicted_value, color='#38a169', linestyle='--', linewidth=2, alpha=0.7, label=f'Предсказание: {predicted_value:.1f}')
                            ax.set_xlabel('Дозировка наполнителя (phr)', fontsize=10)
                            ax.set_ylabel('Прочность керамического остатка (Н/м²)', fontsize=10)
                            ax.set_title(f'Зависимость для {filler_type}', fontsize=12)
                            ax.grid(True, alpha=0.3)
                            ax.legend(loc='best')
                            ax.set_facecolor('#f8f9fa')
                            fig.patch.set_facecolor('white')
                            st.pyplot(fig)
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                        except Exception as e:
                            st.error(f"Ошибка интерполяции: {e}")
                    else:
                        st.info(f"Только одна точка данных для {filler_type} при {silane_content} phr силана")
                        st.metric("Прочность керамического остатка", f"{use_data.iloc[0]['ceramic_strength']:.1f} Н/м²")

with tab2:
    st.dataframe(data_df, use_container_width=True)

st.markdown("""
<div class="footer">
    Версия 3.6
</div>
""", unsafe_allow_html=True)
