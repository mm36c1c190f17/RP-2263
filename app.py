import streamlit as st
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import base64

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
    }
    .header-title { font-weight: 600; font-size: 1.8rem; color: #ffffff; margin: 0; flex: 1; }
    .header-logo { height: 55px; width: auto; margin-left: 20px; }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: #ffffff;
        border: none;
        border-radius: 12px;
        padding: 0.8rem 2rem;
        font-weight: 600;
        width: 100%;
    }
    .stButton > button:hover { transform: translateY(-2px); }
    [data-testid="metric-container"] {
        background: rgba(255, 255, 255, 0.85);
        border-radius: 16px;
        padding: 1.5rem 1rem;
        border: 1px solid rgba(255, 255, 255, 0.3);
        margin-bottom: 0.5rem;
    }
    .footer { margin-top: 3rem; padding-top: 1.5rem; border-top: 2px solid rgba(102, 126, 234, 0.3); text-align: center; }
    .stTabs {
        background: rgba(255, 255, 255, 0.3);
        border-radius: 16px;
        padding: 8px 12px;
        border: 1px solid rgba(255, 255, 255, 0.5);
        margin-bottom: 1rem;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 12px;
        padding: 12px 48px !important;
        font-weight: 500;
        font-size: 1.05rem;
        color: #4a5568;
        text-align: center;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)

if logo_base64:
    st.markdown(f"""
    <div class="header-container">
        <div>
            <div class="header-title">Прогнозирование свойств эластомерных материалов</div>
            <div style="color: rgba(255,255,255,0.9); font-size: 0.9rem;">Система прогнозирования на основе интерполяции экспериментальных данных</div>
        </div>
        <img src="data:image/png;base64,{logo_base64}" class="header-logo" />
    </div>
    """, unsafe_allow_html=True)

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

st.sidebar.write(f"📊 Записей: {len(data_df)}")
st.sidebar.write(f"🏷️ Наполнителей: {len(data_df['filler_type'].unique())}")

tab1, tab2 = st.tabs(["🔮 Прогнозирование", "📊 Данные"])

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
        
        if st.button("🚀 Рассчитать", use_container_width=True):
            # Получаем данные для этого наполнителя
            filler_data = data_df[data_df['filler_type'] == filler_type]
            
            if len(filler_data) == 0:
                st.error(f"Нет данных для {filler_type}")
            else:
                # Группируем по силану
                same_silane = filler_data[filler_data['silane_content'] == silane_content]
                
                # Если есть точки с таким же силаном - используем их
                if len(same_silane) >= 2:
                    use_data = same_silane
                else:
                    use_data = filler_data
                
                # Сортируем по дозировке
                use_data = use_data.sort_values('filler_content')
                
                if len(use_data) >= 2:
                    # Интерполяция
                    x = use_data['filler_content'].values
                    y = use_data['ceramic_strength'].values
                    
                    try:
                        interp = interp1d(x, y, kind='linear', fill_value='extrapolate')
                        predicted_value = float(interp(filler_content))
                        
                        st.success(f"✅ Расчет выполнен (интерполяция между {x.min():.0f} и {x.max():.0f} phr)")
                        st.metric("Прочность керамического остатка", f"{predicted_value:.1f} Н/м²")
                        
                        # Показываем использованные точки
                        st.caption("Использованные экспериментальные точки:")
                        for i in range(len(use_data)):
                            row = use_data.iloc[i]
                            st.caption(f"  {row['filler_content']:.0f} phr + {row['silane_content']:.0f} phr силан → {row['ceramic_strength']:.1f} Н/м²")
                        
                        # Показываем график зависимости
                        st.caption("Зависимость прочности от дозировки:")
                        import matplotlib.pyplot as plt
                        fig, ax = plt.subplots(figsize=(8, 4))
                        ax.plot(x, y, 'o-', color='#667eea', linewidth=2, markersize=8)
                        ax.axvline(x=filler_content, color='red', linestyle='--', label=f'Ваша дозировка: {filler_content} phr')
                        ax.axhline(y=predicted_value, color='green', linestyle='--', alpha=0.5, label=f'Предсказание: {predicted_value:.1f}')
                        ax.set_xlabel('Дозировка наполнителя (phr)')
                        ax.set_ylabel('Прочность керамического остатка (Н/м²)')
                        ax.set_title(f'Зависимость для {filler_type}')
                        ax.grid(True, alpha=0.3)
                        ax.legend()
                        st.pyplot(fig)
                        
                    except Exception as e:
                        st.error(f"Ошибка интерполяции: {e}")
                else:
                    # Только одна точка - показываем ее
                    st.info(f"Только одна точка данных для {filler_type} при {silane_content} phr силана")
                    st.metric("Прочность керамического остатка", f"{use_data.iloc[0]['ceramic_strength']:.1f} Н/м²")

with tab2:
    st.dataframe(data_df, use_container_width=True)

st.markdown("""
<div class="footer">
    Версия 3.5 - Интерполяция между экспериментальными точками
</div>
""", unsafe_allow_html=True)
