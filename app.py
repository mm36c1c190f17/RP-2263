import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
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
            <div style="color: rgba(255,255,255,0.9); font-size: 0.9rem;">Система прогнозирования на основе экспериментальных данных</div>
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
        filler_content = st.number_input("Дозировка (phr)", min_value=10, max_value=50, value=30, step=1)
        silane_content = st.number_input("Силан (phr)", min_value=0, max_value=10, value=0, step=1)
    
    with col2:
        st.info(f"**Каучук:** VMQ (Xiameter, 70)")
        st.info(f"**Наполнитель:** {filler_type} ({filler_content} phr)")
        st.info(f"**Силан:** {silane_content} phr")
        
        if st.button("🚀 Рассчитать", use_container_width=True):
            # Ищем точное совпадение
            result = data_df[
                (data_df['filler_type'] == filler_type) & 
                (data_df['filler_content'] == filler_content) & 
                (data_df['silane_content'] == silane_content)
            ]
            
            if len(result) > 0:
                st.success("✅ Найдено в базе данных")
                row = result.iloc[0]
                
                # Показываем все доступные свойства
                labels = {
                    'ceramic_strength': 'Прочность керамического остатка',
                    'strength_initial': 'Прочность начальная',
                    'elongation_initial': 'Удлинение начальное',
                    'strength_aged_240h_250C': 'Прочность 240ч/250°C',
                    'elongation_aged_240h_250C': 'Удлинение 240ч/250°C',
                    'strength_aged_72h_250C': 'Прочность 72ч/250°C',
                    'elongation_aged_72h_250C': 'Удлинение 72ч/250°C',
                    'resistivity': 'Удельное сопротивление',
                    'permittivity': 'Диэлектрическая проницаемость',
                    'tan_delta': 'Тангенс угла потерь',
                    'dielectric_strength': 'Диэлектрическая прочность'
                }
                
                units = {
                    'ceramic_strength': 'Н/м²',
                    'strength_initial': 'МПа',
                    'elongation_initial': '%',
                    'strength_aged_240h_250C': 'МПа',
                    'elongation_aged_240h_250C': '%',
                    'strength_aged_72h_250C': 'МПа',
                    'elongation_aged_72h_250C': '%',
                    'resistivity': 'Ом·м',
                    'permittivity': '',
                    'tan_delta': '',
                    'dielectric_strength': 'кВ/мм'
                }
                
                formats = {
                    'ceramic_strength': '{:.1f}',
                    'strength_initial': '{:.2f}',
                    'elongation_initial': '{:.0f}',
                    'strength_aged_240h_250C': '{:.2f}',
                    'elongation_aged_240h_250C': '{:.0f}',
                    'strength_aged_72h_250C': '{:.2f}',
                    'elongation_aged_72h_250C': '{:.0f}',
                    'resistivity': '{:.2e}',
                    'permittivity': '{:.3f}',
                    'tan_delta': '{:.4f}',
                    'dielectric_strength': '{:.1f}'
                }
                
                st.subheader("Результаты")
                
                # Сначала показываем керамическую прочность
                if pd.notna(row.get('ceramic_strength')):
                    st.metric(labels['ceramic_strength'], f"{row['ceramic_strength']:.1f} Н/м²")
                
                # Остальные свойства в 2 колонки
                cols = st.columns(2)
                other_props = [p for p in labels.keys() if p != 'ceramic_strength']
                for i, prop in enumerate(other_props):
                    if prop in row and pd.notna(row[prop]):
                        fmt = formats.get(prop, '{:.2f}')
                        unit = units.get(prop, '')
                        with cols[i % 2]:
                            st.metric(labels[prop], f"{fmt.format(row[prop])} {unit}")
            else:
                st.warning("⚠️ Данные для этого состава не найдены")
                
                # Показываем ближайшие составы
                st.caption("Ближайшие известные составы:")
                nearby = data_df[data_df['filler_type'] == filler_type]
                if len(nearby) > 0:
                    for _, row in nearby.iterrows():
                        st.caption(f"  {row['filler_content']} phr + {row['silane_content']} phr силан → {row['ceramic_strength']:.1f} Н/м²")

with tab2:
    st.dataframe(data_df, use_container_width=True)

st.markdown("""
<div class="footer">
    Версия 3.4
</div>
""", unsafe_allow_html=True)
