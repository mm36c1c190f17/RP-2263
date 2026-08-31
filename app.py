import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(
    page_title="RubberAI - Прогнозирование свойств резин",
    page_icon="⚙️",
    layout="wide"
)

st.markdown("""
<style>
    .main { background-color: #f5f5f5; }
    h1 { font-weight: 300; font-size: 2rem; color: #2d2d2d; border-bottom: 2px solid #d0d0d0; padding-bottom: 0.5rem; }
    .stButton > button { background-color: #2d2d2d; color: #ffffff; border: none; border-radius: 2px; padding: 0.6rem 2rem; font-weight: 400; width: 100%; }
    .stButton > button:hover { background-color: #1a1a1a; }
    [data-testid="metric-container"] { background-color: #ffffff; border: 1px solid #e0e0e0; border-radius: 2px; padding: 1rem 0.5rem; }
    [data-testid="metric-container"] label { font-weight: 400; color: #555555; font-size: 0.8rem; }
    .footer { margin-top: 3rem; padding-top: 1rem; border-top: 1px solid #e0e0e0; font-size: 0.75rem; color: #999999; text-align: center; }
    .section-header { color: #2d2d2d; font-weight: 400; font-size: 1.2rem; border-bottom: 1px solid #e0e0e0; padding-bottom: 0.3rem; margin-bottom: 1rem; }
</style>
""", unsafe_allow_html=True)

st.title("RubberAI - Прогнозирование свойств резин")

@st.cache_resource
def load_models():
    try:
        lookup_table = joblib.load('models/lookup_table.pkl')
        df = pd.read_csv('data/raw/experimental_data.csv')
        return lookup_table, df
    except Exception as e:
        st.error(f"Ошибка загрузки: {e}")
        return None, None

lookup_table, data_df = load_models()

if lookup_table is None:
    st.warning("Таблица поиска не загружена")
    st.stop()

st.sidebar.markdown("### Управление")
st.sidebar.write(f"**Записей:** {len(data_df)}")
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
        st.markdown("### Состав")
        st.write(f"**Каучук:** VMQ (Xiameter)")
        st.write(f"**Наполнитель:** {filler_type}")
        st.write(f"**Дозировка:** {filler_content} phr")
        st.write(f"**Силан:** {silane_content} phr")
        
        if st.button("Найти свойства"):
            key = (filler_type, filler_content, silane_content)
            
            if key in lookup_table:
                results = lookup_table[key]
                st.success("Данные найдены")
                
                col3, col4 = st.columns(2)
                with col3:
                    if results.get('strength_initial') is not None:
                        st.metric("Прочность начальная", f"{results['strength_initial']:.2f} МПа")
                    if results.get('elongation_initial') is not None:
                        st.metric("Удлинение начальное", f"{results['elongation_initial']:.0f} %")
                    if results.get('strength_aged_240h_250C') is not None:
                        st.metric("Прочность 240ч/250°C", f"{results['strength_aged_240h_250C']:.2f} МПа")
                with col4:
                    if results.get('elongation_aged_240h_250C') is not None:
                        st.metric("Удлинение 240ч/250°C", f"{results['elongation_aged_240h_250C']:.0f} %")
                    if results.get('ceramic_strength') is not None:
                        st.metric("Прочность керамического остатка", f"{results['ceramic_strength']:.1f} Н/м²")
                    if results.get('resistivity') is not None:
                        st.metric("Удельное сопротивление", f"{results['resistivity']:.2e} Ом·м")
            else:
                st.warning("Данные для этого состава не найдены в базе")

with tab2:
    st.markdown('<p class="section-header">Данные</p>', unsafe_allow_html=True)
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
    RubberAI v2.0 &bull; Алгоритм: таблица поиска
</div>
""", unsafe_allow_html=True)
