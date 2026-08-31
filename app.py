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
def load_models():
    try:
        # Загружаем таблицу поиска
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
            key = (filler_type, filler_content, silane_content)
            
            if key in lookup_table:
                results = lookup_table[key]
                st.success("✅ Данные найдены в базе")
                
                if results.get('ceramic_strength') is not None:
                    st.metric("Прочность керамического остатка", f"{results['ceramic_strength']:.1f} Н/м²")
                    st.caption("Значение из экспериментальных данных")
            else:
                st.warning("⚠️ Данные для этого состава не найдены в базе")
                st.info("💡 Добавьте этот состав в файл данных для точного предсказания")
                
                # Показываем ближайшие известные составы
                st.markdown("---")
                st.caption("Ближайшие известные составы:")
                nearby = data_df[data_df['filler_type'] == filler_type]
                if len(nearby) > 0:
                    for _, row in nearby.iterrows():
                        st.caption(f"  {row['filler_content']} phr + {row['silane_content']} phr силан → {row['ceramic_strength']:.1f} Н/м²")

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
    Система прогнозирования свойств эластомерных материалов &bull; Версия 2.5 &bull; Таблица поиска
</div>
""", unsafe_allow_html=True)
