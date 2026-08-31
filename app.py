import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(
    page_title="RubberAI - Прогнозирование свойств резин",
    page_icon="🧪",
    layout="wide"
)

st.title("🧪 RubberAI - Прогнозирование свойств резин")

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
    st.warning("⚠️ Таблица поиска не загружена")
    st.stop()

st.success(f"✅ Загружено {len(lookup_table)} записей")

# Интерфейс
st.sidebar.write(f"📈 Данные: {len(data_df)} записей")
st.sidebar.write(f"🏷️ Наполнители: {', '.join(data_df['filler_type'].unique())}")

tab1, tab2 = st.tabs(["🔮 Предсказание", "📋 Данные"])

with tab1:
    st.header("🔮 Предсказание свойств")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fillers = sorted(data_df['filler_type'].unique())
        filler_type = st.selectbox("Тип наполнителя", fillers)
        filler_content = st.number_input("Дозировка наполнителя (phr)", min_value=10, max_value=50, value=30, step=1)
        silane_content = st.number_input("Содержание силана (phr)", min_value=0, max_value=10, value=0, step=1)
    
    with col2:
        st.subheader("📊 Параметры состава")
        st.write(f"**Каучук:** VMQ (Xiameter)")
        st.write(f"**Наполнитель:** {filler_type}")
        st.write(f"**Дозировка:** {filler_content} phr")
        st.write(f"**Силан:** {silane_content} phr")
        
        if st.button("🚀 Найти свойства", type="primary"):
            key = (filler_type, filler_content, silane_content)
            
            if key in lookup_table:
                results = lookup_table[key]
                st.success("✅ Найдено в базе данных!")
                
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
                        st.metric("Прочность керамич. остатка", f"{results['ceramic_strength']:.1f} Н/м²")
                    if results.get('resistivity') is not None:
                        st.metric("Уд. сопротивление", f"{results['resistivity']:.2e} Ом·м")
            else:
                st.warning("⚠️ Данные для этого состава не найдены в базе")
                st.info("💡 Добавьте этот состав в файл данных и переобучите модель")

with tab2:
    st.header("📋 Все данные")
    st.dataframe(data_df, use_container_width=True)
    
    # Скачать данные
    csv = data_df.to_csv(index=False)
    st.download_button(
        label="📥 Скачать данные (CSV)",
        data=csv,
        file_name="experimental_data.csv",
        mime="text/csv"
    )
