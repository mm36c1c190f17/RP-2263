import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import plotly.express as px
import base64

st.set_page_config(
    page_title="RubberAI - Прогнозирование свойств резин",
    page_icon="🧪",
    layout="wide"
)

st.markdown("""
<style>
    .main { background-color: #ffffff; }
    h1 {
        font-weight: 400 !important;
        font-size: 1.8rem !important;
        color: #1a1a2e !important;
        border-bottom: 2px solid #e8e8e8;
        padding-bottom: 0.8rem;
        margin-bottom: 1.5rem !important;
    }
    .stButton > button {
        background-color: #2d3436 !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 3px !important;
        padding: 0.6rem 2rem !important;
        font-weight: 400 !important;
    }
    .stButton > button:hover {
        background-color: #1a1a2e !important;
    }
    [data-testid="metric-container"] {
        background-color: #f8f9fa;
        border: 1px solid #e8e8e8;
        border-radius: 3px;
        padding: 1rem 0.5rem;
        text-align: center;
    }
    [data-testid="metric-container"] label {
        font-weight: 400 !important;
        color: #555 !important;
        font-size: 0.8rem !important;
    }
    .footer {
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #e8e8e8;
        font-size: 0.75rem;
        color: #999;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

st.title("🧪 RubberAI - Прогнозирование свойств резин")
st.caption("Модель прогнозирования на основе состава и режима вулканизации")

@st.cache_resource
def load_models():
    try:
        models_dir = Path('models/')
        
        if not models_dir.exists():
            return None, None, None, None, f"Папка models/ не существует"
        
        encoders_files = list(models_dir.glob('encoders_*.pkl'))
        if not encoders_files:
            return None, None, None, None, f"Файлы encoders_*.pkl не найдены"
        
        latest = max(encoders_files, key=lambda x: x.stat().st_mtime)
        timestamp = latest.stem.replace('encoders_', '')
        
        encoders = joblib.load(f'models/encoders_{timestamp}.pkl')
        scaler = joblib.load(f'models/scaler_{timestamp}.pkl')
        
        models = {}
        for f in models_dir.glob(f'*_{timestamp}.pkl'):
            name = f.stem.replace(f'_{timestamp}', '')
            if name not in ['encoders', 'scaler', 'feature_info']:
                models[name] = joblib.load(f)
        
        # Пробуем загрузить данные
        try:
            df = pd.read_csv('data/raw/experimental_data.csv')
        except:
            # Если данных нет, создаем пустой DataFrame с нужными колонками
            st.warning("⚠️ Данные не найдены, используется минимальный набор")
            df = pd.DataFrame({
                'filler_type': ['шпинель', 'волластонит', 'Al2O3'],
                'filler_amount': [30, 30, 30],
                'P_strength': [38.3, 98.8, 96.6]
            })
        
        return models, encoders, scaler, df, None
    except Exception as e:
        import traceback
        return None, None, None, None, f"{str(e)}\n{traceback.format_exc()}"

models, encoders, scaler, data_df, error = load_models()

if error:
    st.error(f"❌ Ошибка загрузки моделей:\n{error}")
    st.stop()

if models is None:
    st.warning("⚠️ Модели не загружены")
    st.stop()

st.success(f"✅ Загружено {len(models)} моделей")

# Интерфейс
st.sidebar.title("📊 Управление")
if data_df is not None and len(data_df) > 0:
    st.sidebar.write(f"📈 Всего записей: {len(data_df)}")
    st.sidebar.write(f"🏷️ Наполнители: {', '.join(data_df['filler_type'].unique())}")

tab1, tab2, tab3 = st.tabs(["🔮 Предсказание", "📈 Визуализация", "📋 Данные"])

with tab1:
    st.header("🔮 Предсказание свойств")
    
    if data_df is not None and len(data_df) > 0:
        fillers = sorted(data_df['filler_type'].unique())
    else:
        fillers = ['шпинель', 'волластонит', 'Al2O3', 'SiO2', 'CaO']
    
    col1, col2 = st.columns(2)
    
    with col1:
        filler_type = st.selectbox("Тип наполнителя", fillers)
        amount = st.slider("Дозировка наполнителя (масс. частей)", 10, 100, 30, 5)
        silane = st.slider("Содержание силана (масс. частей)", 0, 10, 0, 1)
    
    with col2:
        st.subheader("📊 Параметры состава")
        st.write(f"**Наполнитель:** {filler_type}")
        st.write(f"**Дозировка:** {amount} phr")
        st.write(f"**Силан:** {silane} phr")
        
        if st.button("🚀 Предсказать свойства", type="primary"):
            with st.spinner("Выполняется предсказание..."):
                try:
                    data = {
                        'filler_type': filler_type,
                        'manufacturer': 'Дубна',
                        'rubber_type': 'СКТВ-1',
                        'silane_content': silane,
                        'filler_amount': amount
                    }
                    
                    X_dict = {}
                    for col, encoder in encoders.items():
                        if col in data:
                            try:
                                X_dict[f'{col}_encoded'] = encoder.transform([data[col]])[0]
                            except:
                                X_dict[f'{col}_encoded'] = encoder.transform([encoder.classes_[0]])[0]
                    
                    X_dict['silane_content'] = data['silane_content']
                    X_dict['filler_amount'] = data['filler_amount']
                    
                    X_df = pd.DataFrame([X_dict])
                    X_scaled = scaler.transform(X_df)
                    
                    results = {}
                    for name, model in models.items():
                        results[name] = float(model.predict(X_scaled)[0])
                    
                    st.success("✅ Предсказание выполнено!")
                    
                    col3, col4 = st.columns(2)
                    with col3:
                        if 'fp_before' in results:
                            st.metric("🔄 Прочность до старения", f"{results['fp_before']:.2f} МПа")
                        if 'ep_before' in results:
                            st.metric("📏 Удлинение до старения", f"{results['ep_before']:.0f} %")
                        if 'fp_after' in results:
                            st.metric("🔥 Прочность после старения", f"{results['fp_after']:.2f} МПа")
                    with col4:
                        if 'ep_after' in results:
                            st.metric("📐 Удлинение после старения", f"{results['ep_after']:.0f} %")
                        if 'rho_before' in results:
                            st.metric("⚡ Уд. сопротивление до воды", f"{results['rho_before']:.2e} Ом·м")
                        if 'P_strength' in results:
                            st.metric("🏺 Прочность керамического остатка", f"{results['P_strength']:.1f} Н/м²")
                        
                except Exception as e:
                    st.error(f"Ошибка: {e}")

with tab2:
    st.header("📈 Визуализация данных")
    if data_df is not None and len(data_df) > 0:
        try:
            param = st.selectbox("Выберите параметр", 
                                ['P_strength', 'fp_before', 'ep_before', 'fp_after', 'ep_after'])
            fig = px.scatter(data_df, x='filler_amount', y=param, color='filler_type',
                            title=f'Зависимость {param} от дозировки')
            st.plotly_chart(fig, use_container_width=True)
        except:
            st.info("Нет данных для визуализации")
    else:
        st.info("Нет данных для визуализации")

with tab3:
    st.header("📋 Данные")
    if data_df is not None and len(data_df) > 0:
        st.dataframe(data_df, use_container_width=True)
    else:
        st.info("Нет данных для отображения")

st.markdown("""
<div class="footer">
    <span>🧪 RubberAI v2.0 &bull; Алгоритм: KNN &bull; 
    <a href="#" style="color: #999; text-decoration: none;">Документация</a></span>
</div>
""", unsafe_allow_html=True)
