import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import plotly.express as px

st.set_page_config(
    page_title="RubberAI - Прогнозирование свойств резин",
    page_icon="🧪",
    layout="wide"
)

st.title("🧪 RubberAI - Прогнозирование свойств резин")

@st.cache_resource
def load_models():
    try:
        models_dir = Path('models/')
        encoders_files = list(models_dir.glob('encoders_*.pkl'))
        if not encoders_files:
            return None, None, None, None
        
        latest = max(encoders_files, key=lambda x: x.stat().st_mtime)
        timestamp = latest.stem.replace('encoders_', '')
        
        encoders = joblib.load(f'models/encoders_{timestamp}.pkl')
        scaler = joblib.load(f'models/scaler_{timestamp}.pkl')
        
        models = {}
        for f in models_dir.glob(f'*_{timestamp}.pkl'):
            name = f.stem.replace(f'_{timestamp}', '')
            if name not in ['encoders', 'scaler', 'feature_info']:
                models[name] = joblib.load(f)
        
        df = pd.read_csv('data/raw/experimental_data.csv')
        return models, encoders, scaler, df
    except Exception as e:
        st.error(f"Ошибка: {e}")
        return None, None, None, None

models, encoders, scaler, data_df = load_models()

if models is None:
    st.warning("Модели не загружены")
    st.stop()

st.success(f"✅ Загружено {len(models)} моделей на основе {len(data_df)} записей")

st.sidebar.write(f"📈 Данные: {len(data_df)} записей")
st.sidebar.write(f"🏷️ Наполнители: {', '.join(data_df['filler_type'].unique())}")
st.sidebar.write(f"🏭 Производитель каучука: {data_df['base_manufacturer'].unique()[0]}")
st.sidebar.write(f"🧪 Каучук: {data_df['base_type'].unique()[0]}")

tab1, tab2 = st.tabs(["🔮 Предсказание", "📋 Данные"])

with tab1:
    st.header("🔮 Предсказание свойств")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fillers = sorted(data_df['filler_type'].unique())
        filler_type = st.selectbox("Тип наполнителя", fillers)
        filler_content = st.slider("Дозировка наполнителя (phr)", 10, 100, 30, 5)
        temp = st.slider("Температура (°C)", 100, 200, 115, 5)
        time = st.slider("Время (мин)", 10, 60, 15, 5)
    
    with col2:
        st.subheader("📊 Параметры состава")
        st.write(f"**Каучук:** VMQ (Xiameter)")
        st.write(f"**Наполнитель:** {filler_type}")
        st.write(f"**Дозировка:** {filler_content} phr")
        st.write(f"**Температура:** {temp}°C")
        st.write(f"**Время:** {time} мин")
        
        if st.button("🚀 Предсказать свойства", type="primary"):
            with st.spinner("Выполняется предсказание..."):
                try:
                    data = {
                        'base_type': 'VMQ',
                        'base_hardness': 70,
                        'base_manufacturer': 'Xiameter',
                        'filler_type': filler_type,
                        'filler_content': filler_content,
                        'filler_manufacturer': 'OOO_NPO_EkoTek',
                        'temp': temp,
                        'time': time
                    }
                    
                    X_dict = {}
                    for col, encoder in encoders.items():
                        if col in data:
                            try:
                                X_dict[f'{col}_encoded'] = encoder.transform([data[col]])[0]
                            except:
                                X_dict[f'{col}_encoded'] = encoder.transform([encoder.classes_[0]])[0]
                    
                    X_dict['base_hardness'] = data['base_hardness']
                    X_dict['filler_content'] = data['filler_content']
                    X_dict['temp'] = data['temp']
                    X_dict['time'] = data['time']
                    
                    X_df = pd.DataFrame([X_dict])
                    X_scaled = scaler.transform(X_df)
                    
                    results = {}
                    for name, model in models.items():
                        results[name] = float(model.predict(X_scaled)[0])
                    
                    st.success("✅ Предсказание выполнено!")
                    
                    col3, col4 = st.columns(2)
                    with col3:
                        st.metric("Прочность начальная", f"{results['strength_initial']:.2f} МПа")
                        st.metric("Удлинение начальное", f"{results['elongation_initial']:.0f} %")
                        st.metric("Прочность 240ч/250°C", f"{results['strength_aged_240h_250C']:.2f} МПа")
                        st.metric("Удлинение 240ч/250°C", f"{results['elongation_aged_240h_250C']:.0f} %")
                    with col4:
                        st.metric("Прочность 72ч/250°C", f"{results['strength_aged_72h_250C']:.2f} МПа")
                        st.metric("Удлинение 72ч/250°C", f"{results['elongation_aged_72h_250C']:.0f} %")
                        st.metric("Уд. сопротивление", f"{results['resistivity']:.2e} Ом·м")
                        st.metric("Прочность керамич. остатка", f"{results['ceramic_strength']:.1f} Н/м²")
                        
                except Exception as e:
                    st.error(f"Ошибка: {e}")

with tab2:
    st.header("📋 Данные")
    st.dataframe(data_df, use_container_width=True)
