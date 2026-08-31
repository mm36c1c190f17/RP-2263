import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from io import BytesIO
import base64
import os

# ============================================================
#  НАСТРОЙКА СТРАНИЦЫ
# ============================================================
st.set_page_config(
    page_title="RubberAI - Прогнозирование свойств резин",
    page_icon="🧪",
    layout="wide"
)

# === СТИЛИ ===
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
    .stAlert {
        background-color: #f8f9fa !important;
        border-left: 3px solid #2d3436 !important;
        color: #2d3436 !important;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
#  ЗАГРУЗКА МОДЕЛЕЙ
# ============================================================
@st.cache_resource
def load_models():
    """Загрузка KNN моделей"""
    try:
        models_dir = Path('models/')
        encoders_files = list(models_dir.glob('encoders_*.pkl'))
        if not encoders_files:
            return None, None, None, None
        
        # Берем самую свежую модель
        latest = max(encoders_files, key=lambda x: x.stat().st_mtime)
        timestamp = latest.stem.replace('encoders_', '')
        
        encoders = joblib.load(f'models/encoders_{timestamp}.pkl')
        scaler = joblib.load(f'models/scaler_{timestamp}.pkl')
        
        models = {}
        for f in models_dir.glob(f'*_{timestamp}.pkl'):
            name = f.stem.replace(f'_{timestamp}', '')
            if name not in ['encoders', 'scaler', 'feature_info']:
                models[name] = joblib.load(f)
        
        # Загружаем данные
        df = pd.read_csv('data/raw/experimental_data.csv')
        
        return models, encoders, scaler, df
    except Exception as e:
        return None, None, None, None

# ============================================================
#  ЗАГОЛОВОК
# ============================================================
st.title("🧪 RubberAI - Прогнозирование свойств резин")
st.caption("Модель прогнозирования на основе состава и режима вулканизации")

# Загружаем модели
models, encoders, scaler, data_df = load_models()

if models is None:
    st.warning("⚠️ Модели не загружены. Проверьте наличие моделей в папке models/")
    st.info("Для обучения модели запустите локально: python train_knn.py")
    st.stop()

# ============================================================
#  ИНТЕРФЕЙС
# ============================================================
st.sidebar.title("📊 Управление")
st.sidebar.write(f"📈 Всего записей: {len(data_df)}")
st.sidebar.write(f"🏷️ Наполнители: {', '.join(data_df['filler_type'].unique())}")

# Основные вкладки
tab1, tab2, tab3, tab4 = st.tabs(["🔮 Предсказание", "📊 Загрузка данных", "📈 Визуализация", "📋 Данные"])

# ============================================================
#  ВКЛАДКА 1: ПРЕДСКАЗАНИЕ
# ============================================================
with tab1:
    st.header("🔮 Предсказание свойств")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fillers = sorted(data_df['filler_type'].unique())
        filler_type = st.selectbox("Тип наполнителя", fillers)
        amount = st.slider("Дозировка наполнителя (масс. частей)", 
                          min_value=10, max_value=100, value=30, step=5)
        silane = st.slider("Содержание силана (масс. частей)", 
                          min_value=0, max_value=10, value=0, step=1)
    
    with col2:
        st.subheader("📊 Параметры состава")
        st.write(f"**Наполнитель:** {filler_type}")
        st.write(f"**Дозировка:** {amount} phr")
        st.write(f"**Силан:** {silane} phr")
        
        if st.button("🚀 Предсказать свойства", type="primary"):
            with st.spinner("Выполняется предсказание..."):
                try:
                    # Подготовка данных
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
                    
                    # Предсказания
                    results = {}
                    for name, model in models.items():
                        results[name] = model.predict(X_scaled)[0]
                    
                    st.success("✅ Предсказание выполнено!")
                    
                    # Отображение результатов
                    col3, col4 = st.columns(2)
                    
                    with col3:
                        st.metric("🔄 Прочность до старения", f"{results['fp_before']:.2f} МПа")
                        st.metric("📏 Удлинение до старения", f"{results['ep_before']:.0f} %")
                        st.metric("🔥 Прочность после старения", f"{results['fp_after']:.2f} МПа")
                    
                    with col4:
                        st.metric("📐 Удлинение после старения", f"{results['ep_after']:.0f} %")
                        st.metric("⚡ Уд. сопротивление до воды", f"{results['rho_before']:.2e} Ом·м")
                        st.metric("⚡ Уд. сопротивление после воды", f"{results['rho_after']:.2e} Ом·м")
                        st.metric("🏺 Прочность керамического остатка", f"{results['P_strength']:.1f} Н/м²")
                        
                except Exception as e:
                    st.error(f"Ошибка: {e}")

# ============================================================
#  ВКЛАДКА 2: ЗАГРУЗКА ДАННЫХ
# ============================================================
with tab2:
    st.header("📊 Загрузка новых данных")
    
    st.info("""
    📌 **Формат Excel файла:**
    - Колонки: filler_type, filler_amount, silane_content, 
      fp_before, ep_before, fp_after, ep_after, rho_before, rho_after, P_strength
    - Первая строка - заголовки
    - Поддерживаются .xlsx и .xls файлы
    """)
    
    uploaded_file = st.file_uploader("Выберите Excel файл с данными", 
                                     type=['xlsx', 'xls'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_excel(uploaded_file)
            st.subheader("📋 Предпросмотр данных")
            st.dataframe(df.head())
            
            required_cols = ['filler_type', 'filler_amount', 'P_strength']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                st.error(f"❌ Отсутствуют колонки: {missing_cols}")
            else:
                st.success("✅ Данные успешно загружены!")
                
                if st.button("💾 Сохранить и переобучить модель"):
                    try:
                        # Сохраняем данные
                        existing = pd.read_csv('data/raw/experimental_data.csv')
                        combined = pd.concat([existing, df], ignore_index=True)
                        combined.to_csv('data/raw/experimental_data.csv', index=False)
                        
                        st.success("✅ Данные добавлены!")
                        st.info("🔄 Переобучение модели...")
                        
                        # Переобучаем модель (KNN)
                        import subprocess
                        result = subprocess.run(['python', 'train_knn.py'], 
                                               capture_output=True, text=True)
                        st.code(result.stdout)
                        st.success("✅ Модель переобучена!")
                        
                    except Exception as e:
                        st.error(f"Ошибка: {e}")
                        
        except Exception as e:
            st.error(f"Ошибка чтения файла: {e}")

# ============================================================
#  ВКЛАДКА 3: ВИЗУАЛИЗАЦИЯ
# ============================================================
with tab3:
    st.header("📈 Визуализация данных")
    
    if data_df is not None:
        param = st.selectbox("Выберите параметр", 
                            ['P_strength', 'fp_before', 'ep_before', 'fp_after', 'ep_after'])
        
        fig = px.scatter(data_df, x='filler_amount', y=param, 
                         color='filler_type', 
                         title=f'Зависимость {param} от дозировки',
                         labels={'filler_amount': 'Дозировка (phr)', param: param})
        st.plotly_chart(fig, use_container_width=True)
        
        fig2 = px.box(data_df, x='filler_type', y=param,
                      title=f'Распределение {param} по наполнителям')
        st.plotly_chart(fig2, use_container_width=True)

# ============================================================
#  ВКЛАДКА 4: ДАННЫЕ
# ============================================================
with tab4:
    st.header("📋 Все данные")
    st.dataframe(data_df, use_container_width=True)
    
    st.subheader("📊 Статистика")
    st.dataframe(data_df.describe(), use_container_width=True)
    
    csv = data_df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="all_data.csv">📥 Скачать все данные (CSV)</a>'
    st.markdown(href, unsafe_allow_html=True)

# ============================================================
#  ПОДВАЛ
# ============================================================
st.markdown("""
<div class="footer">
    <span>🧪 RubberAI v2.0 &bull; Алгоритм: KNN &bull; 
    <a href="#" style="color: #999; text-decoration: none;">Документация</a></span>
</div>
""", unsafe_allow_html=True)
