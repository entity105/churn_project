import streamlit as st
import pandas as pd
import joblib
import os

# Загрузка модели и scaler
# model = joblib.load('models/churn_model.pkl')
scaler = joblib.load('models/scaler.pkl')
metrics = joblib.load('models/Метрики качества.pkl')
models = {}

d = {
        'logistic_regression_model.pkl' : 'Логистическая регрессия',
        'decision_tree_model.pkl' : 'Дерево решений',
        'knn_model.pkl' : 'KNN',
        'random_forest_model.pkl' : 'Случайный лес',
        'svm_model.pkl' : 'SVM'
    }

def get_name_model(file_model):
    return d[file_model]

def get_model(name):
    dct = {v:k for k, v in d.items()}
    return dct[name]

for file in os.listdir("models"):
    if file.endswith(".pkl") and file not in ('scaler.pkl', 'Метрики качества.pkl'):
        model = joblib.load(os.path.join("models", file))
        models[get_name_model(file)] = model

    # Сделать загрузку всех моделей в список, потом словарь, где ключи - это {selected_model}



st.set_page_config(page_title="Churn Predictor", page_icon="📉")
st.title("📉 Предсказание оттока клиента")
st.write("Введите данные о клиенте и узнайте, уйдёт он или останется")

with st.sidebar:
    st.header("Настройки")
    if st.button("Переобучить модель"):
        # Вызов функции
        st.success("Модель переобучена")

    selected_model_name = st.selectbox(
        "Выберите модель",
        [
            "Логистическая регрессия",
            "Дерево решений",
            "Случайный лес",
            "XGBoost",
            "SVM",
            "KNN",
            "Наивный Байес"
            ],
        index=0
    )
selected_model = models[selected_model_name]

left_col, right_col = st.columns(2)

# Ввод данных
with left_col:
    tenure = st.number_input("Длительность обслуживания (месяцев)", -100, 72, 12)
    monthly_charges = st.number_input("Ежемесячный платёж ($)", 0.0, 200.0, 70.0)
with right_col:
    contract = st.selectbox("Тип контракта", ["Month-to-month", "One year", "Two year"])
    internet_service = st.selectbox("Интернет-сервис", ["DSL", "Fiber optic", "No"])
total_charges = tenure * monthly_charges

col_1, col_2 = st.columns(3)[:2]
with col_1:
    if st.button("🔮 Предсказать отток"):
        # Создаём DataFrame с одним клиентом
        input_data = pd.DataFrame([{
            'tenure': tenure,
            'MonthlyCharges': monthly_charges,
            'TotalCharges': total_charges,
            'Contract': contract,
            'InternetService': internet_service
        }])

        # One-Hot Encoding
        input_encoded = pd.get_dummies(input_data, dtype=int)

        # Жёстко задаём правильный порядок колонок (такой же, как при обучении)
        expected_columns = [
            'tenure',
            'MonthlyCharges',
            'TotalCharges',
            'Contract_Month-to-month',
            'Contract_One year',
            'Contract_Two year',
            'InternetService_DSL',
            'InternetService_Fiber optic',
            'InternetService_No'
        ]

        # Добавляем отсутствующие колонки
        for col in expected_columns:
            if col not in input_encoded.columns:
                input_encoded[col] = 0

        input_encoded = input_encoded[expected_columns]

        # Масштабируем
        input_scaled = scaler.transform(input_encoded)

        # Предсказываем
        proba = selected_model.predict_proba(input_scaled)[0][1]

        if proba > 0.5:
            st.error(f"⚠️ Клиент **уйдёт** с вероятностью {proba:.1%}")
            st.info("Рекомендация: предложить скидку или улучшить обслуживание")
        else:
            st.success(f"✅ Клиент **останется** с вероятностью {(1 - proba):.1%}")

with col_2:
    st.text(f"Текущая модель: {selected_model}")

# Минимальный рабочий код с переключением
if 'show' not in st.session_state:
    st.session_state.show = False

if st.button("📈 Показать/Скрыть метрики"):
    st.session_state.show = not st.session_state.show

if st.session_state.show:
    st.dataframe(metrics.iloc[:, 1:])




