import streamlit as st
import pandas as pd
import joblib

# Загрузка модели и scaler
model = joblib.load('models/churn_model.pkl')
scaler = joblib.load('models/scaler.pkl')

st.set_page_config(page_title="Churn Predictor", page_icon="📉")
st.title("📉 Предсказание оттока клиента")
st.write("Введите данные о клиенте и узнайте, уйдёт он или останется")

# Ввод данных
tenure = st.number_input("Длительность обслуживания (месяцев)", 0, 72, 12)
monthly_charges = st.number_input("Ежемесячный платёж ($)", 0.0, 200.0, 70.0)
total_charges = tenure * monthly_charges
contract = st.selectbox("Тип контракта", ["Month-to-month", "One year", "Two year"])
internet_service = st.selectbox("Интернет-сервис", ["DSL", "Fiber optic", "No"])

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
    input_encoded = pd.get_dummies(input_data, drop_first=True)

    # Жёстко задаём правильный порядок колонок (такой же, как при обучении)
    expected_columns = [
        'tenure',
        'MonthlyCharges',
        'TotalCharges',
        'Contract_One year',
        'Contract_Two year',
        'InternetService_Fiber optic',
        'InternetService_No'
    ]

    # Добавляем отсутствующие колонки
    for col in expected_columns:
        if col not in input_encoded.columns:
            input_encoded[col] = 0

    # Убираем лишние колонки и приводим к нужному порядку
    input_encoded = input_encoded[expected_columns]

    # Масштабируем
    input_scaled = scaler.transform(input_encoded)

    # Предсказываем
    proba = model.predict_proba(input_scaled)[0][1]

    if proba > 0.5:
        st.error(f"⚠️ Клиент **уйдёт** с вероятностью {proba:.1%}")
        st.info("Рекомендация: предложить скидку или улучшить обслуживание")
    else:
        st.success(f"✅ Клиент **останется** с вероятностью {(1 - proba):.1%}")