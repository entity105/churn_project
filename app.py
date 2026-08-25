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
tenure = st.number_input("Длительность обслуживания (месяцев)", -100, 72, 12)
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
    # print(f"input_data:\n{input_data}")

    # One-Hot Encoding
    input_encoded = pd.get_dummies(input_data, dtype=int)
    # print(f"input_encoded:\n{input_encoded}")

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
    # print(f"input_encoded2:\n{input_encoded}")

    input_encoded = input_encoded[expected_columns]
    # print(f'input_data:\n{input_data}\n\n')
    # print(f'input_encoded3:\n{input_encoded}\n\n')

    # Масштабируем
    input_scaled = scaler.transform(input_encoded)
    # print(f'input_scaled:\n{input_scaled}\n\n')

    # Предсказываем
    proba = model.predict_proba(input_scaled)[0][1]
    print(proba, end='\n\n')

    if proba > 0.5:
        st.error(f"⚠️ Клиент **уйдёт** с вероятностью {proba:.1%}")
        st.info("Рекомендация: предложить скидку или улучшить обслуживание")
    else:
        st.success(f"✅ Клиент **останется** с вероятностью {(1 - proba):.1%}")