import streamlit as st
import pandas as pd

class View:
    """Отрисовка интерфейса"""

    @staticmethod
    def setup_page():
        st.set_page_config(page_title="Churn Predictor", page_icon="📉")
        st.title("📉 Предсказание оттока клиента")
        st.write("Введите данные о клиенте и узнайте, уйдёт он или останется")

    @staticmethod
    def sidebar_settings(model_names: list) -> str:
        with st.sidebar:
            st.header("Настройки")
            # retrain = st.button("Переобучить модель")
            selected_model = st.selectbox("Выберите модель", model_names, index=2)
        return selected_model

    @staticmethod
    def input_form(current_model_name : str) -> dict:
        # Ввод данных

        left_col, right_col = st.columns(2)
        with left_col:
            tenure = st.number_input("Длительность обслуживания (месяцев)", -100, 72, 12)
            monthly_charges = st.number_input("Ежемесячный платёж ($)", 0.0, 200.0, 70.0)
        with right_col:
            contract = st.selectbox("Тип контракта", ["Month-to-month", "One year", "Two year"])
            internet_service = st.selectbox("Интернет-сервис", ["DSL", "Fiber optic", "No"])
        total_charges = tenure * monthly_charges

        button = st.button("Предсказать отток")
        st.text(f"Текущая модель: {current_model_name}")

        return {
            "tenure": tenure,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
            "Contract": contract,
            "InternetService": internet_service,
            "button": button,
        }

    @staticmethod
    def show_result(proba):
        if proba > 0.5:
            st.error(f"⚠️ Клиент **уйдёт** с вероятностью {proba:.1%}")
            st.info("Рекомендация: предложить скидку или улучшить обслуживание")
        else:
            st.success(f"✅ Клиент **останется** с вероятностью {(1 - proba):.1%}")

    @staticmethod
    def show_metrics_toggle(metrics: pd.DataFrame):
        if 'show' not in st.session_state:
            st.session_state.show = False

        if st.button("Показать/Скрыть метрики"):
            st.session_state.show = not st.session_state.show

        if st.session_state.show:
            st.dataframe(metrics.iloc[:, 1:])

    # @staticmethod
    # def download_data():
    #     DATA_PATH = "../data/Telco-Customer-Churn_clean.csv"
    #     df = pd.read_csv(DATA_PATH)
    #     df.to_excel("temp.xlsx", index=False)
    #     with open("temp.xlsx", "rb") as f:
    #         st.download_button(
    #             label="Скачать Excel",
    #             data=f,
    #             file_name="telco_churn.xlsx",
    #             mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    #         )

    @staticmethod
    def show_data_button() -> bool:
        """Кнопка 'Показать исходные данные'"""
        return st.button("Показать исходные данные", key="show_data_button")

    @staticmethod
    def show_dataframe(df: pd.DataFrame):
        """Отображает DataFrame"""
        st.subheader("📋 Исходные данные")
        st.dataframe(df)