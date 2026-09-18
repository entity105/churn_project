from src.model import Model
from src.view import View

class Controller:
    def __init__(self):
        self.view = View()
        self.model = Model()

    def run(self):
        self.view.setup_page()

        models_name = self.model.get_models_name()
        selected_model_name = self.view.sidebar_settings(models_name)
        data = self.view.input_form(selected_model_name)
        # self.view.download_data()

        if data["button"]:
            input_data = {
                "tenure": data["tenure"],
                "MonthlyCharges": data["MonthlyCharges"],
                "TotalCharges": data["TotalCharges"],
                "Contract": data["Contract"],
                "InternetService": data["InternetService"]
            }
            proba = self.model.predict(selected_model_name, input_data)
            self.view.show_result(proba)

        self.view.show_metrics_toggle(self.model.get_metrics())

        # Кнопка "Показать данные"
        if self.view.show_data_button():
            df = self.model.load_data()
            self.view.show_dataframe(df)
