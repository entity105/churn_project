from src.model import Model
from src.view import View

class Controller:
    def __init__(self):
        self.model = Model()
        self.view = View()

    def run(self):
        self.view.setup_page()

        models_name = self.model.get_models_name()
        selected_model_name = self.view.sidebar_settings(models_name)
        data = self.view.input_form(selected_model_name)

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
