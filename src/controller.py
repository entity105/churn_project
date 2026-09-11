from src.model import Model
from src.view import View

class Controller:
    def __init__(self):
        self.model = Model()
        self.view = View()

    def run(self):
        self.view.setup_page()

        models_name = self.model.get_models_name()
        retrain, selected_model = self.view.sidebar_settings(models_name)
        data = self.view.input_form()

        if retrain:
            pass

        if data["button"]:
            input_data = {
                "tenure": data["tenure"],
                "MonthlyCharges": data["MonthlyCharges"],
                "TotalCharges": data["TotalCharges"],
                "Contract": data["Contract"],
                "InternetService": data["InternetService"],
            }
            proba = self.model.predict(selected_model, input_data)
            self.view.show_result(proba)

        self.view.show_metrics_toggle(self.model.get_metrics())
