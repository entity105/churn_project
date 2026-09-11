import joblib
import os
import pandas as pd

MODELS_DIR = "models"

MODEL_NAMES = {
        'logistic_regression_model.pkl' : 'Логистическая регрессия',
        'decision_tree_model.pkl' : 'Дерево решений',
        'knn_model.pkl' : 'KNN',
        'random_forest_model.pkl' : 'Случайный лес',
        'svm_model.pkl' : 'SVM',
        'naive_bayes_model.pkl' : 'Наивный Байес',
        'xgboost_model.pkl' : 'XGBoost'
    }


class Model:
    """Загрузка моделей, масштабирование и предсказание"""

    def __init__(self, models_dir : str = MODELS_DIR):
        self.models_dir = models_dir

        self.scaler = None
        self.metrics = None
        self.models = {}

        self.__load_all()

    def __load_all(self):
        self.scaler = joblib.load(os.path.join(self.models_dir, "scaler.pkl"))
        self.metrics = joblib.load(os.path.join(self.models_dir, "Метрики качества.pkl"))

        for file_name in os.listdir(self.models_dir):
            if file_name.endswith('.pkl') and file_name not in {"scaler.pkl", "Метрики качества.pkl"}:
                name = MODEL_NAMES.get(file_name, file_name)
                self.models[name] = joblib.load(os.path.join(self.models_dir, file_name))

    def get_models_name(self) -> list:
        return list(self.models.keys())

    def get_metrics(self) -> pd.DataFrame:
        return self.metrics

    def predict(self, model_name:str, input_data:dict):
        input_data = pd.DataFrame([input_data])
        model = self.models[model_name]

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
        input_scaled = self.scaler.transform(input_encoded)

        # Предсказываем
        return model.predict_proba(input_scaled)[0][1]
