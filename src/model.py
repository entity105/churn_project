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

EXPECTED_COLUMNS = [
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


class Model:
    """Загрузка моделей, масштабирование и предсказание"""

    def __init__(self, models_dir : str = MODELS_DIR):
        self.models_dir = models_dir
        self.df_data = None

        self.scaler = None
        self.metrics = None
        self.models = {}

        self.__load_all()

    @staticmethod
    def get_path_to_data(file_name : str = 'Telco-Customer-Churn_clean.csv'):
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(BASE_DIR, '..', 'data', file_name)

    def __load_all(self):
        if len(os.listdir(self.models_dir)) == 0:
            self.train_models()

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

    def predict(self, model_name: str, input_data: dict):
        input_data = pd.DataFrame([input_data])
        model = self.models[model_name]

        # One-Hot Encoding
        input_encoded = pd.get_dummies(input_data, dtype=int)

        # Добавляем отсутствующие колонки
        for col in EXPECTED_COLUMNS:
            if col not in input_encoded.columns:
                input_encoded[col] = 0

        input_encoded = input_encoded[EXPECTED_COLUMNS]

        # Масштабируем
        input_scaled = self.scaler.transform(input_encoded)

        # Предсказываем
        return model.predict_proba(input_scaled)[0][1]

    def load_data(self) -> pd.DataFrame:
        """Загружает исходный датасет (один раз, кэшируется)"""
        if self.df_data is None:
            self.df_data = pd.read_csv(self.get_path_to_data('Telco-Customer-Churn.csv'))
        return self.df_data

    def train_models(self):
        from src.train_simple_model import CreateFitModel
        path_csv = self.get_path_to_data('Telco-Customer-Churn_clean.csv')
        pipeline = CreateFitModel(path_csv)
        pipeline.fit_all_models()