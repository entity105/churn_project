import numpy as np
import pandas as pd
import os
import joblib

from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier


class CreateFitModel:
    """Создаёт, обучает, сохраняет модель. На вход подаётся путь к очищенному файлу с данными"""
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODELS_DIR = os.path.join(BASE_DIR, '..', 'models')

    def __init__(self, path_to_csv:str):
        self.path = path_to_csv

        self.df_data = self.__parse_data()
        self.scaler = self.load_model("scaler")

        self.metrics = pd.DataFrame(columns=['models', 'models_name', 'accuracy', 'ROC-AUC'])
        self.train_test = self.__get_train_data()

    def fit_all_models(self, exclude=()):
        """Вызывает все методы, оканчивающиеся на '_fit' и не содержащиеся в exclude """
        methods_names = [m for m in dir(self)
                         if callable(getattr(self, m)) and
                         m.endswith('_fit') and
                         m not in exclude]

        for method_name in methods_names:
            print(f"Обучение {method_name}...")
            method = getattr(self, method_name)
            try:
                method()
            except Exception as e:
                print(f"Ошибка в {method_name}: {e}")
                raise
        print("Обучение завершено !")
        self.save_model(self.metrics.sort_values("accuracy", ascending=False), "Метрики качества")

    def __parse_data(self, *args, **kwargs):
        """Читает csv и сохраняет df"""
        return pd.read_csv(self.path, *args, **kwargs)

    def fit_model(self, file_name:str, name_model:str, model):
        """Обучение, тестирование и сохранение модели"""
        new_row = pd.DataFrame([[model, name_model]], columns=['models', 'models_name'])
        self.metrics = pd.concat([self.metrics, new_row], ignore_index=True)

        x_train, x_test, y_train, y_test = self.train_test
        x_train_scaled, x_test_scaled = self.__scale_data(x_train, x_test)

        model.fit(x_train_scaled, y_train)
        self.test_model(model, x_test_scaled, y_test)
        self.save_model(model, file_name)

    def logistic_regression_fit(self):
        """Метод-сборщик (Facade)"""
        model = LogisticRegression(max_iter=1000)
        # self.model = model
        self.fit_model(file_name='logistic_regression_model', name_model='Логистическая регрессия', model=model)

    def decision_tree_fit(self):
        """Метод-сборщик (Facade)"""
        model = DecisionTreeClassifier(
            max_depth=5,
            min_samples_split=20,
            random_state=42
        )
        # self.model = model
        self.fit_model(file_name='decision_tree_model', name_model="Дерево решений" , model=model)

    def random_forest_fit(self):
        """Метод-сборщик (Facade)"""
        model = RandomForestClassifier(
            n_estimators=200,  # достаточно деревьев
            max_depth=15,  # ограничиваем глубину
            min_samples_split=10,  # уменьшает переобучение
            min_samples_leaf=4,  # сглаживание
            max_features='sqrt',  # стандарт для регрессии
            max_samples=0.8,  # используем 80% данных для каждого дерева
            bootstrap=True,  # bootstrap выборки
            oob_score=True,  # оценка на out-of-bag
            random_state=42,  # воспроизводимость
            n_jobs=-1  # все ядра
        )
        self.fit_model(file_name='random_forest_model', name_model="Случайный лес" , model=model)

    def svm_fit(self):
        """Метод-сборщик (Facade)"""
        model = SVC(
            kernel='rbf',
            C=1.0,
            probability=True,  # Чтобы можно было получить predict_proba
            random_state=42
        )
        self.fit_model('svm_model', name_model="SVM" , model=model)

    def knn_fit(self):
        """Метод-сборщик (Facade) для KNN"""
        model = KNeighborsClassifier(
            n_neighbors=5,  # количество соседей
            weights='uniform',  # 'uniform' — все соседи равны, 'distance' — с весами
            algorithm='auto',  # 'auto', 'ball_tree', 'kd_tree', 'brute'
            leaf_size=30,  # параметр для BallTree/KDTree
            p=2,  # 2 — евклидово, 1 — манхэттенское расстояние
            metric='minkowski',  # метрика расстояния
            n_jobs=-1  # используем все ядра
        )
        self.fit_model('knn_model', name_model="KNN" , model=model)

    def xgboost_fit(self):
        """Метод-сборщик (Facade) для XGBoost"""
        model = XGBClassifier(
            n_estimators=200,  # количество деревьев
            max_depth=5,  # глубина деревьев
            learning_rate=0.1,  # шаг обучения
            subsample=0.8,  # доля объектов для каждого дерева
            colsample_bytree=0.8,  # доля признаков для каждого дерева
            gamma=0.1,  # минимальное уменьшение потерь для разбиения
            reg_alpha=0.1,  # L1-регуляризация
            reg_lambda=1.0,  # L2-регуляризация
            eval_metric='logloss',  # метрика для оценки
            random_state=42,
            n_jobs=-1  # все ядра
        )
        self.fit_model(
            file_name='xgboost_model',
            name_model='XGBoost',
            model=model
        )

    def naive_bayes_fit(self):
        """Метод-сборщик (Facade) для Gaussian Naive Bayes"""
        model = GaussianNB(
            var_smoothing=1e-9  # сглаживание для стабильности
        )
        self.fit_model(
            file_name='naive_bayes_model',
            name_model='Наивный Байес',
            model=model
        )

    def save_model(self, model, name:str, dont_rewrite=False):
        """Сохраняет объект в models. По умолчанию объект берётся из self.
        dont_rewrite = True - не даёт перезаписать объект"""
        path = self._get_model_path(name)

        if self.is_file_exist(path):
            if dont_rewrite:
                print("Объект уже существует")
                return
            message = "Объект перезаписан"
        else:
            message = "Объект сохранён"

        joblib.dump(model, path)
        print(message)

    def load_model(self, name:str):
        """Подгружает объект из папки models"""
        path = self._get_model_path(name)
        if self.is_file_exist(path):
            return joblib.load(path)

    def test_model(self, model, x_test_scaled, y_test):
        """Тестирует модель. Вычисляет accuracy и ROC-AUC метрики"""
        y_pred = model.predict(x_test_scaled)
        y_proba = model.predict_proba(x_test_scaled)[:, 1]
        accuracy = np.round(accuracy_score(y_test, y_pred), 3)
        roc_auc = np.round(roc_auc_score(y_test, y_proba), 3)
        self.metrics.loc[self.metrics["models"] == model, ['accuracy', 'ROC-AUC']] = [accuracy, roc_auc]

    @staticmethod
    def is_file_exist(file_path:str):
        """Проверка существования файла"""
        if os.path.exists(file_path):
            return True
        return False

    def _get_model_path(self, name):
        """Выдаёт путь к файлу в папке models"""
        return os.path.join(self.MODELS_DIR, f'{name}.pkl')

    def __train_scaler(self, x_train):
        """Обучает и сохраняет масштабатор"""
        scaler = StandardScaler()
        scaler.fit(x_train)

        self.scaler = scaler
        self.save_model(scaler, "scaler")

    def __scale_data(self, x_train, x_test) -> tuple:
        """Возвращает масштабированные данные для обучения и тестирования"""
        if not self.scaler:
            self.__train_scaler(x_train)
        try:
            x_train_scaled = self.scaler.transform(x_train)
            x_test_scaled = self.scaler.transform(x_test)
        except Exception as e:
            raise Exception(f"Ошибка масштабирования данных. Ошибка: {e}")
        return x_train_scaled, x_test_scaled

    def __get_train_data(self, select_cols=None, mark:str = 'Churn') -> tuple:
        """Возвращает данные для обучения и тестирования в формате: X_train, X_test, y_train, y_test"""
        if not select_cols:
            select_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'Contract', 'InternetService']
        try:
            X = self.df_data[select_cols]
            y = self.df_data[mark]
            X = pd.get_dummies(X, dtype=int)
            return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        except Exception as e:
            raise ValueError(f"Неверные исходные данные для обучения, ошибка: \n {e}")
        except KeyError:
            raise KeyError("Выбраны несуществующие столбцы или не выбраны вовсе")


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    path_csv = os.path.join(BASE_DIR, '..', 'data', 'Telco-Customer-Churn_clean.csv')
    pipeline = CreateFitModel(path_csv)
    pipeline.fit_all_models()
    print(pipeline.metrics.iloc[:, 1:])