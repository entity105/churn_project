import pandas as pd
from sklearn.model_selection import train_test_split        # готовая функция для разделения данных на train/test
from sklearn.linear_model import LogisticRegression         # готовая реализацию логистической регрессии (класс)
from sklearn.metrics import accuracy_score, roc_auc_score   # функции для оценки качества
from sklearn.preprocessing import StandardScaler            # Класс для масштабирования

# Загружаем очищенные данные
df = pd.read_csv('../data/Telco-Customer-Churn_clean.csv')

# Отделяем признаки (X) от целевой переменной (y)
X = df.drop('Churn', axis=1)    # Churn - это столбец с 0 или 1
y = df['Churn']

# Превращаем текстовые столбцы в числа (One-Hot Encoding)
X = pd.get_dummies(X, drop_first=True)

# Разделяем на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Масштабирование
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Обучаем scaler на тренировочной выборке
X_test_scaled = scaler.transform(X_test)        # То же самое на тестовой

# Обучаем логистическую регрессию
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# Предсказания
y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]

# Оценка качества
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.3f}")

# Получаем названия признаков
feature_names = X_train.columns

# Получаем коэффициенты модели
coefficients = model.coef_[0]

# Сортируем и выводим топ-5 самых влиятельных признаков
coeff_df = pd.DataFrame({'feature': feature_names, 'coef': coefficients})
coeff_df['abs_coef'] = coeff_df['coef'].abs()
coeff_df = coeff_df.sort_values('abs_coef', ascending=False)

print("\n=== Top 5 most important features for churn prediction ===")
print(coeff_df.head(5))

# import joblib
#
# # Сохраняем модель и scaler
# joblib.dump(model, '../models/churn_model.pkl')
# joblib.dump(scaler, '../models/scaler.pkl')
# print("Model and scaler saved successfully!")