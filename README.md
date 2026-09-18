# Churn Prediction — прогноз оттока клиентов

Проект по предсказанию оттока клиентов телеком-оператора на основе датасета [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).

## Задача

Предсказать, уйдёт ли клиент (Churn = 1) или останется (Churn = 0), на основе его демографических данных, подключённых услуг и платежей. Бизнес-цель — заранее выявлять клиентов в зоне риска и удерживать их.

## Данные

- **Источник:** Kaggle, Telco Customer Churn
- **Размер:** 7043 записи, 21 признак
- **Признаки:** tenure, MonthlyCharges, TotalCharges, Contract, InternetService, PaymentMethod и др.

## Стек

- Python 3.11
- Pandas, NumPy — обработка данных
- Scikit-learn — модели ML
- XGBoost, LightGBM — градиентный бустинг
- Matplotlib, Seaborn — визуализация
- Streamlit — веб-интерфейс
- Joblib — сохранение моделей

## Пайплайн

1. **EDA** — анализ распределений, корреляций, пропусков
2. **Предобработка** — One-Hot Encoding, масштабирование (StandardScaler)
3. **Обучение моделей:**
   - Логистическая регрессия
   - Дерево решений
   - Случайный лес
   - XGBoost
   - SVM
   - KNN
   - Наивный Байес
4. **Оценка качества** — Accuracy, ROC-AUC
5. **Интерпретация** — важность признаков
6. **Деплой** — Streamlit-приложение

## Результаты

| Модель | Accuracy | ROC-AUC |
|--------|----------|---------|
| Логистическая регрессия | 0.783 | **0.831** |
| Random Forest | 0.783 | 0.819 |
| XGBoost | ... | ... |
| KNN | 0.764 | 0.773 |

**Лучшая модель:** логистическая регрессия (ROC-AUC 0.831)

**Ключевые факторы оттока:**
- Стаж клиента (tenure)
- Ежемесячный платёж (MonthlyCharges)
- Тип контракта (помесячный → высокий риск)


## Запуск

### 1. Установка зависимостей

```bash
pip install -r requirements.txt
```

### 2. Обучение моделей
```bash
cd src
python train_simple_model.py
```

### 3. Запуск веб-приложения
```bash
streamlit run app.py
```


