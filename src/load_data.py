import pandas as pd

# Загрузка данных
df = pd.read_csv('../data/Telco-Customer-Churn.csv')

print("=== ДО ОЧИСТКИ ===")
print(df.info())

# 1. Удаляем столбец customerID (не влияет на отток)
df.drop('customerID', axis=1, inplace=True)

# 2. Превращаем TotalCharges в число (ошибочные строки станут NaN)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# 3. Удаляем строки с пропусками (их немного)
initial_rows = len(df)
df = df.dropna(subset=['TotalCharges'])
print(f"\nУдалено строк с пропусками: {initial_rows - len(df)}")

# 4. Превращаем Churn в 0/1 (Yes → 1, No → 0)
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

print("\n=== ПОСЛЕ ОЧИСТКИ ===")
print(df.info())
print(f"\nРаспределение оттока:\n{df['Churn'].value_counts()}")

# Сохраняем очищенный датасет (опционально)
df.to_csv('../data/Telco-Customer-Churn_clean.csv', index=False)
print("\nОчищенный датасет сохранён!")