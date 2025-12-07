import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

#загружаем столбцы для датасета, который будет использоваться для ml
df_ml = dataflow[['Gender', 'ParentalEducation', 'GPA_numeric', 'study_hours']].copy()
#убираем пустые значение
df_ml = df_ml.dropna()

#так как у нас в таблице есть столбцы с текстом, то нам необходимо перейти от текста к числам, так как иначе модель просто не поймет
#функция get_dummies смотрит на категориальные столбцы и просто преобразует их так, чтобы значение определялось либо 1, либо 0
#drop_first нужен, когда у нас несколько возможных вариантов категорий.

# Просто выведи названия и выбери нужные кликом
df_ml.columns

# Потом копируешь точные названия в код:
df_ml = pd.get_dummies(df_ml, columns=['Gender', 'ParentalEducation', 'study_hours'], drop_first=True)
df_ml.head()

# После подготовки таблицы для обучения мы должны выбрать параметр, который должен быть предсказан
y = df_ml['GPA_numeric']          # это то, что хотим угадать
X = df_ml.drop('GPA_numeric', axis=1)   # всё остальное — признаки

# Теперь нужно разделить данные для тренировки(80%) и проверки(20%) 
#random_state=42 отвечает за то, как перемешиваются значения
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Теперь нужно выбрать модель для работы
# 1. Создаём модель
model = RandomForestRegressor(n_estimators=300, random_state=42)

# 2. Обучаем её на тренировочных данных
model.fit(X_train, y_train)



pred = model.predict(X_test)

print(f"R² = {r2_score(y_test, pred):.3f}")
print(f"Средняя ошибка = {mean_absolute_error(y_test, pred):.3f} балла из 4.0")

#R² = 0.036
#Средняя ошибка = 0.737 балла из 4.0

# 1. Модель на ПЕРВОМ датасете (Student_performance_data_ — 2392 записи)
df1 = pd.read_csv('Student_performance_data _.csv')

df1_ml = df1[['Gender', 'ParentalEducation', 'GPA', 'StudyTimeWeekly',
              'Extracurricular', 'Sports', 'Music', 'Volunteering']].copy()

df1_ml[['Extracurricular','Sports','Music','Volunteering']] = df1_ml[['Extracurricular','Sports','Music','Volunteering']].fillna(0)
df1_ml['study_hours'] = pd.cut(df1_ml['StudyTimeWeekly'], bins=[0,5,10,100], labels=['low','medium','high'])

df1_ml = pd.get_dummies(df1_ml, columns=['Gender','ParentalEducation','study_hours'], drop_first=True)

y1 = df1_ml['GPA']
X1 = df1_ml.drop('GPA', axis=1)

model1 = RandomForestRegressor(n_estimators=300, random_state=42)
model1.fit(X1, y1)
print("Датасет 1 → R² =", round(model1.score(X1, y1), 3))

# 2. Модель на ВТОРОМ датасете (StudentsPerformance — 1000 записей)
df2 = pd.read_csv('StudentsPerformance.csv')

df2['GPA_numeric'] = df2[['math score','reading score','writing score']].mean(axis=1) / 25
df2_ml = df2[['gender', 'parental level of education', 'test preparation course']].copy()
df2_ml.columns = ['Gender', 'ParentalEducation', 'study_hours']
df2_ml['study_hours'] = df2_ml['study_hours'].map({'none':'low', 'completed':'high'})

df2_ml = pd.get_dummies(df2_ml, columns=['Gender','ParentalEducation','study_hours'], drop_first=True)
df2_ml['GPA_numeric'] = df2['GPA_numeric']

y2 = df2_ml['GPA_numeric']
X2 = df2_ml.drop('GPA_numeric', axis=1)

model2 = RandomForestRegressor(n_estimators=300, random_state=42)
model2.fit(X2, y2)
print("Датасет 2 → R² =", round(model2.score(X2, y2),3))

#Датасет 1 → R² = 0.841
#Датасет 2 → R² = 0.141
