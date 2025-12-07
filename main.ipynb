import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# ====================== df1 ======================
df1 = pd.read_csv('Student_performance_data _.csv')
df2 = pd.read_csv('StudentsPerformance.csv')

# --- dataflow1 (из современного датасета) ---
dataflow1 = df1[['Gender', 'ParentalEducation', 'GPA', 'StudyTimeWeekly',
                 'Extracurricular', 'Sports', 'Music', 'Volunteering']].copy()

# Оставляем числовой GPA (0-4) и делаем отдельно буквенный
dataflow1['GPA_numeric'] = dataflow1['GPA']                                # сохраняем число
dataflow1['GPA_letter'] = dataflow1['GPA'].apply(lambda x: 
    'A' if x >= 3.5 else 'B' if x >= 3.0 else 'C' if x >= 2.5 else 'D' if x >= 2.0 else 'F')

# Категории часов
dataflow1['study_hours'] = pd.cut(dataflow1['StudyTimeWeekly'],
                                  bins=[0, 9.5, 14.5, float('inf')],
                                  labels=['low', 'medium', 'high'])

# Пол и образование родителей
dataflow1['Gender'] = dataflow1['Gender'].map({0: 'male', 1: 'female'})
dataflow1['ParentalEducation'] = dataflow1['ParentalEducation'].map({
    0: 'None', 1: 'High School', 2: 'Some College', 3: "Bachelor's", 4: "Master's", 5: 'Doctorate'
})

dataflow1['source'] = 'Modern_dataset'

# --- dataflow2 (из старого датасета) ---
dataflow2 = df2[['gender', 'parental level of education',
                 'math score', 'reading score', 'writing score',
                 'test preparation course']].copy()

dataflow2 = dataflow2.rename(columns={'gender': 'Gender'})

# Числовой GPA (0-100 → переводим в 0-4)
dataflow2['GPA_numeric'] = (df2[['math score', 'reading score', 'writing score']].mean(axis=1)) / 25

# Буквенный GPA
dataflow2['GPA_letter'] = dataflow2['GPA_numeric'].apply(lambda x:
    'A' if x >= 3.6 else 'B' if x >= 3.2 else 'C' if x >= 2.8 else 'D' if x >= 2.4 else 'F')

# Родители
dataflow2['ParentalEducation'] = df2['parental level of education'].map({
    "some high school": "High School",
    "high school": "High School",
    "some college": "Some College",
    "associate's degree": "Some College",
    "bachelor's degree": "Bachelor's",
    "master's degree": "Master's"
})

# Часы учёбы
dataflow2['study_hours'] = df2['test preparation course'].map({'none': 'low', 'completed': 'high'})

# Пустые внеурочки
dataflow2[['Extracurricular','Sports','Music','Volunteering']] = np.nan

dataflow2['source'] = 'Exams_dataset'

# Урезаем до нужных колонок
dataflow2 = dataflow2[['Gender', 'ParentalEducation', 'GPA_numeric', 'GPA_letter',
                       'study_hours', 'Extracurricular', 'Sports', 'Music', 'Volunteering', 'source']]

# ====================== ОБЪЕДИНЕНИЕ ======================
# Приводим dataflow1 к тем же колонкам
dataflow1 = dataflow1[['Gender', 'ParentalEducation', 'GPA_numeric', 'GPA_letter',
                       'study_hours', 'Extracurricular', 'Sports', 'Music', 'Volunteering', 'source']]

# Финальный датафрейм
dataflow = pd.concat([dataflow1, dataflow2], ignore_index=True)

print(f"Итого строк: {len(dataflow)}")
dataflow.head()
