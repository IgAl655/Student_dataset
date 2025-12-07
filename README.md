# Student_dataset
В проекте проведён сравнительный анализ двух популярных открытых датасетов по успеваемости студентов (2392 и 1000 записей).
Попытка объединения данных привела к R² ≈ 0.04 из-за фундаментальных различий в шкалах оценок и распределениях признаков.
При обучении моделей отдельно на каждом датасете Random Forest показал:
• R² = 0.841 — на современном датасете (GPA 0–4, широкий разброс)
• R² = 0.141 — на классическом датасете (оценки 60–100 баллов у 95 % студентов)
Главный вывод: качество предсказательной модели определяется не алгоритмом, а однородностью и информативностью исходных данных




Графики
<img width="1079" height="656" alt="image" src="https://github.com/user-attachments/assets/61992a75-f454-4dd7-b9b6-3a603b721306" />
<img width="850" height="661" alt="image" src="https://github.com/user-attachments/assets/f5fc2415-3689-4c0a-bd5f-bd94bc5cabc4" />
<img width="1066" height="688" alt="image" src="https://github.com/user-attachments/assets/73f385b2-c72f-4a3f-830a-b947384c2301" />
<img width="892" height="666" alt="image" src="https://github.com/user-attachments/assets/dae106b5-18a7-4546-963d-d7c2d4d364ac" />
<img width="969" height="666" alt="image" src="https://github.com/user-attachments/assets/c6db1615-0cae-4ea9-b1af-bb395086b12e" />
<img width="1046" height="685" alt="image" src="https://github.com/user-attachments/assets/ae5b9cd9-e59f-451a-a80e-41187cd0cc7d" />
