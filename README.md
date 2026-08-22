# חיזוי הלוואות (Loan Prediction)

מערכת לחיזוי אישור הלוואות מבוססת למידת מכונה, עם שרת FastAPI וממשק משתמש בעברית (RTL) הכולל טופס חיזוי ולוח בקרה לניתוח ביצועי המודל.

הפרויקט מבוסס על [`Loan_Forecasting_Project.ipynb`](Loan_Forecasting_Project.ipynb) — המחברת (notebook) שבה פותח ואומן ה-pipeline המקורי (עיבוד נתונים, `StandardScaler` + `SVC` עם קרנל RBF, וחיפוש היפרפרמטר `C`). קוד השרת ב-`app.py` משכפל את אותה שיטת אימון בדיוק, כדי שהתוצאות יהיו עקביות בין המחברת לאפליקציה החיה.

## מבנה הפרויקט

- `app.py` — שרת FastAPI: טעינת המודל, endpoints לחיזוי ולניתוח ביצועים, הגשת קבצים סטטיים.
- `loan_model.pkl` — ה-pipeline המאומן (StandardScaler + SVC).
- `loan_data_v2.csv` — דאטהסט האימון.
- `Loan_Forecasting_Project.ipynb` — המחברת המקורית שממנה נגזר המודל.
- `static/` — ממשק המשתמש (HTML/CSS/JS ללא תלות בספריות frontend חיצוניות).

## הרצה מקומית

```bash
pip install -r requirements.txt
python app.py
```

השרת עולה בכתובת `http://localhost:8000`.

## API עיקרי

| Method | Path | תיאור |
| --- | --- | --- |
| GET | `/api/health` | בדיקת תקינות |
| GET | `/api/model-info` | מידע על המודל והגורמים המשפיעים |
| POST | `/api/predict` | חיזוי אישור הלוואה עבור בקשה בודדת |
| GET | `/api/model-analytics` | מטריקות ביצועים מלאות (מטריצת בלבול, דוח סיווג, גבול החלטה) |

תיעוד אינטראקטיבי מלא זמין ב-`/docs`.
