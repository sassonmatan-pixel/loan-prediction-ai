from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


APP_ROOT = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(APP_ROOT, 'static')
MODEL_PATH = r'C:\Users\Matan\loan_model.pkl'
DATASET_PATH = r'C:\Users\Matan\loan_data.csv'

app = FastAPI(title='Loan Forecast API', version='1.0.0')
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)


@app.on_event('startup')
async def startup_event():
    """On app startup, verify model is working correctly. If not, retrain it."""
    try:
        # Load current model
        current_model = load_model()
        dataset = get_dataset()
        
        # Quick test - predict on full dataset
        db2 = dataset.copy()
        db2 = db2.drop(['person_age','person_gender','cb_person_cred_hist_length','person_education','loan_intent','person_home_ownership'], axis=1)
        db2 = pd.get_dummies(db2, columns=['previous_loan_defaults_on_file'], drop_first=True)
        
        X = db2.drop('loan_status', axis=1).values
        y = db2['loan_status'].values
        
        preds = current_model.predict(X)
        approved_preds = (preds == 1).sum()
        approved_actual = (y == 1).sum()
        
        print(f"[STARTUP] Model predictions: {approved_preds} approved, actual: {approved_actual} approved", flush=True)
        
        # If model is predicting < 5% approvals when it should predict ~10%, retrain
        approve_rate = approved_preds / len(preds) if len(preds) > 0 else 0
        actual_approve_rate = approved_actual / len(y) if len(y) > 0 else 0
        
        if approve_rate < 0.05 and actual_approve_rate > 0.08:
            print("[STARTUP] WARNING: Model is under-approving. Retraining with class weight...", flush=True)
            _retrain_with_class_weight(dataset)
    except Exception as e:
        print(f"[STARTUP] Error during model check: {e}", flush=True)


def _retrain_with_class_weight(dataset: pd.DataFrame) -> None:
    """Retrain model with class_weight='balanced' to handle imbalance"""
    try:
        from sklearn.model_selection import train_test_split, cross_val_score
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        
        db2 = dataset.copy()
        db2 = db2.drop(['person_age','person_gender','cb_person_cred_hist_length','person_education','loan_intent','person_home_ownership'], axis=1)
        db2 = pd.get_dummies(db2, columns=['previous_loan_defaults_on_file'], drop_first=True)
        
        X = db2.drop('loan_status', axis=1).values
        y = db2['loan_status'].values
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        best_score = 0
        best_C = None
        best_pipeline = None
        
        C_range = [0.1, 1, 10]
        for c_val in C_range:
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', SVC(kernel='rbf', C=c_val, class_weight='balanced'))  # KEY: class_weight='balanced'
            ])
            
            scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')
            mean_score = scores.mean()
            
            if mean_score > best_score:
                best_score = mean_score
                best_C = c_val
                best_pipeline = pipeline
        
        best_pipeline.fit(X_train, y_train)
        test_acc = best_pipeline.score(X_test, y_test)
        
        joblib.dump(best_pipeline, MODEL_PATH)
        print(f"[RETRAIN] Model retrained with class_weight='balanced'. Test accuracy: {test_acc:.4f}", flush=True)
    except Exception as e:
        print(f"[RETRAIN] Failed: {e}", flush=True)

FEATURE_COLUMNS: List[str] = [
    'person_income',
    'person_emp_exp',
    'loan_amnt',
    'loan_int_rate',
    'loan_percent_income',
    'credit_score',
    'previous_loan_defaults_on_file_Yes',
]

MODEL_NAME = os.path.basename(MODEL_PATH)
DATASET_NAME = os.path.basename(DATASET_PATH)
MODEL_TYPE = 'SVM'


class LoanPredictionInput(BaseModel):
    person_income: float = Field(..., gt=0)
    person_emp_exp: int = Field(..., ge=0)
    loan_amnt: float = Field(..., gt=0)
    loan_int_rate: float = Field(..., ge=0)
    loan_percent_income: float = Field(..., ge=0)
    credit_score: int = Field(..., ge=300, le=850)
    previous_loan_defaults_on_file: str = Field(...)

    @field_validator('previous_loan_defaults_on_file')
    @classmethod
    def validate_default_flag(cls, value: str) -> str:
        normalized = str(value).strip().lower()
        if normalized not in {'yes', 'no'}:
            raise ValueError("previous_loan_defaults_on_file must be either 'Yes' or 'No'.")
        return normalized

    @property
    def normalized_payload(self) -> Dict[str, Any]:
        return {
            'person_income': float(self.person_income),
            'person_emp_exp': int(self.person_emp_exp),
            'loan_amnt': float(self.loan_amnt),
            'loan_int_rate': float(self.loan_int_rate),
            'loan_percent_income': float(self.loan_percent_income),
            'credit_score': int(self.credit_score),
            'previous_loan_defaults_on_file': self.previous_loan_defaults_on_file,
        }


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=400,
        content={
            'error': 'Invalid input payload.',
            'details': exc.errors(),
        },
    )


def load_model() -> Any:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f'Model file not found at {MODEL_PATH}')
    return joblib.load(MODEL_PATH)


def get_dataset() -> Optional[pd.DataFrame]:
    if not os.path.exists(DATASET_PATH):
        return None
    return pd.read_csv(DATASET_PATH)


def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy()
    prepared = prepared.drop(
        columns=[
            'person_age',
            'person_gender',
            'cb_person_cred_hist_length',
            'person_education',
            'loan_intent',
            'person_home_ownership',
        ],
        errors='ignore',
    )
    prepared = pd.get_dummies(prepared, columns=['previous_loan_defaults_on_file'], drop_first=True)
    return prepared


def build_model_frame(payload: Dict[str, Any]) -> pd.DataFrame:
    # Create row with exact feature names and order
    row = {
        'person_income': float(payload['person_income']),
        'person_emp_exp': int(payload['person_emp_exp']),
        'loan_amnt': float(payload['loan_amnt']),
        'loan_int_rate': float(payload['loan_int_rate']),
        'loan_percent_income': float(payload['loan_percent_income']),
        'credit_score': int(payload['credit_score']),
        'previous_loan_defaults_on_file': str(payload['previous_loan_defaults_on_file']).strip(),  # Keep original case
    }
    df = pd.DataFrame([row])
    
    # CRITICAL: ONE-HOT ENCODE exactly like training
    # Create the one-hot encoded column explicitly to match training behavior
    df = pd.get_dummies(df, columns=['previous_loan_defaults_on_file'], drop_first=True)
    
    # Ensure all feature columns exist in exact order
    for column in FEATURE_COLUMNS:
        if column not in df.columns:
            df[column] = 0
    
    # Return features in EXACT same order as training
    result = df[FEATURE_COLUMNS]
    
    # DEBUG: Log the transformation
    import sys
    print(f"DEBUG build_model_frame:", file=sys.stderr)
    print(f"  Input: {payload}", file=sys.stderr)
    print(f"  After OHE columns: {list(df.columns)}", file=sys.stderr)
    print(f"  Final vector: {result.values[0]}", file=sys.stderr)
    
    return result


@app.get('/api/model-info')
def model_info() -> Dict[str, Any]:
    dataset = get_dataset()
    sample_count = int(len(dataset)) if dataset is not None else None
    model = None
    accuracy: Optional[float] = None

    try:
        model = load_model()
    except Exception:
        model = None

    if model is not None and dataset is not None:
        try:
            prepared = preprocess_dataset(dataset)
            if 'loan_status' in prepared.columns:
                X = prepared.drop(columns=['loan_status'])
                y = prepared['loan_status']
                accuracy = float(model.score(X, y))
        except Exception:
            accuracy = None

    return {
        'model_name': 'מודל לחיזוי אישור הלוואות',
        'features': FEATURE_COLUMNS,
        'num_features': len(FEATURE_COLUMNS),
        'accuracy': accuracy,
        'num_samples': sample_count,
        'dataset_name': DATASET_NAME if os.path.exists(DATASET_PATH) else None,
        'model_type': MODEL_TYPE,
        'pipeline_status': 'פעיל' if model is not None else 'לא פעיל',
    }


@app.post('/api/debug-features')
def debug_features(payload: LoanPredictionInput) -> Dict[str, Any]:
    """Debug endpoint - shows exactly how features are processed"""
    feature_frame = build_model_frame(payload.normalized_payload)
    return {
        'feature_columns': FEATURE_COLUMNS,
        'feature_vector': feature_frame.values[0].tolist(),
        'feature_dataframe': feature_frame.to_dict(orient='records')[0],
    }


@app.post('/api/model-full-audit')
def model_full_audit() -> Dict[str, Any]:
    """Complete model audit - compares saved model vs expected features"""
    try:
        model = load_model()
        dataset = get_dataset()
        
        # Get feature names from saved model
        svc = model.named_steps['model']
        model_n_features = svc.n_features_in_
        
        # Preprocess dataset same as training
        db2 = dataset.copy()
        db2 = db2.drop(['person_age','person_gender','cb_person_cred_hist_length','person_education','loan_intent','person_home_ownership'], axis=1)
        db2 = pd.get_dummies(db2, columns=['previous_loan_defaults_on_file'], drop_first=True)
        
        actual_features = list(db2.drop('loan_status', axis=1).columns)
        X = db2.drop('loan_status', axis=1).values
        y = db2['loan_status'].values
        
        # Test predictions
        preds = model.predict(X)
        accuracy = (preds == y).mean()
        
        # Find misclassifications
        false_approvals = ((preds == 1) & (y == 0)).sum()
        false_rejections = ((preds == 0) & (y == 1)).sum()
        
        return {
            'model_n_features_in': model_n_features,
            'expected_features': FEATURE_COLUMNS,
            'actual_features_from_dataset': actual_features,
            'features_match': model_n_features == len(FEATURE_COLUMNS),
            'dataset_accuracy': float(accuracy),
            'false_approvals': int(false_approvals),
            'false_rejections': int(false_rejections),
            'approved_count': int((preds == 1).sum()),
            'rejected_count': int((preds == 0).sum()),
            'actual_approved': int((y == 1).sum()),
            'actual_rejected': int((y == 0).sum()),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Audit failed: {str(e)}')


@app.post('/api/predict')
def predict(payload: LoanPredictionInput) -> Dict[str, Any]:
    try:
        model = load_model()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail='Model file is missing.') from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail='Unable to load model.') from exc

    feature_frame = build_model_frame(payload.normalized_payload)
    
    # DEBUG: Log feature vector
    import sys
    feature_vector = feature_frame.values[0]
    print(f"DEBUG: Feature columns: {FEATURE_COLUMNS}", file=sys.stderr)
    print(f"DEBUG: Feature vector: {feature_vector}", file=sys.stderr)
    print(f"DEBUG: Input payload: {payload.normalized_payload}", file=sys.stderr)

    try:
        prediction = int(model.predict(feature_frame)[0])
        print(f"DEBUG: Model prediction: {prediction}", file=sys.stderr)
        
        approved = prediction == 1
        response: Dict[str, Any] = {
            'prediction': prediction,
            'approved': approved,
            'message': 'ההלוואה מאושרת' if approved else 'ההלוואה לא מאושרת',
        }
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(feature_frame)[0]
            if len(probabilities) > 1:
                response['probability'] = round(float(probabilities[1]), 4)
        return response
    except Exception as exc:
        raise HTTPException(status_code=500, detail='Prediction failed using the saved model.') from exc


@app.get('/api/health')
def health() -> Dict[str, str]:
    return {'status': 'ok'}


@app.post('/api/retrain-model')
def retrain_model() -> Dict[str, Any]:
    """EMERGENCY: Retrain model from scratch using original notebook code"""
    try:
        from sklearn.model_selection import cross_val_score
        from sklearn.preprocessing import StandardScaler as ScalerClass
        from sklearn.pipeline import Pipeline as PipelineClass
        from sklearn.metrics import accuracy_score
        
        # Load and preprocess data
        dataset = pd.read_csv(DATASET_PATH)
        db2 = dataset.copy()
        db2 = db2.drop(['person_age','person_gender','cb_person_cred_hist_length','person_education','loan_intent','person_home_ownership'], axis=1)
        db2 = pd.get_dummies(db2, columns=['previous_loan_defaults_on_file'], drop_first=True)
        
        X = db2.drop('loan_status', axis=1).values
        y = db2['loan_status'].values
        
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Hyperparameter tuning
        C_range = [0.1, 1, 10]
        best_score = 0
        best_C = None
        best_pipeline = None
        
        for c_val in C_range:
            loan_pipeline = PipelineClass([
                ('scaler', ScalerClass()),
                ('model', SVC(kernel='rbf', C=c_val))
            ])
            
            scores = cross_val_score(loan_pipeline, X_train, y_train, cv=5, scoring='accuracy')
            mean_score = scores.mean()
            
            if mean_score > best_score:
                best_score = mean_score
                best_C = c_val
                best_pipeline = loan_pipeline
        
        # Fit and evaluate
        best_pipeline.fit(X_train, y_train)
        test_acc = best_pipeline.score(X_test, y_test)
        full_acc = best_pipeline.score(X, y)
        
        # Save
        joblib.dump(best_pipeline, MODEL_PATH)
        
        return {
            'status': 'success',
            'best_C': best_C,
            'cv_accuracy': float(best_score),
            'test_accuracy': float(test_acc),
            'full_accuracy': float(full_acc),
            'message': 'Model retrained and saved successfully'
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Retraining failed: {str(e)}')



@app.get('/')
def root() -> FileResponse:
    return FileResponse(os.path.join(STATIC_DIR, 'index.html'))


app.mount('/static', StaticFiles(directory=STATIC_DIR), name='static')


if __name__ == '__main__':
    import uvicorn

    uvicorn.run('app:app', host='0.0.0.0', port=8000, reload=False)
