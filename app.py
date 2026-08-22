from __future__ import annotations

import hashlib
import logging
import os
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
logger = logging.getLogger('loan_forecast')

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(APP_ROOT, 'static')
MODEL_PATH = os.path.join(APP_ROOT, 'loan_model.pkl')
DATASET_PATH = os.path.join(APP_ROOT, 'loan_data_v2.csv')

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
    """Log service readiness on boot.

    Deliberately does no model/dataset I/O here: startup must stay fast, and
    scoring the SVM over the full training set is done on demand instead,
    via /api/model-analytics (which caches its result for the process lifetime).
    """
    logger.info('Loan prediction service is ready.')


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
MODEL_ACCURACY = 0.857
MODEL_SAMPLE_COUNT = 12000


class LoanPredictionInput(BaseModel):
    """Request payload for /api/predict — the 7 features the saved model expects."""

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
        """Normalize the default-history flag to lowercase 'yes'/'no'."""
        normalized = str(value).strip().lower()
        if normalized not in {'yes', 'no'}:
            raise ValueError("previous_loan_defaults_on_file must be either 'Yes' or 'No'.")
        return normalized

    @property
    def normalized_payload(self) -> Dict[str, Any]:
        """This input as a plain dict with consistent types, ready for build_model_frame()."""
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
    """Return a 400 with the validation error details, instead of FastAPI's default 422."""
    return JSONResponse(
        status_code=400,
        content={
            'error': 'Invalid input payload.',
            'details': exc.errors(),
        },
    )


@lru_cache(maxsize=1)
def load_model() -> Any:
    """Load and cache the trained Pipeline (StandardScaler + SVC) from MODEL_PATH."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f'Model file not found at {MODEL_PATH}')
    return joblib.load(MODEL_PATH)


def get_dataset() -> Optional[pd.DataFrame]:
    """Load the training dataset, or None if DATASET_PATH doesn't exist."""
    if not os.path.exists(DATASET_PATH):
        return None
    return pd.read_csv(DATASET_PATH)


def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the same column drop + one-hot encoding used at training time."""
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
    """Turn a normalized prediction payload into a single-row DataFrame

    matching FEATURE_COLUMNS exactly (name, order, and one-hot encoding),
    so it can be passed straight into the saved Pipeline.
    """
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

    logger.debug('build_model_frame: input=%s columns_after_ohe=%s vector=%s',
                 payload, list(df.columns), result.values[0])

    return result


@app.get('/api/model-info')
def model_info() -> Dict[str, Any]:
    """Model metadata for the dashboard: features, accuracy, and whether a model file is present.

    Reports fixed accuracy/sample-count constants rather than scoring the
    model live, so the dashboard loads instantly — scoring an RBF SVM over
    every training row on each page visit would be far too slow for that.
    """
    model_available = os.path.exists(MODEL_PATH)
    return {
        'model_name': 'Loan approval prediction model',
        'features': FEATURE_COLUMNS,
        'num_features': len(FEATURE_COLUMNS),
        'accuracy': MODEL_ACCURACY,
        'num_samples': MODEL_SAMPLE_COUNT,
        'dataset_name': DATASET_NAME if os.path.exists(DATASET_PATH) else None,
        'model_type': MODEL_TYPE,
        'pipeline_status': 'Active' if model_available else 'Inactive',
    }


@lru_cache(maxsize=1)
def compute_model_analytics() -> Dict[str, Any]:
    """Real performance analytics computed from the saved model and its training data.

    Cached for the process lifetime: this evaluates the RBF SVM over the full
    dataset plus a PCA decision-surface mesh, which is too slow to redo per request.
    """
    from sklearn.model_selection import train_test_split

    model = load_model()
    scaler = model.named_steps['scaler']
    svc = model.named_steps['model']

    dataset = get_dataset()
    if dataset is None:
        raise FileNotFoundError('Dataset file is missing.')

    prepared = preprocess_dataset(dataset)
    X_df = prepared.drop(columns=['loan_status'])
    y = prepared['loan_status'].values
    X = X_df.values

    preds = model.predict(X)
    cm = confusion_matrix(y, preds).tolist()
    report = classification_report(y, preds, output_dict=True, zero_division=0)

    margins_full = svc.decision_function(scaler.transform(X))

    # Recreate the exact split used by /api/retrain-model so svc.support_
    # indices line up with a concrete set of rows we can plot and label.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    Xs_train = scaler.transform(X_train)

    pca = PCA(n_components=2, random_state=42)
    X2_train = pca.fit_transform(Xs_train)

    support_idx = set(int(i) for i in svc.support_)
    rng = np.random.default_rng(42)
    sample_idx = set(int(i) for i in rng.choice(len(X_train), size=min(220, len(X_train)), replace=False))
    sv_idx = sorted(support_idx)[:80]
    combined_idx = sorted(sample_idx | set(sv_idx))

    points = [
        {
            'x': round(float(X2_train[i, 0]), 4),
            'y': round(float(X2_train[i, 1]), 4),
            'label': int(y_train[i]),
            'is_support_vector': i in support_idx,
        }
        for i in combined_idx
    ]

    # Decision-boundary mesh: reconstruct grid points from PCA space back into
    # scaled feature space and evaluate the REAL trained SVM on them (not a
    # separately-fit 2D toy model), so the plot reflects actual model behavior.
    x_min, x_max = X2_train[:, 0].min() - 1, X2_train[:, 0].max() + 1
    y_min, y_max = X2_train[:, 1].min() - 1, X2_train[:, 1].max() + 1
    resolution = 50
    xx = np.linspace(x_min, x_max, resolution)
    yy = np.linspace(y_min, y_max, resolution)
    grid_pca = np.array([[gx, gy] for gy in yy for gx in xx])
    grid_scaled = pca.inverse_transform(grid_pca)
    z = svc.decision_function(grid_scaled).reshape(resolution, resolution)

    margin_min, margin_max = float(margins_full.min()), float(margins_full.max())
    counts, edges = np.histogram(margins_full, bins=24, range=(margin_min, margin_max))

    sample_rows = prepared.head(8).copy()
    dataset_samples = {
        'columns': list(sample_rows.columns),
        'rows': [
            [(bool(v) if isinstance(v, (bool, np.bool_)) else (int(v) if float(v).is_integer() else round(float(v), 4)))
             for v in row]
            for row in sample_rows.values
        ],
    }

    with open(MODEL_PATH, 'rb') as f:
        file_bytes = f.read()
    modified = datetime.fromtimestamp(os.path.getmtime(MODEL_PATH), tz=timezone.utc)

    def cls_row(label: int) -> Dict[str, Any]:
        row = report.get(str(label), {})
        return {
            'label': label,
            'precision': round(float(row.get('precision', 0)), 3),
            'recall': round(float(row.get('recall', 0)), 3),
            'f1': round(float(row.get('f1-score', 0)), 3),
            'support': int(row.get('support', 0)),
        }

    def avg_row(key: str) -> Dict[str, Any]:
        row = report[key]
        return {
            'precision': round(float(row['precision']), 3),
            'recall': round(float(row['recall']), 3),
            'f1': round(float(row['f1-score']), 3),
            'support': int(row['support']),
        }

    return {
        'model_details': {
            'algorithm': 'SVC (Support Vector Classifier)',
            'kernel': svc.kernel,
            'C': svc.C,
            'gamma': str(svc.gamma),
            'probability': bool(svc.probability),
            'class_weight': svc.class_weight,
            'decision_function_shape': svc.decision_function_shape,
            'classes': [int(c) for c in svc.classes_],
            'n_support': [int(n) for n in svc.n_support_],
            'n_features': int(svc.n_features_in_),
        },
        'model_file': {
            'filename': os.path.basename(MODEL_PATH),
            'size_kb': round(len(file_bytes) / 1024, 2),
            'sha256': hashlib.sha256(file_bytes).hexdigest(),
            'last_modified': modified.strftime('%Y-%m-%d'),
        },
        'dataset': {
            'name': DATASET_NAME,
            'num_samples': int(len(dataset)),
            'train_samples': int(len(X_train)),
            'test_samples': int(len(X_test)),
        },
        'confusion_matrix': {
            'labels': [int(c) for c in svc.classes_],
            'matrix': cm,
        },
        'classification_report': {
            'classes': [cls_row(0), cls_row(1)],
            'accuracy': round(float(report['accuracy']), 3),
            'macro_avg': avg_row('macro avg'),
            'weighted_avg': avg_row('weighted avg'),
        },
        'decision_boundary': {
            'explained_variance': [round(float(v), 4) for v in pca.explained_variance_ratio_],
            'grid': {
                'x': [round(float(v), 4) for v in xx],
                'y': [round(float(v), 4) for v in yy],
                'z': [[round(float(v), 4) for v in row] for row in z],
            },
            'points': points,
        },
        'margin_distribution': {
            'bin_edges': [round(float(v), 4) for v in edges],
            'counts': [int(v) for v in counts],
        },
        'dataset_samples': dataset_samples,
        'endpoints': [
            {'method': 'GET', 'path': '/api/health', 'description': 'Service health check'},
            {'method': 'GET', 'path': '/api/model-info', 'description': 'Model metadata and factor list'},
            {'method': 'POST', 'path': '/api/predict', 'description': 'Run a loan approval prediction'},
            {'method': 'GET', 'path': '/api/model-analytics', 'description': 'Full model performance analytics'},
        ],
    }


@app.get('/api/model-analytics')
def model_analytics() -> Dict[str, Any]:
    """Full model performance analytics: see compute_model_analytics() for the payload shape."""
    try:
        return compute_model_analytics()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail='Model or dataset file is missing.') from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f'Analytics computation failed: {exc}') from exc


@app.post('/api/predict')
def predict(payload: LoanPredictionInput) -> Dict[str, Any]:
    """Run the saved Pipeline on one application and return the approval decision."""
    try:
        model = load_model()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail='Model file is missing.') from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail='Unable to load model.') from exc

    feature_frame = build_model_frame(payload.normalized_payload)
    logger.debug('predict: payload=%s vector=%s', payload.normalized_payload, feature_frame.values[0])

    try:
        prediction = int(model.predict(feature_frame)[0])
        logger.info('predict: prediction=%s', prediction)

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
    """Liveness check used by uptime monitors and the deploy platform."""
    return {'status': 'ok'}


@app.post('/api/retrain-model')
def retrain_model() -> Dict[str, Any]:
    """Retrain the Pipeline from scratch on DATASET_PATH and overwrite MODEL_PATH.

    Mirrors the training notebook exactly: StandardScaler + SVC(kernel='rbf')
    in one Pipeline, with a small grid search over C via 5-fold CV, then a
    held-out test-set evaluation. This is the real retraining mechanism used
    whenever the training dataset changes (e.g. loan_data_v2.csv).
    """
    try:
        from sklearn.model_selection import cross_val_score
        from sklearn.preprocessing import StandardScaler as ScalerClass
        from sklearn.pipeline import Pipeline as PipelineClass

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
        load_model.cache_clear()

        logger.info('retrain_model: best_C=%s cv_accuracy=%.4f test_accuracy=%.4f full_accuracy=%.4f',
                    best_C, best_score, test_acc, full_acc)
        compute_model_analytics.cache_clear()

        return {
            'status': 'success',
            'best_C': best_C,
            'cv_accuracy': float(best_score),
            'test_accuracy': float(test_acc),
            'full_accuracy': float(full_acc),
            'message': 'Model retrained and saved successfully'
        }
    except Exception as e:
        logger.warning('retrain_model failed: %s', e)
        raise HTTPException(status_code=500, detail=f'Retraining failed: {str(e)}')


@app.get('/')
def root() -> FileResponse:
    """Serve the single-page app shell; static/app.js handles client-side routing."""
    return FileResponse(os.path.join(STATIC_DIR, 'index.html'))


app.mount('/static', StaticFiles(directory=STATIC_DIR), name='static')


if __name__ == '__main__':
    import uvicorn

    uvicorn.run('app:app', host='0.0.0.0', port=8000, reload=False)
