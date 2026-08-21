const TEXT = {
  he: {
    pageTitle: 'חיזוי הלוואות', brandName: 'חיזוי הלוואות', modelLegend: 'מקרא המודל', loanPrediction: 'חיזוי הלוואה', sidebarNote: 'המערכת נעזרת במודל למידת מכונה כדי להעריך את סיכויי אישור ההלוואה.',
    overview: 'תמונת מצב', modelIntro: 'פרטי המודל והגורמים המשמשים לחיזוי אישור ההלוואה.', predictionFactors: 'גורמים בחיזוי ההלוואה', formIntro: 'הזינו את פרטי הבקשה לקבלת הערכת אישור.',
    resetForm: 'ניקוי הטופס', personalDataGroup: 'נתונים אישיים', loanDataGroup: 'נתוני ההלוואה', creditHistoryGroup: 'היסטוריית אשראי',
    annualIncome: 'הכנסה שנתית', yearsExperience: 'שנות ניסיון', loanAmount: 'סכום ההלוואה', interestRate: 'שיעור הריבית', loanIncomeRatio: 'יחס הלוואה להכנסה',
    automaticallyCalculated: '(מחושב אוטומטית)', ratioHint: 'סכום ההלוואה ÷ הכנסה שנתית', creditScore: 'ציון אשראי', previousDefault: 'פיגור בהלוואה קודמת', chooseOption: 'בחרו אפשרות', yes: 'כן', no: 'לא',
    incomePlaceholder: 'לדוגמה: 85000', experiencePlaceholder: 'לדוגמה: 5', loanPlaceholder: 'לדוגמה: 35000', ratePlaceholder: 'לדוגמה: 12.5', ratioPlaceholder: 'יחושב לאחר הזנת הכנסה וסכום הלוואה', creditPlaceholder: 'לדוגמה: 620',
    active: 'מערכת פעילה', unavailable: 'המערכת אינה זמינה', modelName: 'שם המודל', modelNameValue: 'מודל לחיזוי אישור הלוואות', modelNameDescription: 'מודל חיזוי להערכת אישור בקשות',
    accuracy: 'דיוק המודל', accuracyDescription: 'דיוק על נתוני האימון', sampleCount: 'מספר דוגמאות', sampleDescription: 'בקשות הלוואה בנתוני ההדרכה', criteriaCount: 'מספר קריטריונים', criteriaDescription: 'גורמים המשפיעים על החלטת האישור',
    modelType: 'סוג המודל', status: 'סטטוס', ready: 'פעיל', readyDescription: 'המודל מוכן לחיזוי', validValue: 'יש להזין ערך תקין.', validNumber: 'יש להזין מספר תקין.',
    atLeast: 'הערך חייב להיות לפחות', atMost: 'הערך יכול להיות לכל היותר', cannotComplete: 'לא ניתן להשלים את החיזוי', error: 'שגיאה', fixFields: 'יש לתקן את השדות המסומנים לפני הפעלת החיזוי.',
    calculating: 'מחשב חיזוי...', runPrediction: 'הפעלת חיזוי הלוואה', serverError: 'השרת החזיר שגיאה.', approvedTitle: 'ההלוואה צפויה להיות מאושרת', rejectedTitle: 'ההלוואה אינה צפויה להיות מאושרת',
    approved: 'מאושר', rejected: 'לא אושר', completed: 'החיזוי הושלם בהצלחה.', predictionResult: 'תוצאת החיזוי', approvalChance: 'סיכוי לאישור',
    feature: { person_income: ['הכנסה שנתית', 'סך ההכנסה השנתית של מבקש ההלוואה.'], person_emp_exp: ['שנות ניסיון', 'מספר שנות הניסיון התעסוקתי.'], loan_amnt: ['סכום ההלוואה', 'הסכום המבוקש במסגרת ההלוואה.'], loan_int_rate: ['שיעור הריבית', 'שיעור הריבית השנתית על ההלוואה.'], loan_percent_income: ['יחס הלוואה להכנסה', 'היחס המחושב בין סכום ההלוואה להכנסה השנתית.'], credit_score: ['ציון אשראי', 'ציון המשקף את היסטוריית האשראי.'], previous_loan_defaults_on_file_Yes: ['פיגור בהלוואה קודמת', 'האם נרשם פיגור בתשלום הלוואה בעבר.'] }
  },
  en: {
    pageTitle: 'Loan Prediction', brandName: 'Loan Prediction', modelLegend: 'Model Overview', loanPrediction: 'Loan Prediction', sidebarNote: 'The system uses a machine-learning model to estimate the likelihood of loan approval.',
    overview: 'Overview', modelIntro: 'Model details and the factors used to forecast loan approval.', predictionFactors: 'Loan prediction factors', formIntro: 'Enter the application details to receive an approval estimate.',
    resetForm: 'Reset form', personalDataGroup: 'Personal details', loanDataGroup: 'Loan details', creditHistoryGroup: 'Credit history',
    annualIncome: 'Annual income', yearsExperience: 'Years of experience', loanAmount: 'Loan amount', interestRate: 'Interest rate', loanIncomeRatio: 'Loan-to-income ratio',
    automaticallyCalculated: '(calculated automatically)', ratioHint: 'Loan amount ÷ annual income', creditScore: 'Credit score', previousDefault: 'Previous loan default', chooseOption: 'Choose an option', yes: 'Yes', no: 'No',
    incomePlaceholder: 'Example: 85000', experiencePlaceholder: 'Example: 5', loanPlaceholder: 'Example: 35000', ratePlaceholder: 'Example: 12.5', ratioPlaceholder: 'Calculated after entering income and loan amount', creditPlaceholder: 'Example: 620',
    active: 'System active', unavailable: 'System unavailable', modelName: 'Model name', modelNameValue: 'Loan approval prediction model', modelNameDescription: 'A model for evaluating loan applications',
    accuracy: 'Model accuracy', accuracyDescription: 'Accuracy on training data', sampleCount: 'Sample count', sampleDescription: 'Loan applications in the training data', criteriaCount: 'Criteria count', criteriaDescription: 'Factors affecting the approval decision',
    modelType: 'Model type', status: 'Status', ready: 'Active', readyDescription: 'The model is ready for prediction', validValue: 'Enter a valid value.', validNumber: 'Enter a valid number.',
    atLeast: 'Value must be at least', atMost: 'Value can be at most', cannotComplete: 'Unable to complete the prediction', error: 'Error', fixFields: 'Correct the marked fields before running the prediction.',
    calculating: 'Calculating prediction...', runPrediction: 'Run loan prediction', serverError: 'The server returned an error.', approvedTitle: 'The loan is expected to be approved', rejectedTitle: 'The loan is not expected to be approved',
    approved: 'Approved', rejected: 'Rejected', completed: 'Prediction completed successfully.', predictionResult: 'Prediction result', approvalChance: 'Approval chance',
    feature: { person_income: ['Annual income', 'The applicant’s total annual income.'], person_emp_exp: ['Years of experience', 'Number of years of employment experience.'], loan_amnt: ['Loan amount', 'The amount requested in the loan application.'], loan_int_rate: ['Interest rate', 'The annual interest rate for the loan.'], loan_percent_income: ['Loan-to-income ratio', 'The calculated ratio of loan amount to annual income.'], credit_score: ['Credit score', 'A score reflecting the applicant’s credit history.'], previous_loan_defaults_on_file_Yes: ['Previous loan default', 'Whether there was a past loan-payment default.'] }
  }
};

const ICONS = { person_income: 'wallet', person_emp_exp: 'briefcase', loan_amnt: 'hand-coins', loan_int_rate: 'percent', loan_percent_income: 'pie-chart', credit_score: 'gauge', previous_loan_defaults_on_file_Yes: 'alert-triangle' };
const STAT_ICONS = { modelName: 'cpu', accuracy: 'target', sampleCount: 'database', criteriaCount: 'list-checks', modelType: 'layers', status: 'circle-check-big' };
let language = localStorage.getItem('loan-language') || 'he';
let modelInfo = {};
let features = Object.keys(ICONS);
const t = (key) => TEXT[language][key] ?? key;

const apiStatus = document.getElementById('api-status');
const statsGrid = document.getElementById('stats-grid');
const featuresList = document.getElementById('features-list');
const form = document.getElementById('prediction-form');
const submitBtn = document.getElementById('submit-btn');
const btnLabel = submitBtn.querySelector('.btn-label');
const spinner = submitBtn.querySelector('.spinner');
const resultCard = document.getElementById('result-card');
const resetBtn = document.getElementById('reset-form');
const incomeInput = document.getElementById('person_income');
const loanAmountInput = document.getElementById('loan_amnt');
const loanIncomeRatioInput = document.getElementById('loan_percent_income');
const navButtons = document.querySelectorAll('.nav-item');
const views = { dashboard: document.getElementById('dashboard'), predictor: document.getElementById('predictor') };

function refreshIcons() { if (window.lucide) window.lucide.createIcons(); }
function setView(target) { navButtons.forEach((button) => button.classList.toggle('active', button.dataset.target === target)); Object.entries(views).forEach(([key, view]) => view.classList.toggle('active', key === target)); }
function setApiStatus(type, iconName, message) { apiStatus.className = `status-pill ${type}`; apiStatus.innerHTML = `<i data-lucide="${iconName}"></i><span>${message}</span>`; refreshIcons(); }
function updateLoanIncomeRatio() { const income = Number(incomeInput.value); const loanAmount = Number(loanAmountInput.value); loanIncomeRatioInput.value = income > 0 && loanAmount > 0 ? (loanAmount / income).toFixed(4) : ''; }

function renderStatSkeleton() { statsGrid.innerHTML = Array.from({ length: 6 }).map(() => '<article class="stat-card skeleton-card"><div class="skeleton-line short"></div><div class="skeleton-line tall"></div><div class="skeleton-line"></div></article>').join(''); }
function renderFeatureSkeleton() { featuresList.innerHTML = Array.from({ length: 7 }).map(() => '<article class="feature-item skeleton-card"><div class="skeleton-icon"></div><div class="skeleton-text"><div class="skeleton-line short"></div><div class="skeleton-line"></div></div></article>').join(''); }

function renderStatCards(data) {
  const cards = [
    ['modelName', 'modelName', t('modelNameValue'), 'modelNameDescription'], ['accuracy', 'accuracy', data.accuracy != null ? `${(Number(data.accuracy) * 100).toFixed(1)}%` : '—', 'accuracyDescription'],
    ['sampleCount', 'sampleCount', data.num_samples ?? '—', 'sampleDescription'], ['criteriaCount', 'criteriaCount', data.num_features ?? 7, 'criteriaDescription'],
    ['modelType', 'modelType', 'SVM', 'Support Vector Machine'], ['status', 'status', t('ready'), 'readyDescription'],
  ];
  statsGrid.innerHTML = cards.map(([icon, label, value, description]) => `<article class="stat-card"><div class="stat-card-top"><span class="stat-icon"><i data-lucide="${STAT_ICONS[icon]}"></i></span><span class="stat-label">${t(label)}</span></div><div class="stat-value">${value}</div><div class="stat-desc">${t(description)}</div></article>`).join('');
  refreshIcons();
}
function renderFeatures(list) {
  featuresList.innerHTML = list.map((key) => { const content = TEXT[language].feature[key] || [key, '']; return `<article class="feature-item"><span class="feature-icon"><i data-lucide="${ICONS[key] || 'circle'}"></i></span><div><div class="feature-name">${content[0]}</div><div class="feature-desc">${content[1]}</div></div></article>`; }).join('');
  refreshIcons();
}
function applyLanguage(nextLanguage) {
  language = nextLanguage; localStorage.setItem('loan-language', language);
  document.documentElement.lang = language; document.documentElement.dir = language === 'he' ? 'rtl' : 'ltr'; document.title = t('pageTitle');
  document.querySelectorAll('[data-i18n]').forEach((element) => { element.textContent = t(element.dataset.i18n); });
  document.querySelectorAll('[data-i18n-placeholder]').forEach((element) => { element.placeholder = t(element.dataset.i18nPlaceholder); });
  document.querySelectorAll('.language-btn').forEach((button) => button.classList.toggle('active', button.dataset.lang === language));
  renderStatCards(modelInfo); renderFeatures(features); setApiStatus('success', 'circle-check', t('active'));
}

async function fetchModelInfo() {
  renderStatSkeleton(); renderFeatureSkeleton(); setApiStatus('loading', 'loader-2', t('calculating'));
  try { const response = await fetch('/api/model-info'); if (!response.ok) throw new Error(); modelInfo = await response.json(); features = modelInfo.features || features; renderStatCards(modelInfo); renderFeatures(features); setApiStatus('success', 'circle-check', t('active')); }
  catch (error) { console.error(error); renderStatCards({}); renderFeatures(features); setApiStatus('error', 'circle-alert', t('unavailable')); }
}
function markInvalid(field, message) { field.classList.add('invalid'); let error = field.parentElement.querySelector('.field-error'); if (!error) { error = document.createElement('div'); error.className = 'field-error'; field.parentElement.appendChild(error); } error.textContent = message; }
function clearInvalid(field) { field.classList.remove('invalid'); const error = field.parentElement.querySelector('.field-error'); if (error) error.textContent = ''; }
function validateForm() {
  let valid = true; form.querySelectorAll('input, select').forEach((field) => { const value = field.value.trim(); if (value === '') { valid = false; markInvalid(field, t('validValue')); return; } if (field.type === 'number' && Number.isNaN(Number(value))) { valid = false; markInvalid(field, t('validNumber')); return; } if (field.min && Number(value) < Number(field.min)) { valid = false; markInvalid(field, `${t('atLeast')} ${field.min}.`); return; } if (field.max && Number(value) > Number(field.max)) { valid = false; markInvalid(field, `${t('atMost')} ${field.max}.`); return; } clearInvalid(field); }); return valid;
}
function renderErrorState(message) {
  resultCard.className = 'result-card hidden'; resultCard.innerHTML = `<div class="result-header"><div class="result-title"><i data-lucide="triangle-alert" class="result-icon"></i><h4>${t('cannotComplete')}</h4></div><span class="badge error">${t('error')}</span></div><p class="result-message">${message}</p>`;
  requestAnimationFrame(() => { resultCard.className = 'result-card fade-in error'; refreshIcons(); });
}
function gaugeColorClass(pct) { if (pct >= 66) return 'var(--teal)'; if (pct >= 33) return 'var(--warning)'; return 'var(--danger)'; }
function renderGauge(probability) {
  const pct = Math.round(Number(probability) * 100);
  return `<div class="gauge" style="--pct:${pct}; --gauge-color:${gaugeColorClass(pct)};" role="img" aria-label="${t('approvalChance')} ${pct}%"><div class="gauge-center"><span class="gauge-value">${pct}%</span><span class="gauge-label">${t('approvalChance')}</span></div></div>`;
}
function renderPredictionResult(data) {
  const approved = Boolean(data.approved);
  const gaugeHtml = data.probability != null ? renderGauge(data.probability) : '';
  resultCard.className = 'result-card hidden';
  resultCard.innerHTML = `
    <div class="result-header"><div class="result-title"><i data-lucide="${approved ? 'circle-check-big' : 'circle-x'}" class="result-icon"></i><h4>${approved ? t('approvedTitle') : t('rejectedTitle')}</h4></div><span class="badge ${approved ? 'success' : 'error'}">${approved ? t('approved') : t('rejected')}</span></div>
    <p class="result-message">${t('completed')}</p>
    <div class="result-body">
      ${gaugeHtml}
      <div class="result-row"><span>${t('predictionResult')}</span><strong>${data.prediction}</strong></div>
    </div>`;
  requestAnimationFrame(() => { resultCard.className = `result-card fade-in ${approved ? 'success' : 'error'}`; refreshIcons(); });
}
async function submitPrediction(event) {
  event.preventDefault(); updateLoanIncomeRatio(); if (!validateForm()) { renderErrorState(t('fixFields')); return; }
  const payload = Object.fromEntries(new FormData(form)); ['person_income', 'person_emp_exp', 'loan_amnt', 'loan_int_rate', 'loan_percent_income', 'credit_score'].forEach((key) => { payload[key] = Number(payload[key]); });
  submitBtn.disabled = true; btnLabel.textContent = t('calculating'); spinner.classList.remove('hidden');
  try { const response = await fetch('/api/predict', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) }); const data = await response.json(); if (!response.ok) throw new Error(data.detail || data.error || t('serverError')); renderPredictionResult(data); }
  catch (error) { renderErrorState(error.message || t('serverError')); }
  finally { submitBtn.disabled = false; btnLabel.textContent = t('runPrediction'); spinner.classList.add('hidden'); }
}
function resetForm() { form.reset(); resultCard.className = 'result-card hidden'; resultCard.innerHTML = ''; form.querySelectorAll('input, select').forEach(clearInvalid); }
navButtons.forEach((button) => button.addEventListener('click', () => setView(button.dataset.target)));
document.querySelectorAll('.language-btn').forEach((button) => button.addEventListener('click', () => applyLanguage(button.dataset.lang)));
incomeInput.addEventListener('input', updateLoanIncomeRatio); loanAmountInput.addEventListener('input', updateLoanIncomeRatio); form.addEventListener('submit', submitPrediction); resetBtn.addEventListener('click', resetForm);
applyLanguage(language); fetchModelInfo(); setView('dashboard'); refreshIcons();
