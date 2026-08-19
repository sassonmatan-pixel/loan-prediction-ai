const FEATURE_DESCRIPTIONS = {
  person_income: { name: 'הכנסה שנתית', description: 'סך ההכנסה השנתית של מבקש ההלוואה.', icon: '₪' },
  person_emp_exp: { name: 'שנות ניסיון', description: 'מספר שנות הניסיון התעסוקתי.', icon: '◷' },
  loan_amnt: { name: 'סכום ההלוואה', description: 'הסכום המבוקש במסגרת ההלוואה.', icon: '▣' },
  loan_int_rate: { name: 'שיעור הריבית', description: 'שיעור הריבית השנתית על ההלוואה.', icon: '%' },
  loan_percent_income: { name: 'יחס ההחזר להכנסה', description: 'החלק מההכנסה שמוקדש להחזר ההלוואה.', icon: '◔' },
  credit_score: { name: 'ציון אשראי', description: 'ציון המשקף את היסטוריית האשראי.', icon: '★' },
  previous_loan_defaults_on_file_Yes: { name: 'פיגור בהלוואה קודמת', description: 'האם נרשם פיגור בתשלום הלוואה בעבר.', icon: '!' },
};

const apiStatus = document.getElementById('api-status');
const statsGrid = document.getElementById('stats-grid');
const featuresList = document.getElementById('features-list');
const form = document.getElementById('prediction-form');
const submitBtn = document.getElementById('submit-btn');
const btnLabel = submitBtn.querySelector('.btn-label');
const spinner = submitBtn.querySelector('.spinner');
const resultCard = document.getElementById('result-card');
const resetBtn = document.getElementById('reset-form');
const navButtons = document.querySelectorAll('.nav-item');
const views = { dashboard: document.getElementById('dashboard'), predictor: document.getElementById('predictor') };

function setView(target) {
  navButtons.forEach((button) => button.classList.toggle('active', button.dataset.target === target));
  Object.entries(views).forEach(([key, view]) => view.classList.toggle('active', key === target));
}

function setApiStatus(type, message) {
  apiStatus.classList.remove('success', 'error');
  apiStatus.classList.add(type);
  apiStatus.textContent = message;
}

function renderStatCards(data) {
  const cards = [
    { icon: '◈', label: 'שם המודל', value: 'מודל לחיזוי אישור הלוואות', description: 'מודל חיזוי להערכת אישור בקשות' },
    { icon: '◌', label: 'דיוק המודל', value: data.accuracy != null ? `${(Number(data.accuracy) * 100).toFixed(1)}%` : '—', description: 'דיוק על נתוני האימון' },
    { icon: '▦', label: 'מספר דוגמאות', value: data.num_samples ?? '—', description: 'בקשות הלוואה בנתוני ההדרכה' },
    { icon: '✦', label: 'מספר קריטריונים', value: data.num_features ?? 7, description: 'גורמים המשפיעים על החלטת האישור' },
    { icon: '⌘', label: 'סוג המודל', value: 'SVM', description: 'Support Vector Machine' },
    { icon: '●', label: 'סטטוס', value: 'פעיל', description: 'המודל מוכן לחיזוי' },
  ];
  statsGrid.innerHTML = cards.map((card) => `<article class="stat-card"><div class="stat-card-top"><span class="stat-icon">${card.icon}</span><span class="stat-label">${card.label}</span></div><div class="stat-value">${card.value}</div><div class="stat-desc">${card.description}</div></article>`).join('');
}

function renderFeatures(features) {
  featuresList.innerHTML = features.map((key) => {
    const feature = FEATURE_DESCRIPTIONS[key] || { name: key, description: 'גורם המשמש את המודל בחיזוי.', icon: '•' };
    return `<article class="feature-item"><span class="feature-icon">${feature.icon}</span><div><div class="feature-name">${feature.name}</div><div class="feature-desc">${feature.description}</div></div></article>`;
  }).join('');
}

async function fetchModelInfo() {
  try {
    const response = await fetch('/api/model-info');
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();
    renderStatCards(data);
    renderFeatures(data.features || []);
    setApiStatus('success', '● מערכת פעילה');
  } catch (error) {
    console.error(error);
    renderStatCards({});
    renderFeatures(Object.keys(FEATURE_DESCRIPTIONS));
    setApiStatus('error', '● המערכת אינה זמינה');
  }
}

function markInvalid(field, message) {
  field.classList.add('invalid');
  let error = field.parentElement.querySelector('.field-error');
  if (!error) { error = document.createElement('div'); error.className = 'field-error'; field.parentElement.appendChild(error); }
  error.textContent = message;
}
function clearInvalid(field) {
  field.classList.remove('invalid');
  const error = field.parentElement.querySelector('.field-error');
  if (error) error.textContent = '';
}
function validateForm() {
  let valid = true;
  form.querySelectorAll('input, select').forEach((field) => {
    const value = field.value.trim();
    if (value === '') { valid = false; markInvalid(field, 'יש להזין ערך תקין.'); return; }
    if (field.type === 'number' && Number.isNaN(Number(value))) { valid = false; markInvalid(field, 'יש להזין מספר תקין.'); return; }
    if (field.min && Number(value) < Number(field.min)) { valid = false; markInvalid(field, `הערך חייב להיות לפחות ${field.min}.`); return; }
    if (field.max && Number(value) > Number(field.max)) { valid = false; markInvalid(field, `הערך יכול להיות לכל היותר ${field.max}.`); return; }
    clearInvalid(field);
  });
  return valid;
}

function renderErrorState(message) {
  resultCard.className = 'result-card error';
  resultCard.innerHTML = `<div class="result-header"><h4>לא ניתן להשלים את החיזוי</h4><span class="badge error">שגיאה</span></div><p>${message}</p>`;
}
function renderPredictionResult(data) {
  const approved = Boolean(data.approved);
  const probability = data.probability != null ? `<span>סיכוי לאישור</span><strong>${(Number(data.probability) * 100).toFixed(1)}%</strong>` : '';
  resultCard.className = `result-card ${approved ? 'success' : 'error'}`;
  resultCard.innerHTML = `<div class="result-header"><h4>${approved ? 'ההלוואה צפויה להיות מאושרת' : 'ההלוואה אינה צפויה להיות מאושרת'}</h4><span class="badge ${approved ? 'success' : 'error'}">${approved ? 'מאושר' : 'לא אושר'}</span></div><p>${data.message || 'החיזוי הושלם בהצלחה.'}</p><div class="result-row"><span>תוצאת החיזוי</span><strong>${data.prediction}</strong>${probability}</div>`;
}

async function submitPrediction(event) {
  event.preventDefault();
  if (!validateForm()) { renderErrorState('יש לתקן את השדות המסומנים לפני הפעלת החיזוי.'); return; }
  const payload = Object.fromEntries(new FormData(form));
  ['person_income', 'person_emp_exp', 'loan_amnt', 'loan_int_rate', 'loan_percent_income', 'credit_score'].forEach((key) => { payload[key] = Number(payload[key]); });
  submitBtn.disabled = true; btnLabel.textContent = 'מחשב חיזוי...'; spinner.classList.remove('hidden');
  try {
    const response = await fetch('/api/predict', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || data.error || 'השרת החזיר שגיאה.');
    renderPredictionResult(data);
  } catch (error) { renderErrorState(error.message || 'אירעה שגיאה במהלך החיזוי.'); }
  finally { submitBtn.disabled = false; btnLabel.textContent = '✦ הפעלת חיזוי הלוואה'; spinner.classList.add('hidden'); }
}

function resetForm() { form.reset(); resultCard.className = 'result-card hidden'; form.querySelectorAll('input, select').forEach(clearInvalid); }
navButtons.forEach((button) => button.addEventListener('click', () => setView(button.dataset.target)));
form.addEventListener('submit', submitPrediction);
resetBtn.addEventListener('click', resetForm);
fetchModelInfo();
setView('dashboard');
