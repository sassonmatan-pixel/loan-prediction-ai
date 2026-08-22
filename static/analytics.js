let analyticsData = null;
let analyticsLoaded = false;
let analyticsLoading = false;

const analyticsLoadingEl = document.getElementById('analytics-loading');
const analyticsErrorEl = document.getElementById('analytics-error');
const analyticsContentEl = document.getElementById('analytics-content');

const PALETTE = { approve: [13, 148, 136], reject: [192, 70, 58], axis: '#5b6b85', grid: 'rgba(15,35,64,0.18)' };

function detailRow(icon, label, value) {
  return `<div class="detail-row"><span class="detail-icon"><i data-lucide="${icon}"></i></span><span class="detail-label">${label}</span><span class="detail-value">${value}</span></div>`;
}

function classLabelName(label) { return label === 1 ? t('approvedClass') : t('rejectedClass'); }

function renderModelDetails(d) {
  const md = d.model_details;
  const rows = [
    ['cpu', t('algorithm'), md.algorithm],
    ['waves', t('kernelLabel'), md.kernel.toUpperCase()],
    ['sliders-horizontal', t('regularization'), md.C],
    ['sigma', t('gammaLabel'), md.gamma],
    ['percent', t('probabilityLabel'), md.probability ? t('yes') : t('no')],
    ['scale', t('classWeightLabel'), md.class_weight ?? '—'],
    ['git-branch', t('decisionShapeLabel'), md.decision_function_shape.toUpperCase()],
    ['tags', t('classesLabel'), md.classes.join(' / ')],
    ['anchor', t('supportVectorsLabel'), md.n_support.join(' / ')],
    ['list-checks', t('featuresCountLabel'), md.n_features],
  ];
  document.getElementById('model-details-list').innerHTML = rows.map((r) => detailRow(...r)).join('');
}

function renderModelFile(d) {
  const mf = d.model_file;
  const ds = d.dataset;
  const rows = [
    ['file-text', t('filenameLabel'), mf.filename],
    ['hard-drive', t('fileSizeLabel'), `${mf.size_kb} KB`],
    ['fingerprint', t('checksumLabel'), `<span class="mono-truncate" title="${mf.sha256}">${mf.sha256.slice(0, 16)}…</span>`],
    ['calendar', t('lastModifiedLabel'), mf.last_modified],
    ['database', t('datasetLabel'), `${ds.name} (${ds.num_samples})`],
    ['split', t('trainSamplesLabel'), ds.train_samples],
    ['flask-conical', t('testSamplesLabel'), ds.test_samples],
  ];
  document.getElementById('model-file-list').innerHTML = rows.map((r) => detailRow(...r)).join('');
}

function renderEndpoints(d) {
  document.getElementById('endpoints-list').innerHTML = d.endpoints.map((ep) => `
    <div class="endpoint-row">
      <span class="method-pill method-${ep.method.toLowerCase()}">${ep.method}</span>
      <div class="endpoint-info"><code class="endpoint-path">${ep.path}</code><span class="endpoint-desc">${ep.description}</span></div>
    </div>`).join('');
}

function renderConfusion(d) {
  const cm = d.confusion_matrix;
  let html = `<div class="table-scroll"><table class="analytics-table"><thead><tr><th></th>${cm.labels.map((l) => `<th>${t('predictedLabel')}<br>${classLabelName(l)}</th>`).join('')}<th>${t('supportLabel')}</th></tr></thead><tbody>`;
  cm.labels.forEach((rowLabel, i) => {
    const rowSum = cm.matrix[i].reduce((a, b) => a + b, 0);
    html += `<tr><th>${t('actualLabel')}<br>${classLabelName(rowLabel)}</th>`;
    cm.matrix[i].forEach((val, j) => {
      html += `<td class="${i === j ? 'cell-correct' : 'cell-wrong'}">${val}</td>`;
    });
    html += `<td class="cell-total">${rowSum}</td></tr>`;
  });
  html += '</tbody></table></div>';
  document.getElementById('confusion-table').innerHTML = html;
}

function renderReport(d) {
  const r = d.classification_report;
  let html = `<div class="table-scroll"><table class="analytics-table"><thead><tr><th>${t('classLabel')}</th><th>${t('precisionLabel')}</th><th>${t('recallLabel')}</th><th>${t('f1Label')}</th><th>${t('supportLabel')}</th></tr></thead><tbody>`;
  r.classes.forEach((c) => {
    html += `<tr><td>${classLabelName(c.label)}</td><td>${c.precision}</td><td>${c.recall}</td><td>${c.f1}</td><td>${c.support}</td></tr>`;
  });
  html += `<tr class="row-summary"><td>${t('accuracyLabel')}</td><td colspan="3">${r.accuracy}</td><td>${r.macro_avg.support}</td></tr>`;
  html += `<tr class="row-summary"><td>${t('macroAvgLabel')}</td><td>${r.macro_avg.precision}</td><td>${r.macro_avg.recall}</td><td>${r.macro_avg.f1}</td><td>${r.macro_avg.support}</td></tr>`;
  html += `<tr class="row-summary"><td>${t('weightedAvgLabel')}</td><td>${r.weighted_avg.precision}</td><td>${r.weighted_avg.recall}</td><td>${r.weighted_avg.f1}</td><td>${r.weighted_avg.support}</td></tr>`;
  html += '</tbody></table></div>';
  document.getElementById('report-table').innerHTML = html;
}

function setupCanvas(canvas, cssWidth, cssHeight) {
  const dpr = window.devicePixelRatio || 1;
  canvas.style.width = `${cssWidth}px`;
  canvas.style.height = `${cssHeight}px`;
  canvas.width = Math.max(1, Math.round(cssWidth * dpr));
  canvas.height = Math.max(1, Math.round(cssHeight * dpr));
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return ctx;
}

function renderBoundary(d) {
  const db = d.decision_boundary;
  const wrap = document.querySelector('#card-boundary .chart-wrap');
  const canvas = document.getElementById('boundary-canvas');
  const width = Math.max(280, wrap.clientWidth);
  const height = 380;
  const ctx = setupCanvas(canvas, width, height);
  ctx.clearRect(0, 0, width, height);

  const pad = { left: 20, right: 20, top: 16, bottom: 40 };
  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;

  const gx = db.grid.x;
  const gy = db.grid.y;
  const z = db.grid.z;
  const xMin = gx[0];
  const xMax = gx[gx.length - 1];
  const yMin = gy[0];
  const yMax = gy[gy.length - 1];
  const sx = (vx) => pad.left + ((vx - xMin) / (xMax - xMin)) * plotW;
  const sy = (vy) => pad.top + (1 - (vy - yMin) / (yMax - yMin)) * plotH;

  let maxAbs = 0.0001;
  z.forEach((row) => row.forEach((v) => { maxAbs = Math.max(maxAbs, Math.abs(v)); }));
  const cellW = plotW / gx.length;
  const cellH = plotH / gy.length;
  for (let i = 0; i < gy.length; i += 1) {
    for (let j = 0; j < gx.length; j += 1) {
      const v = z[i][j];
      const strength = Math.min(1, Math.abs(v) / maxAbs);
      const color = v >= 0 ? PALETTE.approve : PALETTE.reject;
      const alpha = 0.08 + strength * 0.3;
      ctx.fillStyle = `rgba(${color[0]},${color[1]},${color[2]},${alpha})`;
      const cx = pad.left + j * cellW;
      const cy = pad.top + (gy.length - 1 - i) * cellH;
      ctx.fillRect(cx - 0.5, cy - 0.5, cellW + 1, cellH + 1);
    }
  }

  ctx.strokeStyle = PALETTE.grid;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.rect(pad.left, pad.top, plotW, plotH);
  ctx.stroke();

  db.points.forEach((p) => {
    const px = sx(p.x);
    const py = sy(p.y);
    const color = p.label === 1 ? PALETTE.approve : PALETTE.reject;
    ctx.beginPath();
    ctx.arc(px, py, p.is_support_vector ? 4.5 : 3, 0, Math.PI * 2);
    ctx.fillStyle = `rgba(${color[0]},${color[1]},${color[2]},0.85)`;
    ctx.fill();
    if (p.is_support_vector) {
      ctx.lineWidth = 1.3;
      ctx.strokeStyle = '#101a2c';
      ctx.stroke();
    }
  });

  ctx.fillStyle = PALETTE.axis;
  ctx.font = '11px Heebo, sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText('PC1', pad.left + plotW / 2, height - 10);
  ctx.save();
  ctx.translate(12, pad.top + plotH / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText('PC2', 0, 0);
  ctx.restore();

  const pct = Math.round((db.explained_variance[0] + db.explained_variance[1]) * 100);
  document.getElementById('boundary-subtitle').textContent = t('boundarySubtitle').replace('{pct}', pct);
  document.getElementById('boundary-legend').innerHTML = `
    <span class="legend-item"><span class="legend-dot" style="background:rgb(${PALETTE.approve.join(',')})"></span>${t('approvedClass')}</span>
    <span class="legend-item"><span class="legend-dot" style="background:rgb(${PALETTE.reject.join(',')})"></span>${t('rejectedClass')}</span>
    <span class="legend-item"><span class="legend-ring"></span>${t('supportVectorLegend')}</span>`;
}

function renderMargin(d) {
  const md = d.margin_distribution;
  const wrap = document.querySelector('#card-margin .chart-wrap');
  const canvas = document.getElementById('margin-canvas');
  const width = Math.max(280, wrap.clientWidth);
  const height = 300;
  const ctx = setupCanvas(canvas, width, height);
  ctx.clearRect(0, 0, width, height);

  const pad = { left: 46, right: 16, top: 16, bottom: 40 };
  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;

  const edges = md.bin_edges;
  const counts = md.counts;
  const maxCount = Math.max(...counts, 1);
  const xMin = edges[0];
  const xMax = edges[edges.length - 1];
  const sx = (v) => pad.left + ((v - xMin) / (xMax - xMin)) * plotW;
  const barW = plotW / counts.length;

  counts.forEach((c, i) => {
    const binCenter = (edges[i] + edges[i + 1]) / 2;
    const color = binCenter >= 0 ? PALETTE.approve : PALETTE.reject;
    const h = (c / maxCount) * plotH;
    ctx.fillStyle = `rgba(${color[0]},${color[1]},${color[2]},0.75)`;
    ctx.fillRect(pad.left + i * barW + 1, pad.top + plotH - h, Math.max(1, barW - 2), h);
  });

  if (xMin < 0 && xMax > 0) {
    const zx = sx(0);
    ctx.strokeStyle = '#101a2c';
    ctx.setLineDash([4, 3]);
    ctx.beginPath();
    ctx.moveTo(zx, pad.top);
    ctx.lineTo(zx, pad.top + plotH);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  ctx.strokeStyle = PALETTE.grid;
  ctx.beginPath();
  ctx.rect(pad.left, pad.top, plotW, plotH);
  ctx.stroke();

  ctx.fillStyle = PALETTE.axis;
  ctx.font = '11px Heebo, sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText(t('marginXLabel'), pad.left + plotW / 2, height - 10);
  ctx.save();
  ctx.translate(12, pad.top + plotH / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText(t('marginYLabel'), 0, 0);
  ctx.restore();
}

function renderAnalytics() {
  if (!analyticsData) return;
  renderModelDetails(analyticsData);
  renderModelFile(analyticsData);
  renderEndpoints(analyticsData);
  renderConfusion(analyticsData);
  renderReport(analyticsData);
  renderBoundary(analyticsData);
  renderMargin(analyticsData);
  refreshIcons();
}

async function loadAnalytics() {
  if (analyticsLoaded || analyticsLoading) return;
  analyticsLoading = true;
  try {
    const response = await fetch('/api/model-analytics');
    if (!response.ok) throw new Error();
    analyticsData = await response.json();
    analyticsLoaded = true;
    renderAnalytics();
    analyticsLoadingEl.classList.add('hidden');
    analyticsErrorEl.classList.add('hidden');
    analyticsContentEl.classList.remove('hidden');
  } catch (error) {
    analyticsLoadingEl.classList.add('hidden');
    analyticsErrorEl.classList.remove('hidden');
    analyticsErrorEl.textContent = t('analyticsFailed');
  } finally {
    analyticsLoading = false;
  }
}

document.querySelector('.nav-item[data-target="analytics"]').addEventListener('click', loadAnalytics);
document.querySelectorAll('.language-btn').forEach((btn) => btn.addEventListener('click', () => { if (analyticsLoaded) renderAnalytics(); }));

let resizeTimer = null;
window.addEventListener('resize', () => {
  if (!analyticsLoaded) return;
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => {
    if (document.getElementById('analytics').classList.contains('active')) {
      renderBoundary(analyticsData);
      renderMargin(analyticsData);
    }
  }, 150);
});
