const rows = [...document.querySelectorAll('#results-table tbody tr')];
const svg = document.querySelector('#results-chart');
const ns = 'http://www.w3.org/2000/svg';
const x = (horizon) => 60 + (horizon - 1) * 135;
const y = (value) => 262 - (value / 0.6) * 225;

function element(tag, attributes, text) {
  const node = document.createElementNS(ns, tag);
  for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, value);
  if (text !== undefined) node.textContent = text;
  svg.append(node);
  return node;
}

for (let tick = 0; tick <= 6; tick += 1) {
  const position = y(tick / 10);
  element('line', { x1: 60, x2: 600, y1: position, y2: position, stroke: '#e1e8eb' });
  element('text', { x: 44, y: position + 4, 'text-anchor': 'end' }, (tick / 10).toFixed(1));
}
element('text', { x: 60, y: 19, 'font-size': 11 }, 'PR-AUC');
for (const row of rows) {
  element('text', { x: x(Number(row.dataset.horizon)), y: 286, 'text-anchor': 'middle' }, `${row.dataset.horizon} yr`);
}
const highlight = element('rect', { x: 38, y: 31, width: 44, height: 237, rx: 6, fill: '#122840', opacity: '.04' });
for (const [field, color, dashed] of [['prevalence', '#84929d', true], ['screening-ap', '#284f75', false], ['monotonic-ap', '#006e66', false]]) {
  const points = rows.map((row) => {
    const text = row.querySelector(`[data-field="${field}"]`).textContent;
    return [x(Number(row.dataset.horizon)), y(parseFloat(text) / (field === 'prevalence' ? 100 : 1))];
  });
  element('polyline', { points: points.map((point) => point.join(',')).join(' '), fill: 'none', stroke: color, 'stroke-width': dashed ? 2 : 3, ...(dashed ? { 'stroke-dasharray': '5 5' } : {}) });
  if (!dashed) points.forEach(([cx, cy]) => element('circle', { cx, cy, r: 4.5, fill: 'white', stroke: color, 'stroke-width': 2 }));
}

function selectHorizon(horizon) {
  const row = rows.find((candidate) => candidate.dataset.horizon === horizon);
  for (const field of ['screening-model', 'screening-ap', 'monotonic-model', 'monotonic-ap', 'prevalence']) {
    document.getElementById(field).textContent = row.querySelector(`[data-field="${field}"]`).textContent;
  }
  document.querySelector('#horizon-label').textContent = `${horizon}-year forecast · 5-year history`;
  const delta = Number(row.dataset.delta);
  document.querySelector('#delta-ap').textContent = `${delta >= 0 ? '+' : '−'}${Math.abs(delta).toFixed(4)}`;
  document.querySelector('#delta-ap').classList.toggle('negative', delta < 0);
  document.querySelector('#lift').textContent = `${row.dataset.lift}×`;
  document.querySelector('#roc').textContent = row.dataset.roc;
  document.querySelector('#brier').textContent = row.dataset.brier;
  highlight.setAttribute('x', x(Number(horizon)) - 22);
  document.querySelectorAll('.horizon-buttons button').forEach((button) => button.setAttribute('aria-pressed', String(button.dataset.horizon === horizon)));
  rows.forEach((candidate) => candidate.classList.toggle('is-selected', candidate === row));
}
document.querySelectorAll('.horizon-buttons button').forEach((button) => button.addEventListener('click', () => selectHorizon(button.dataset.horizon)));
selectHorizon('1');
document.querySelector('.result-explorer').hidden = false;

const copyButton = document.querySelector('#copy-citation');
copyButton.hidden = false;
copyButton.addEventListener('click', async () => {
  const status = document.querySelector('#copy-status');
  try {
    await navigator.clipboard.writeText(document.querySelector('#bibtex').textContent);
    status.textContent = 'BibTeX copied to clipboard.';
  } catch {
    status.textContent = 'Copy unavailable in this browser. Select the reference text or download the BibTeX file.';
  }
});

const dialog = document.querySelector('#figure-dialog');
document.querySelectorAll('[data-figure]').forEach((link) => link.addEventListener('click', (event) => {
  if (typeof dialog.showModal !== 'function' || event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
  event.preventDefault();
  const source = link.closest('figure').querySelector('img');
  const expanded = document.querySelector('#expanded-figure');
  expanded.src = source.src;
  expanded.alt = source.alt;
  document.querySelector('#figure-dialog-title').textContent = link.dataset.figure === 'pipeline' ? 'Study framework · Figure 1' : 'Monotonic versus screening · Figure 2';
  dialog.showModal();
  document.querySelector('.dialog-image-scroll').scrollTo(0, 0);
}));
document.querySelector('#close-figure').addEventListener('click', () => dialog.close());
dialog.addEventListener('click', (event) => {
  if (event.target === dialog) {
    const bounds = dialog.getBoundingClientRect();
    if (event.clientX < bounds.left || event.clientX > bounds.right || event.clientY < bounds.top || event.clientY > bounds.bottom) dialog.close();
  }
});
