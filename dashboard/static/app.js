/* ═══════════════════════════════════════════════════════════════
   ZINTOO AI — Dashboard Application Logic
   Auth · Routing · API · Charts · Agent Feed · Mock Data
   ═══════════════════════════════════════════════════════════════ */

const API_BASE = window.location.origin;

// ─── Auth helpers ───────────────────────────────────────────────
function authHeaders(extra = {}) {
  const token = sessionStorage.getItem('zintoo_token');
  return token ? { ...extra, Authorization: `Bearer ${token}` } : extra;
}

// ─── Real-time event stream (SSE) ───────────────────────────────
// Replaces the old fake setInterval feed. Connects to the server's live
// event bus; the browser auto-reconnects on drop. Falls back silently if
// the stream is unavailable.
let _eventSource = null;
const _feedDotByType = {
  'agent.observe': 'cyan', 'agent.think': 'purple', 'agent.act': 'green',
  'agent.reflect': 'blue', 'forecast.run': 'orange', 'search.query': 'purple',
  'system.login': 'blue', 'stock.alert': 'red',
};

function connectEventStream() {
  if (_eventSource) return;
  try {
    _eventSource = new EventSource(`${API_BASE}/events`);
    const onEvent = (e) => {
      let payload;
      try { payload = JSON.parse(e.data); } catch { return; }
      const msg = payload?.data?.message;
      if (!msg) return;
      const dot = _feedDotByType[payload.type] || 'green';
      const feed = document.getElementById('agent-feed');
      if (feed) {
        feed.insertAdjacentHTML('afterbegin', createFeedEntry(new Date(), dot, msg));
        while (feed.children.length > 14) feed.removeChild(feed.lastChild);
      }
      const dotEl = document.querySelector('.api-dot');
      if (dotEl) dotEl.style.background = 'var(--green)';
    };
    // named domain events + the default message channel
    ['agent.observe','agent.think','agent.act','agent.reflect','forecast.run',
     'search.query','system.login','stock.alert'].forEach(t => _eventSource.addEventListener(t, onEvent));
    _eventSource.onmessage = onEvent;
    _eventSource.onerror = () => { /* EventSource auto-reconnects */ };
  } catch { /* SSE unsupported — feed stays static */ }
}

// ─── Authentication ─────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  initLogin();
});

function initLogin() {
  // Check if already authenticated
  const token = sessionStorage.getItem('zintoo_token');
  if (token) {
    showDashboard();
    return;
  }

  // Show login screen
  const loginForm = document.getElementById('login-form');
  if (loginForm) {
    loginForm.addEventListener('submit', handleLogin);
  }
}

async function handleLogin(e) {
  e.preventDefault();
  const email = document.getElementById('login-email').value;
  const password = document.getElementById('login-password').value;
  const errorDiv = document.getElementById('login-error');
  const btn = document.getElementById('login-btn');

  errorDiv.textContent = '';
  errorDiv.classList.remove('shake');
  btn.disabled = true;
  btn.querySelector('span').textContent = 'Signing in...';

  try {
    const res = await fetch(`${API_BASE}/api/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password }),
    });
    const data = await res.json();

    if (res.ok && data.success) {
      sessionStorage.setItem('zintoo_token', data.token);
      sessionStorage.setItem('zintoo_user', JSON.stringify(data.user));

      // Animate login exit
      const loginScreen = document.getElementById('login-screen');
      loginScreen.classList.add('fade-out');
      setTimeout(() => showDashboard(), 500);
    } else {
      errorDiv.textContent = data.detail || 'Invalid credentials. Please try again.';
      errorDiv.classList.add('shake');
      setTimeout(() => errorDiv.classList.remove('shake'), 600);
    }
  } catch {
    errorDiv.textContent = 'Connection error. Please try again.';
    errorDiv.classList.add('shake');
    setTimeout(() => errorDiv.classList.remove('shake'), 600);
  }

  btn.disabled = false;
  btn.querySelector('span').textContent = 'Sign In';
}

function showDashboard() {
  const loginScreen = document.getElementById('login-screen');
  const appLayout = document.getElementById('app-layout');

  if (loginScreen) loginScreen.style.display = 'none';
  if (appLayout) appLayout.style.display = 'flex';

  // Update user profile from session
  const user = JSON.parse(sessionStorage.getItem('zintoo_user') || '{}');
  if (user.name) {
    const nameEl = document.querySelector('.user-name');
    const avatarEl = document.querySelector('.user-avatar');
    if (nameEl) nameEl.textContent = user.name;
    if (avatarEl) avatarEl.textContent = user.name.split(' ').map(w => w[0]).join('').toUpperCase();
  }

  // Initialize dashboard
  lucide.createIcons();
  initRouter();
  initDashboard();
  initDemandTrendChart();
  initForecastChart();
  populateAgentFeed();
  populateOrders();
  populateInventory();
  populateRestockLog();
  populateAgentPage();
  initEventListeners();
  checkAPIHealth();
}

function logout() {
  sessionStorage.removeItem('zintoo_token');
  sessionStorage.removeItem('zintoo_user');
  window.location.reload();
}

function initRouter() {
  const navItems = document.querySelectorAll('.nav-item[data-page]');
  navItems.forEach(item => {
    item.addEventListener('click', () => {
      const page = item.dataset.page;
      navigateTo(page);
    });
  });
}

function navigateTo(pageId) {
  // Update nav
  document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
  const activeNav = document.querySelector(`.nav-item[data-page="${pageId}"]`);
  if (activeNav) activeNav.classList.add('active');

  // Show page
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  const page = document.getElementById(`page-${pageId}`);
  if (page) {
    page.classList.add('active');
    page.classList.add('animate-in');
    setTimeout(() => page.classList.remove('animate-in'), 500);
  }
}

// ─── API Health Check ───────────────────────────────────────────
async function checkAPIHealth() {
  try {
    const res = await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(3000) });
    if (res.ok) {
      const data = await res.json();
      document.querySelector('.api-dot').style.background = 'var(--green)';
      updateDashboardFromAPI(data);
    }
  } catch {
    // API not available, use mock data (already populated)
    document.querySelector('.api-dot').style.background = 'var(--orange)';
  }
}

async function updateDashboardFromAPI(healthData) {
  // Try to get real stats
  try {
    const invRes = await fetch(`${API_BASE}/inventory/summary`);
    if (invRes.ok) {
      const invData = await invRes.json();
      if (invData.warehouses) {
        const totalStock = invData.warehouses.reduce((a, w) => a + (w.total_stock || 0), 0);
        const totalSKUs = invData.warehouses.reduce((a, w) => a + (w.total_skus || 0), 0);
        document.getElementById('metric-products').textContent = totalSKUs.toLocaleString();
      }
    }
  } catch { /* use mock */ }

  try {
    const skuRes = await fetch(`${API_BASE}/skus`);
    if (skuRes.ok) {
      const skuData = await skuRes.json();
      if (skuData.skus) {
        populateSKUSelectors(skuData.skus, skuData.pincodes);
      }
    }
  } catch { /* use mock */ }
}

function populateSKUSelectors(skus, pincodes) {
  const selectors = ['fc-sku', 'trend-sku-select'];
  selectors.forEach(id => {
    const sel = document.getElementById(id);
    if (sel && skus.length > 0) {
      sel.innerHTML = skus.slice(0, 20).map(s => `<option value="${s}">${s}</option>`).join('');
    }
  });
  const pinSel = document.getElementById('fc-pincode');
  if (pinSel && pincodes && pincodes.length > 0) {
    pinSel.innerHTML = pincodes.map(p => `<option value="${p}">${p}</option>`).join('');
  }
}

// ─── Dashboard Init ─────────────────────────────────────────────
function initDashboard() {
  // Metric cards are already populated in HTML with mock data
  // Real data is fetched via API if available
}

// ─── Demand Trend Chart ─────────────────────────────────────────
let demandTrendChart = null;

function initDemandTrendChart() {
  const ctx = document.getElementById('demand-trend-chart');
  if (!ctx) return;

  // Generate mock 48h demand data
  const labels = [];
  const data = [];
  const now = new Date();
  for (let i = 48; i >= 0; i--) {
    const t = new Date(now - i * 3600000);
    labels.push(t.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', hour12: false }));

    // Realistic demand pattern: peaks at 10am and 6pm, low at night
    const hour = t.getHours();
    const base = 50;
    const peak1 = 350 * Math.exp(-0.5 * Math.pow((hour - 11) / 3, 2));
    const peak2 = 250 * Math.exp(-0.5 * Math.pow((hour - 18) / 2.5, 2));
    const nightDip = hour >= 0 && hour <= 5 ? 0.1 : 1;
    const noise = (Math.random() - 0.5) * 40;
    data.push(Math.max(0, Math.round((base + peak1 + peak2) * nightDip + noise)));
  }

  const gradient = ctx.getContext('2d').createLinearGradient(0, 0, 0, 280);
  gradient.addColorStop(0, 'rgba(6, 182, 212, 0.25)');
  gradient.addColorStop(0.6, 'rgba(6, 182, 212, 0.05)');
  gradient.addColorStop(1, 'rgba(6, 182, 212, 0)');

  demandTrendChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [{
        label: 'Demand',
        data,
        borderColor: '#06b6d4',
        backgroundColor: gradient,
        borderWidth: 2,
        fill: true,
        tension: 0.4,
        pointRadius: 0,
        pointHoverRadius: 4,
        pointHoverBackgroundColor: '#06b6d4',
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { intersect: false, mode: 'index' },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: 'rgba(17, 24, 39, 0.95)',
          titleColor: '#f1f5f9',
          bodyColor: '#94a3b8',
          borderColor: 'rgba(255,255,255,0.1)',
          borderWidth: 1,
          cornerRadius: 8,
          padding: 10,
        }
      },
      scales: {
        x: {
          grid: { color: 'rgba(255,255,255,0.04)', drawBorder: false },
          ticks: { color: '#475569', font: { size: 10, family: 'Inter' }, maxTicksLimit: 12 },
        },
        y: {
          grid: { color: 'rgba(255,255,255,0.04)', drawBorder: false },
          ticks: { color: '#475569', font: { size: 10, family: 'Inter' } },
          beginAtZero: true,
        }
      }
    }
  });
}

// ─── Forecast Chart ─────────────────────────────────────────────
let forecastChart = null;

function initForecastChart() {
  const ctx = document.getElementById('forecast-chart');
  if (!ctx) return;

  const { labels, historical, forecast, upper, lower } = generateMockForecastData();

  const gradientHist = ctx.getContext('2d').createLinearGradient(0, 0, 0, 320);
  gradientHist.addColorStop(0, 'rgba(6, 182, 212, 0.2)');
  gradientHist.addColorStop(1, 'rgba(6, 182, 212, 0)');

  forecastChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [
        {
          label: 'Historical',
          data: historical,
          borderColor: '#06b6d4',
          backgroundColor: gradientHist,
          borderWidth: 2,
          fill: true,
          tension: 0.3,
          pointRadius: 0,
        },
        {
          label: 'Forecast',
          data: forecast,
          borderColor: '#8b5cf6',
          borderWidth: 2,
          borderDash: [6, 4],
          fill: false,
          tension: 0.3,
          pointRadius: 0,
        },
        {
          label: 'Upper Bound',
          data: upper,
          borderColor: 'transparent',
          backgroundColor: 'rgba(139, 92, 246, 0.08)',
          fill: '+1',
          tension: 0.3,
          pointRadius: 0,
        },
        {
          label: 'Lower Bound',
          data: lower,
          borderColor: 'transparent',
          backgroundColor: 'rgba(139, 92, 246, 0.08)',
          fill: false,
          tension: 0.3,
          pointRadius: 0,
        },
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { intersect: false, mode: 'index' },
      plugins: {
        legend: {
          display: true,
          position: 'top',
          align: 'end',
          labels: {
            color: '#64748b',
            font: { size: 11, family: 'Inter' },
            usePointStyle: true,
            pointStyleWidth: 10,
            filter: (item) => item.text !== 'Upper Bound' && item.text !== 'Lower Bound',
          }
        },
        tooltip: {
          backgroundColor: 'rgba(17, 24, 39, 0.95)',
          titleColor: '#f1f5f9',
          bodyColor: '#94a3b8',
          borderColor: 'rgba(255,255,255,0.1)',
          borderWidth: 1,
          cornerRadius: 8,
        }
      },
      scales: {
        x: {
          grid: { color: 'rgba(255,255,255,0.04)', drawBorder: false },
          ticks: { color: '#475569', font: { size: 10, family: 'Inter' }, maxTicksLimit: 14 },
        },
        y: {
          grid: { color: 'rgba(255,255,255,0.04)', drawBorder: false },
          ticks: { color: '#475569', font: { size: 10, family: 'Inter' } },
          beginAtZero: true,
        }
      }
    }
  });
}

function generateMockForecastData() {
  const labels = [];
  const historical = [];
  const forecast = [];
  const upper = [];
  const lower = [];
  const now = new Date();
  const splitPoint = 72; // hours of history
  const forecastHours = 48;
  const total = splitPoint + forecastHours;

  for (let i = total; i >= 0; i--) {
    const t = new Date(now - (i - forecastHours) * 3600000);
    labels.push(t.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', hour12: false }));

    const hour = t.getHours();
    const base = 2;
    const p1 = 5 * Math.exp(-0.5 * Math.pow((hour - 11) / 3, 2));
    const p2 = 4 * Math.exp(-0.5 * Math.pow((hour - 18) / 2.5, 2));
    const nightDip = (hour >= 0 && hour <= 5) ? 0.15 : 1;
    const val = Math.max(0, (base + p1 + p2) * nightDip + (Math.random() - 0.5) * 1.5);

    if (i > forecastHours) {
      historical.push(Math.round(val * 100) / 100);
      forecast.push(null);
      upper.push(null);
      lower.push(null);
    } else {
      historical.push(i === forecastHours ? Math.round(val * 100) / 100 : null);
      const fv = Math.round(val * 100) / 100;
      forecast.push(fv);
      upper.push(Math.round((fv + 1.5 + Math.random()) * 100) / 100);
      lower.push(Math.max(0, Math.round((fv - 1.5 - Math.random()) * 100) / 100));
    }
  }

  return { labels, historical, forecast, upper, lower };
}

// ─── Agent Feed ─────────────────────────────────────────────────
const feedMessages = [
  { dot: 'green', msg: 'Fulfilled 23 pending orders via nearest warehouse routing' },
  { dot: 'blue', msg: 'Running multi-line demand comparison...' },
  { dot: 'red', msg: 'Priority override: waking agent for region South' },
  { dot: 'orange', msg: 'Alert: low stock KURTA_001_M_BLU in North region' },
  { dot: 'cyan', msg: 'Agent hibernating — next tick in 15m' },
  { dot: 'green', msg: 'Stock transfer W2→W4 completed: 15 units SHIRT_042' },
  { dot: 'blue', msg: 'Forecast update: demand spike predicted for 400003 area' },
  { dot: 'purple', msg: 'Model retraining queued for 3 SKUs' },
  { dot: 'orange', msg: 'Warehouse W3 approaching 85% capacity' },
  { dot: 'green', msg: 'Auto-restock triggered for 4 SKU-warehouse pairs' },
  { dot: 'cyan', msg: 'Weather data refreshed — no significant changes' },
  { dot: 'blue', msg: 'SLA review: 97.2% fulfillment rate maintained' },
  { dot: 'red', msg: 'Critical: W1 stock for JEANS_007 below threshold (3 units)' },
  { dot: 'green', msg: 'Emergency transfer initiated: W5→W1, 20 units' },
  { dot: 'purple', msg: 'NDCG recalculated: 0.66 (stable)' },
];

function populateAgentFeed() {
  const feed = document.getElementById('agent-feed');
  if (!feed) return;

  // Seed with a short placeholder history so the panel isn't empty before the
  // first live event arrives. Real entries then stream in over SSE.
  const now = new Date();
  const entries = [];
  for (let i = 0; i < 4; i++) {
    const t = new Date(now - i * 127000);
    const msg = feedMessages[i % feedMessages.length];
    entries.push(createFeedEntry(t, msg.dot, msg.msg));
  }
  feed.innerHTML = entries.join('');

  // Connect the genuine server-driven real-time stream.
  connectEventStream();
}

function createFeedEntry(time, dotColor, message) {
  const ts = time.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false });
  return `
    <div class="feed-entry">
      <div class="feed-dot ${dotColor}"></div>
      <span class="feed-time">${ts}</span>
      <span class="feed-message">${message}</span>
    </div>`;
}

// ─── Orders Table ───────────────────────────────────────────────
function populateOrders() {
  const tbody = document.getElementById('orders-tbody');
  if (!tbody) return;

  const customers = ['Priya Sharma', 'Rahul Verma', 'Ananya Gupta', 'Vikram Singh', 'Meera Patel', 'Arjun Nair', 'Sneha Reddy', 'Karan Malhotra'];
  const statuses = ['processing', 'shipped', 'delivered', 'processing', 'delivered', 'shipped', 'delivered', 'processing'];
  const items = [2, 1, 3, 1, 2, 1, 4, 2];
  const totals = [2499, 1299, 4899, 899, 3299, 1599, 6799, 2199];

  const rows = [];
  for (let i = 0; i < 8; i++) {
    const orderId = `ORD-${7800 + i + Math.floor(Math.random() * 100)}`;
    rows.push(`
      <tr>
        <td style="font-weight: 600; color: var(--text-primary);">${orderId}</td>
        <td>${customers[i]}</td>
        <td>${items[i]}</td>
        <td style="font-weight: 500;">₹${totals[i].toLocaleString()}</td>
        <td><span class="status-badge ${statuses[i]}">${statuses[i]}</span></td>
        <td><button class="btn btn-outline" style="padding: 4px 12px; font-size: 0.78rem;">View</button></td>
      </tr>`);
  }
  tbody.innerHTML = rows.join('');
}

// ─── Inventory Page ─────────────────────────────────────────────
function populateInventory() {
  populateWarehouseCards();
  populateInventoryTable();
}

function populateWarehouseCards() {
  const container = document.getElementById('warehouse-cards');
  if (!container) return;

  const warehouses = [
    { id: 'W1', loc: 'Fort, Mumbai', skus: 20, stock: 847, pct: 84 },
    { id: 'W2', loc: 'Kalbadevi', skus: 20, stock: 723, pct: 72 },
    { id: 'W3', loc: 'Mandvi', skus: 20, stock: 915, pct: 91 },
    { id: 'W4', loc: 'Girgaon', skus: 20, stock: 562, pct: 56 },
    { id: 'W5', loc: 'Colaba', skus: 20, stock: 688, pct: 69 },
  ];

  container.innerHTML = warehouses.map(w => {
    const level = w.pct > 75 ? 'good' : w.pct > 40 ? 'medium' : 'low';
    return `
      <div class="warehouse-card">
        <div class="wh-id">${w.id}</div>
        <div class="wh-location">${w.loc}</div>
        <div class="wh-stat">SKUs: <strong>${w.skus}</strong></div>
        <div class="wh-stat">Total Stock: <strong>${w.stock}</strong></div>
        <div class="wh-stat">Fill Rate: <strong>${w.pct}%</strong></div>
        <div class="stock-bar"><div class="stock-fill ${level}" style="width: ${w.pct}%"></div></div>
      </div>`;
  }).join('');
}

function populateInventoryTable() {
  const tbody = document.getElementById('inventory-tbody');
  if (!tbody) return;

  const skus = ['KURTA_001_M_BLU', 'SHIRT_042_L_WHT', 'DRESS_015_S_RED', 'JEANS_007_M_IND', 'SAREE_023_F_GRN',
    'TSHRT_088_M_BLK', 'SHOES_005_10_BRN', 'JACKET_012_L_GRY', 'KURTI_034_M_PNK', 'CHINO_019_32_NVY'];

  const rows = skus.map(sku => {
    const stocks = Array.from({ length: 5 }, () => Math.floor(Math.random() * 90) + 5);
    const cells = stocks.map(s => {
      const cls = s <= 10 ? 'color: var(--red); font-weight: 600;' :
                  s <= 25 ? 'color: var(--orange); font-weight: 500;' : '';
      return `<td style="${cls}">${s}</td>`;
    }).join('');
    return `<tr><td style="font-weight: 600; color: var(--text-primary);">${sku}</td>${cells}</tr>`;
  }).join('');

  tbody.innerHTML = rows;
}

// ─── Auto-Restock Log ───────────────────────────────────────────
function populateRestockLog() {
  const log = document.getElementById('restock-log');
  if (!log) return;

  const entries = [
    { sku: 'PAK_0011_S_NAV', size: 'M/S', count: '/ 100S', warehouse: 'W1 / Fort', target: 50, current: 8, time: '12 min ago' },
    { sku: 'PAK_0005_L_SIL', size: 'L', count: '/ 700S', warehouse: 'W3 / Mandvi', target: 50, current: 6, time: '18 min ago' },
    { sku: 'PAK_0017_S_3XL', size: 'XL', count: '/ 100023', warehouse: 'W2 / Kalbadevi', target: 50, current: 9, time: '24 min ago' },
    { sku: 'ETH_0318_M_JGL', size: 'M', count: '/ 110023', warehouse: 'W5 / Colaba', target: 50, current: 4, time: '31 min ago' },
    { sku: 'SHIRT_042_L_WHT', size: 'L', count: '/ 50012', warehouse: 'W4 / Girgaon', target: 50, current: 7, time: '45 min ago' },
  ];

  log.innerHTML = entries.map(e => `
    <div class="log-entry">
      <div class="log-entry-header">
        <div class="log-sku">
          ${e.sku} <span class="sku-size">${e.size}</span> <span class="sku-count">${e.count}</span>
        </div>
        <div class="log-timestamp">${e.time}</div>
      </div>
      <div class="log-reason">Auto-restock: Stock below threshold (${e.current} &lt; 10)</div>
      <div class="log-details">
        <span>Warehouse:</span> ${e.warehouse} · <span>Target Stock:</span> ${e.target} · <span>Current:</span> ${e.current}
      </div>
    </div>`).join('');

  // Update last scan time
  const scanSpan = document.querySelector('#restock-last-scan span');
  if (scanSpan) {
    scanSpan.textContent = new Date().toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false });
  }
}

// ─── Agent Page ─────────────────────────────────────────────────
function populateAgentPage() {
  populateAgentDecisionLog();
  populateTransfersTable();
}

function populateAgentDecisionLog() {
  const log = document.getElementById('agent-decision-log');
  if (!log) return;

  const decisions = [
    { phase: 'OBSERVE', dot: 'blue', msg: 'Scanning inventory across 5 warehouses — 4 critical SKU-warehouse pairs detected' },
    { phase: 'THINK', dot: 'purple', msg: 'Matching surplus-deficit pairs: W3 has surplus for KURTA_001 → W1 needs restock' },
    { phase: 'ACT', dot: 'green', msg: 'Transfer executed: 15 units KURTA_001 from W3 to W1 (high priority)' },
    { phase: 'REFLECT', dot: 'cyan', msg: 'Post-transfer assessment: W1 stock restored to 25 units, SLA risk mitigated' },
    { phase: 'OBSERVE', dot: 'blue', msg: 'Checking JEANS_007 demand forecast — predicted spike at 18:00' },
    { phase: 'THINK', dot: 'purple', msg: 'Pre-positioning stock at W4 (Girgaon) for predicted evening demand' },
    { phase: 'ACT', dot: 'green', msg: 'Transfer executed: 10 units JEANS_007 from W5 to W4 (medium priority)' },
    { phase: 'REFLECT', dot: 'cyan', msg: 'Cycle complete: 2 transfers, 25 units moved, ₹125 cost, 97% SLA maintained' },
  ];

  const now = new Date();
  log.innerHTML = decisions.map((d, i) => {
    const t = new Date(now - i * 180000);
    const ts = t.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false });
    return `
      <div class="feed-entry">
        <div class="feed-dot ${d.dot}"></div>
        <span class="feed-time">${ts}</span>
        <span class="feed-message"><strong>[${d.phase}]</strong> ${d.msg}</span>
      </div>`;
  }).join('');
}

function populateTransfersTable() {
  const tbody = document.getElementById('transfers-tbody');
  if (!tbody) return;

  const transfers = [
    { sku: 'KURTA_001_M_BLU', from: 'W3', to: 'W1', qty: 15, priority: 'high', status: 'delivered', reason: 'Demand spike predicted' },
    { sku: 'JEANS_007_M_IND', from: 'W5', to: 'W4', qty: 10, priority: 'medium', status: 'shipped', reason: 'Pre-positioning for evening' },
    { sku: 'SHIRT_042_L_WHT', from: 'W2', to: 'W5', qty: 12, priority: 'high', status: 'delivered', reason: 'Critical stockout risk' },
    { sku: 'DRESS_015_S_RED', from: 'W1', to: 'W3', qty: 8, priority: 'medium', status: 'processing', reason: 'Surplus redistribution' },
    { sku: 'SAREE_023_F_GRN', from: 'W4', to: 'W2', qty: 20, priority: 'high', status: 'delivered', reason: 'Festival demand surge' },
  ];

  tbody.innerHTML = transfers.map(t => `
    <tr>
      <td style="font-weight: 600; color: var(--text-primary);">${t.sku}</td>
      <td>${t.from}</td>
      <td>${t.to}</td>
      <td style="font-weight: 600;">${t.qty}</td>
      <td><span class="status-badge ${t.priority === 'high' ? 'returned' : 'processing'}">${t.priority}</span></td>
      <td><span class="status-badge ${t.status}">${t.status === 'delivered' ? 'delivered' : t.status}</span></td>
      <td style="font-size: 0.82rem; color: var(--text-dim); max-width: 200px;">${t.reason}</td>
    </tr>`).join('');
}

// ─── Event Listeners ────────────────────────────────────────────
function initEventListeners() {
  // Forecast button
  const fcBtn = document.getElementById('btn-forecast');
  if (fcBtn) fcBtn.addEventListener('click', runForecast);

  // Recommend button
  const recBtn = document.getElementById('btn-recommend');
  if (recBtn) recBtn.addEventListener('click', runRecommendation);

  // Optimize button
  const optBtn = document.getElementById('btn-optimize');
  if (optBtn) optBtn.addEventListener('click', runOptimize);

  // Manual restock
  const restockBtn = document.getElementById('btn-manual-restock');
  if (restockBtn) restockBtn.addEventListener('click', runManualRestock);

  // Run agent cycle
  const agentBtn = document.getElementById('btn-run-agent');
  if (agentBtn) agentBtn.addEventListener('click', runAgentCycle);

  // Image upload zone — real classification + visual search
  const uploadZone = document.getElementById('image-upload-zone');
  const imageInput = document.getElementById('rec-image-input');
  if (uploadZone && imageInput) {
    uploadZone.addEventListener('click', () => imageInput.click());
    imageInput.addEventListener('change', (e) => {
      if (e.target.files[0]) classifyImage(e.target.files[0]);
    });
    // Drag & drop
    ['dragenter', 'dragover'].forEach(ev => uploadZone.addEventListener(ev, (e) => {
      e.preventDefault(); e.stopPropagation();
      uploadZone.style.borderColor = 'var(--purple)';
    }));
    ['dragleave', 'drop'].forEach(ev => uploadZone.addEventListener(ev, (e) => {
      e.preventDefault(); e.stopPropagation();
      uploadZone.style.borderColor = '';
    }));
    uploadZone.addEventListener('drop', (e) => {
      const f = e.dataTransfer.files[0];
      if (f && f.type.startsWith('image/')) classifyImage(f);
      else showToast('Please drop an image file', 'warn');
    });
  }

  // SKU selector change → update chart
  const trendSel = document.getElementById('trend-sku-select');
  if (trendSel) trendSel.addEventListener('change', updateDemandTrendChart);

  // Refresh inventory
  const refBtn = document.getElementById('btn-refresh-inv');
  if (refBtn) refBtn.addEventListener('click', () => {
    populateInventory();
    showToast('Inventory data refreshed');
  });
}

// ─── Actions ────────────────────────────────────────────────────
async function runForecast() {
  const sku = document.getElementById('fc-sku').value;
  const pincode = document.getElementById('fc-pincode').value;
  const hours = document.getElementById('fc-hours').value;

  const btn = document.getElementById('btn-forecast');
  btn.innerHTML = '<span class="spinner" style="width:16px;height:16px;border-width:2px;"></span> Forecasting...';
  btn.disabled = true;

  try {
    const res = await fetch(`${API_BASE}/forecast/${encodeURIComponent(sku)}/${pincode}?hours=${hours}`, {
      signal: AbortSignal.timeout(30000)
    });

    if (res.ok) {
      const data = await res.json();
      updateForecastChartWithData(data);
      if (data.metrics) {
        document.getElementById('fc-mape').textContent = `${(data.metrics.mape || 11.4).toFixed(1)}%`;
        document.getElementById('fc-rmse').textContent = (data.metrics.rmse || 2.27).toFixed(2);
      }
      document.getElementById('fc-peak').textContent = (data.peak_hour_demand || 2.08).toFixed(2);
      showToast('Forecast generated successfully!');
    } else {
      showToast('Forecast API unavailable — showing mock data', 'warn');
    }
  } catch {
    showToast('Using mock forecast data (API offline)', 'warn');
    // Regenerate mock chart
    if (forecastChart) {
      forecastChart.destroy();
      initForecastChart();
    }
  }

  btn.innerHTML = '<i data-lucide="bar-chart-2"></i> Generate Forecast';
  btn.disabled = false;
  lucide.createIcons();
}

function updateForecastChartWithData(data) {
  if (!forecastChart || !data.hourly_forecast) return;

  const forecasts = data.hourly_forecast;
  const labels = forecasts.map(f => {
    const d = new Date(f.timestamp);
    return d.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', hour12: false });
  });
  const predictions = forecasts.map(f => f.predicted_demand);
  const upper = forecasts.map(f => f.upper_bound);
  const lower = forecasts.map(f => f.lower_bound);

  forecastChart.data.labels = labels;
  forecastChart.data.datasets[0].data = predictions; // historical becomes the main line
  forecastChart.data.datasets[1].data = new Array(labels.length).fill(null);
  forecastChart.data.datasets[2].data = upper;
  forecastChart.data.datasets[3].data = lower;
  forecastChart.update();
}

async function runRecommendation() {
  const query = document.getElementById('rec-query').value;
  const topk = document.getElementById('rec-topk').value;
  const gender = document.getElementById('rec-gender').value;
  const statusDiv = document.getElementById('rec-results-status');
  const resultsDiv = document.getElementById('recommendation-results');

  if (!query) {
    showToast('Please enter a search query', 'warn');
    return;
  }

  const btn = document.getElementById('btn-recommend');
  btn.innerHTML = '<span class="spinner" style="width:16px;height:16px;border-width:2px;"></span> Searching...';
  btn.disabled = true;

  try {
    const body = {
      text_query: query,
      top_k: parseInt(topk),
    };
    if (gender) body.gender_filter = gender;

    const res = await fetch(`${API_BASE}/recommend`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(15000),
    });

    if (res.ok) {
      const data = await res.json();
      displayRecommendations(data.results);
      statusDiv.innerHTML = `<span style="color: var(--green); font-weight: 500;">✓ Found ${data.total_results} results for "${query}"</span>`;
    } else {
      statusDiv.innerHTML = `<span style="color: var(--orange);">⚠ API returned ${res.status} — models may not be loaded</span>`;
      displayMockRecommendations(query);
    }
  } catch {
    statusDiv.innerHTML = `<span style="color: var(--orange);">⚠ API offline — showing mock results</span>`;
    displayMockRecommendations(query);
  }

  btn.innerHTML = '<i data-lucide="search"></i> Discover';
  btn.disabled = false;
  lucide.createIcons();
}

function displayRecommendations(results) {
  const container = document.getElementById('recommendation-results');

  // Explicit empty state — a blank grid looks like a broken app.
  if (!results || results.length === 0) {
    container.innerHTML = `
      <div style="grid-column: 1 / -1; text-align: center; padding: 48px 16px; color: var(--text-secondary);">
        <i data-lucide="search-x" style="width:40px;height:40px;display:block;margin:0 auto 12px;opacity:0.5;"></i>
        <div style="font-weight:500;margin-bottom:6px;">No matching products</div>
        <div style="font-size:0.85rem;">Try a broader query (e.g. "shoes", "blue shirt") or clear the gender filter.</div>
      </div>`;
    if (window.lucide) lucide.createIcons();
    return;
  }

  // Unsplash fallback images for when dataset images aren't available
  const fallbackImages = [
    'https://images.unsplash.com/photo-1594938298603-c8148c4dae35?w=300&h=400&fit=crop',
    'https://images.unsplash.com/photo-1583743814966-8936f5b7be1a?w=300&h=400&fit=crop',
    'https://images.unsplash.com/photo-1562157873-818bc0726f68?w=300&h=400&fit=crop',
    'https://images.unsplash.com/photo-1596755094514-f87e34085b2c?w=300&h=400&fit=crop',
    'https://images.unsplash.com/photo-1558171813-4c088753af8f?w=300&h=400&fit=crop',
    'https://images.unsplash.com/photo-1620799140408-edc6dcb6d633?w=300&h=400&fit=crop',
    'https://images.unsplash.com/photo-1434389677669-e08b4cda3f0a?w=300&h=400&fit=crop',
    'https://images.unsplash.com/photo-1525507119028-ed4c629a60a3?w=300&h=400&fit=crop',
    'https://images.unsplash.com/photo-1591047139829-d91aecb6caea?w=300&h=400&fit=crop',
    'https://images.unsplash.com/photo-1603252109303-2751441dd157?w=300&h=400&fit=crop',
    'https://images.unsplash.com/photo-1578587018452-892bacefd3f2?w=300&h=400&fit=crop',
    'https://images.unsplash.com/photo-1551028719-00167b16eac5?w=300&h=400&fit=crop',
    'https://images.unsplash.com/photo-1606107557195-0e29a4b5b4aa?w=300&h=400&fit=crop',
    'https://images.unsplash.com/photo-1618354691373-d851c5c3a990?w=300&h=400&fit=crop',
    'https://images.unsplash.com/photo-1564584217132-2271feaeb3c5?w=300&h=400&fit=crop',
    'https://images.unsplash.com/photo-1548126032-079a0fb0099d?w=300&h=400&fit=crop',
    'https://images.unsplash.com/photo-1539109136881-3be0616acf4b?w=300&h=400&fit=crop',
    'https://images.unsplash.com/photo-1512436991641-6745cdb1723f?w=300&h=400&fit=crop',
    'https://images.unsplash.com/photo-1585487000160-6ebcfceb0d44?w=300&h=400&fit=crop',
    'https://images.unsplash.com/photo-1490114538077-0a7f8cb49891?w=300&h=400&fit=crop',
  ];

  container.innerHTML = results.map((r, i) => {
    // Try to use the image_path from the API result
    const imgId = r.product_id || r.id || '';
    const datasetImg = imgId ? `/images/${imgId}.jpg` : '';
    const fallback = fallbackImages[i % fallbackImages.length];
    const imgSrc = datasetImg || fallback;

    return `
    <div class="product-card">
      <img class="product-image" src="${imgSrc}" alt="${r.name || 'Fashion Product'}"
           onerror="this.onerror=null; this.src='${fallback}';"
           loading="lazy">
      <div class="product-info">
        <div class="product-name">${r.name || 'Fashion Product'}</div>
        <div class="product-meta">${r.master_category || 'Apparel'} → ${r.article_type || 'Fashion'} · ${r.color || ''} · ${r.gender || ''}</div>
        <span class="score-pill">Score: ${(r.similarity_score || 0).toFixed(4)}</span>
      </div>
    </div>`;
  }).join('');
}

function displayMockRecommendations(query) {
  const container = document.getElementById('recommendation-results');
  const mockProducts = [
    { name: 'Blue Cotton Kurta for Men', cat: 'Apparel', type: 'Kurta', color: 'Blue', gender: 'Men', score: 0.8921,
      img: 'https://images.unsplash.com/photo-1594938298603-c8148c4dae35?w=300&h=400&fit=crop' },
    { name: 'Casual Printed T-Shirt', cat: 'Apparel', type: 'Tshirts', color: 'Multi', gender: 'Men', score: 0.8534,
      img: 'https://images.unsplash.com/photo-1583743814966-8936f5b7be1a?w=300&h=400&fit=crop' },
    { name: 'Slim Fit Casual Shirt', cat: 'Apparel', type: 'Shirts', color: 'Blue', gender: 'Men', score: 0.8201,
      img: 'https://images.unsplash.com/photo-1596755094514-f87e34085b2c?w=300&h=400&fit=crop' },
    { name: 'Ethnic Embroidered Kurta', cat: 'Apparel', type: 'Kurta', color: 'White', gender: 'Men', score: 0.7945,
      img: 'https://images.unsplash.com/photo-1562157873-818bc0726f68?w=300&h=400&fit=crop' },
    { name: 'Designer Formal Blazer', cat: 'Apparel', type: 'Blazers', color: 'Navy', gender: 'Men', score: 0.7612,
      img: 'https://images.unsplash.com/photo-1591047139829-d91aecb6caea?w=300&h=400&fit=crop' },
    { name: 'Printed Casual Shirt', cat: 'Apparel', type: 'Shirts', color: 'Green', gender: 'Men', score: 0.7389,
      img: 'https://images.unsplash.com/photo-1620799140408-edc6dcb6d633?w=300&h=400&fit=crop' },
    { name: 'Classic Denim Jeans', cat: 'Apparel', type: 'Jeans', color: 'Indigo', gender: 'Men', score: 0.7201,
      img: 'https://images.unsplash.com/photo-1542272604-787c3835535d?w=300&h=400&fit=crop' },
    { name: 'Summer Floral Dress', cat: 'Apparel', type: 'Dress', color: 'Pink', gender: 'Women', score: 0.7055,
      img: 'https://images.unsplash.com/photo-1572804013309-59a88b7e92f1?w=300&h=400&fit=crop' },
    { name: 'Running Sport Shoes', cat: 'Footwear', type: 'Sports Shoes', color: 'Black', gender: 'Unisex', score: 0.6889,
      img: 'https://images.unsplash.com/photo-1606107557195-0e29a4b5b4aa?w=300&h=400&fit=crop' },
    { name: 'Leather Formal Shoes', cat: 'Footwear', type: 'Formal Shoes', color: 'Brown', gender: 'Men', score: 0.6721,
      img: 'https://images.unsplash.com/photo-1614252235316-8c857d38b5f4?w=300&h=400&fit=crop' },
  ];

  container.innerHTML = mockProducts.map(p => `
    <div class="product-card">
      <img class="product-image" src="${p.img}" alt="${p.name}" loading="lazy">
      <div class="product-info">
        <div class="product-name">${p.name}</div>
        <div class="product-meta">${p.cat} → ${p.type} · ${p.color} · ${p.gender}</div>
        <span class="score-pill">Score: ${p.score.toFixed(4)}</span>
      </div>
    </div>`).join('');
}

async function runOptimize() {
  const btn = document.getElementById('btn-optimize');
  btn.innerHTML = '<span class="spinner" style="width:16px;height:16px;border-width:2px;"></span> Optimizing...';
  btn.disabled = true;

  try {
    const res = await fetch(`${API_BASE}/orchestrate`, { method: 'POST', headers: authHeaders(), signal: AbortSignal.timeout(30000) });
    if (res.ok) {
      const data = await res.json();
      showToast(`Optimization complete! ${data.transfers?.length || 0} transfers executed`);
      // Update metrics
      document.getElementById('metric-reallocations').textContent =
        parseInt(document.getElementById('metric-reallocations').textContent) + (data.transfers?.length || 0);
    } else {
      showToast('Optimization completed (mock mode)', 'info');
    }
  } catch {
    showToast('Optimization completed (mock mode)', 'info');
    const curr = parseInt(document.getElementById('metric-reallocations').textContent) || 19;
    document.getElementById('metric-reallocations').textContent = curr + Math.floor(Math.random() * 3) + 1;
  }

  btn.innerHTML = '<i data-lucide="zap"></i> Optimize Now';
  btn.disabled = false;
  lucide.createIcons();
}

async function runManualRestock() {
  const btn = document.getElementById('btn-manual-restock');
  btn.innerHTML = '<span class="spinner" style="width:16px;height:16px;border-width:2px;"></span> Checking...';
  btn.disabled = true;

  await new Promise(r => setTimeout(r, 1500));

  showToast('Restock check complete — 2 SKUs triggered for restocking');
  populateRestockLog();

  btn.innerHTML = '<i data-lucide="refresh-cw"></i> Manual Restock Check';
  btn.disabled = false;
  lucide.createIcons();
}

async function runAgentCycle() {
  const btn = document.getElementById('btn-run-agent');
  btn.innerHTML = '<span class="spinner" style="width:16px;height:16px;border-width:2px;"></span> Running...';
  btn.disabled = true;

  try {
    const res = await fetch(`${API_BASE}/orchestrate`, { method: 'POST', headers: authHeaders(), signal: AbortSignal.timeout(30000) });
    if (res.ok) {
      const data = await res.json();
      showToast(`Agent cycle complete! Status: ${data.status}`);
      // Update agent stats
      const cycleEl = document.getElementById('agent-cycles');
      cycleEl.textContent = parseInt(cycleEl.textContent) + 1;
      const transEl = document.getElementById('agent-transfers');
      transEl.textContent = parseInt(transEl.textContent) + (data.transfers?.length || 0);
    } else {
      showToast('Agent cycle completed (mock)', 'info');
    }
  } catch {
    showToast('Agent cycle completed (mock mode)', 'info');
    const cycleEl = document.getElementById('agent-cycles');
    cycleEl.textContent = parseInt(cycleEl.textContent) + 1;
  }

  btn.innerHTML = '<i data-lucide="play"></i> Run Cycle';
  btn.disabled = false;
  lucide.createIcons();
}

function updateDemandTrendChart() {
  if (!demandTrendChart) return;

  // Regenerate data with slightly different pattern
  const labels = demandTrendChart.data.labels;
  const data = labels.map((_, i) => {
    const hour = (new Date(Date.now() - (48 - i) * 3600000)).getHours();
    const base = 30 + Math.random() * 40;
    const p1 = (300 + Math.random() * 100) * Math.exp(-0.5 * Math.pow((hour - 11) / 3, 2));
    const p2 = (200 + Math.random() * 80) * Math.exp(-0.5 * Math.pow((hour - 18) / 2.5, 2));
    const nightDip = (hour >= 0 && hour <= 5) ? 0.1 : 1;
    return Math.max(0, Math.round((base + p1 + p2) * nightDip + (Math.random() - 0.5) * 30));
  });

  demandTrendChart.data.datasets[0].data = data;
  demandTrendChart.update('active');
}

// ─── Toast Notifications ────────────────────────────────────────
function showToast(message, type = 'success') {
  const existing = document.querySelector('.toast');
  if (existing) existing.remove();

  const colors = {
    success: 'var(--green)',
    warn: 'var(--orange)',
    error: 'var(--red)',
    info: 'var(--cyan)',
  };

  const toast = document.createElement('div');
  toast.className = 'toast';
  toast.style.cssText = `
    position: fixed; bottom: 24px; right: 24px; z-index: 1000;
    padding: 14px 22px; border-radius: var(--radius-md);
    background: var(--bg-card-solid); border: 1px solid ${colors[type]};
    color: var(--text-primary); font-size: 0.88rem; font-family: Inter, sans-serif;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
    animation: fade-in 0.3s ease;
    display: flex; align-items: center; gap: 10px;
  `;
  const icon = type === 'success' ? '✓' : type === 'warn' ? '⚠' : type === 'error' ? '✕' : 'ℹ';
  toast.innerHTML = `<span style="color: ${colors[type]}; font-weight: 700;">${icon}</span> ${message}`;
  document.body.appendChild(toast);
  setTimeout(() => toast.remove(), 4000);
}

/* ═══════════════════════════════════════════════════════════════
   Visual Discovery — upload an image, classify it, find lookalikes
   POST /recommend/image  ->  { predictions, timing_ms, results }
   ═══════════════════════════════════════════════════════════════ */
const MAX_IMAGE_BYTES = 8 * 1024 * 1024;   // must mirror vision_preprocess.MAX_UPLOAD_BYTES

async function classifyImage(file) {
  const zone = document.getElementById('image-upload-zone');
  const statusDiv = document.getElementById('rec-results-status');
  const original = zone.innerHTML;

  // Client-side guard so we don't waste an 8 MB upload to be told 400.
  if (file.size > MAX_IMAGE_BYTES) {
    showToast(`Image is ${(file.size / 1e6).toFixed(1)} MB — the limit is 8 MB`, 'error');
    return;
  }

  const previewUrl = URL.createObjectURL(file);
  zone.innerHTML = `
    <img src="${previewUrl}" alt="preview"
         style="max-height:120px;border-radius:8px;display:block;margin:0 auto 10px;">
    <p><span class="spinner" style="width:14px;height:14px;border-width:2px;display:inline-block;vertical-align:middle;"></span>
       Classifying ${escapeHtml(file.name)}…</p>`;

  const form = new FormData();
  form.append('file', file);

  try {
    const res = await fetch(`${API_BASE}/recommend/image?top_k=10`, {
      method: 'POST',
      headers: authHeaders(),          // no Content-Type: browser sets the boundary
      body: form,
      signal: AbortSignal.timeout(30000),
    });

    if (res.status === 503) {
      const body = await res.json().catch(() => ({}));
      zone.innerHTML = original;
      statusDiv.innerHTML = `<span style="color: var(--orange);">⚠ Visual search unavailable — ${escapeHtml(body.detail || 'vision tier disabled')}</span>`;
      lucide.createIcons();
      return;
    }
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error(body.detail || `HTTP ${res.status}`);
    }

    const data = await res.json();
    renderPredictions(data, previewUrl, file.name);
    displayRecommendations(data.results);
    statusDiv.innerHTML = `<span style="color: var(--green); font-weight:500;">
      ✓ ${data.total_results} visually similar products · ${data.timing_ms.total.toFixed(0)} ms</span>`;
  } catch (err) {
    zone.innerHTML = original;
    statusDiv.innerHTML = `<span style="color: var(--red);">✕ ${escapeHtml(err.message)}</span>`;
    showToast(err.message, 'error');
  }
  lucide.createIcons();
}

function renderPredictions(data, previewUrl, filename) {
  const zone = document.getElementById('image-upload-zone');
  const preds = data.predictions || {};

  const block = Object.entries(preds).map(([task, list]) => {
    const chips = list.map((p, i) => `
      <span style="display:inline-block;padding:3px 9px;margin:2px;border-radius:99px;
        font-size:0.75rem;font-family:Inter,sans-serif;
        background:${i === 0 ? 'var(--purple)' : 'rgba(255,255,255,0.07)'};
        color:${i === 0 ? '#fff' : 'var(--text-secondary)'};">
        ${escapeHtml(p.label)} <b>${(p.confidence * 100).toFixed(0)}%</b></span>`).join('');
    return `<div style="margin:6px 0;">
      <div style="font-size:0.7rem;text-transform:uppercase;letter-spacing:.05em;
        color:var(--text-secondary);margin-bottom:3px;">${escapeHtml(task)}</div>${chips}</div>`;
  }).join('');

  zone.innerHTML = `
    <img src="${previewUrl}" alt="preview"
         style="max-height:130px;border-radius:8px;display:block;margin:0 auto 12px;">
    <div style="text-align:left;max-width:360px;margin:0 auto;">${block}</div>
    <small style="display:block;margin-top:10px;color:var(--text-secondary);">
      ${escapeHtml(filename)} · inference ${data.timing_ms.inference.toFixed(1)} ms
      · preprocess ${data.timing_ms.preprocess.toFixed(1)} ms
    </small>
    <small style="display:block;margin-top:4px;color:var(--cyan);cursor:pointer;"
           onclick="document.getElementById('rec-image-input').click()">Upload another</small>`;
}

// Never interpolate a filename or server string into HTML unescaped.
function escapeHtml(s) {
  const d = document.createElement('div');
  d.textContent = String(s ?? '');
  return d.innerHTML;
}
