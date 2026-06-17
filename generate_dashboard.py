#!/usr/bin/env python3
"""Generate and open the trading performance dashboard."""

import json, os, webbrowser
from datetime import date as _date
from pathlib import Path

LOG_PATH  = Path(__file__).parent / 'trade_log.json'
OUT_PATH  = Path(__file__).parent / 'dashboard.html'
ENV_PATH  = Path(__file__).parent / '.env'

# ── Balance: auto-computed from logged P&L ─────────────────────────────────────
BASE_BALANCE_GBP = 10_459   # confirmed balance as of 27 May 2026
BASELINE_DATE    = '2026-05-27'  # pnl_gbp from trades AFTER this date is auto-added

with open(LOG_PATH) as f:
    trades = json.load(f)

incremental_pnl = sum(
    t.get('pnl_gbp') or 0
    for t in trades
    if (t.get('pnl_gbp') or 0) != 0 and t.get('date', '') > BASELINE_DATE
)
ACCOUNT_GBP = round(BASE_BALANCE_GBP + incremental_pnl, 2)

# ── GBP/USD: fetched live at generation time, fallback 1.27 ───────────────────
def _fetch_gbpusd() -> float:
    try:
        from dotenv import load_dotenv
        load_dotenv(ENV_PATH)
        key = os.getenv('ALPHA_VANTAGE_API_KEY', '')
        if key:
            import urllib.request as _req
            url = (f'https://www.alphavantage.co/query?function=CURRENCY_EXCHANGE_RATE'
                   f'&from_currency=GBP&to_currency=USD&apikey={key}')
            with _req.urlopen(url, timeout=5) as r:
                d = json.loads(r.read())
                return float(d['Realtime Currency Exchange Rate']['5. Exchange Rate'])
    except Exception:
        pass
    return 1.27

GBPUSD = _fetch_gbpusd()

for t in trades:
    if t.get("date"):
        t["day_of_week"] = _date.fromisoformat(t["date"]).strftime("%A")

data = json.dumps(trades)

HTML = '''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Trading Journal — Scott Taylor</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background: #0d1421; color: #e2e8f0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; font-size: 13px; overflow: hidden; height: 100vh; display: flex; flex-direction: column; }

/* Header */
.header { background: #111827; border-bottom: 1px solid #1e2d3d; padding: 10px 24px; display: flex; align-items: center; justify-content: space-between; flex-shrink: 0; }
.header h1 { font-size: 16px; font-weight: 700; color: #fff; letter-spacing: -0.01em; }
.header h1 span { color: #3b82f6; }
.header .meta { color: #9ca3af; font-size: 11px; }

/* Stats strip */
.stats-strip { display: grid; grid-template-columns: repeat(7, 1fr); background: #0d1421; border-bottom: 1px solid #1e2d3d; flex-shrink: 0; }
.stat-card { background: #111827; padding: 10px 16px; border-right: 1px solid #1e2d3d; }
.stat-card:last-child { border-right: none; }
.stat-card .label { color: #cbd5e1; font-size: 10px; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 3px; }
.stat-card .value { font-size: 22px; font-weight: 800; color: #fff; line-height: 1; }
.stat-card .value.green { color: #22c55e; }
.stat-card .value.red { color: #ef4444; }
.stat-card .value.blue { color: #3b82f6; }
.stat-card .sub { font-size: 10px; color: #9ca3af; margin-top: 3px; }

/* Body layout */
.body { display: flex; flex: 1; min-height: 0; }

/* Calendar panel */
.calendar-panel { flex: 1; min-width: 0; display: flex; flex-direction: column; border-right: 1px solid #1e2d3d; overflow: hidden; }
.cal-top { padding: 12px 16px 8px; display: flex; align-items: center; justify-content: space-between; flex-shrink: 0; }
.month-nav { display: flex; align-items: center; gap: 8px; }
.month-nav h2 { font-size: 14px; font-weight: 600; min-width: 130px; text-align: center; }
.month-nav button { background: #1e2d3d; border: none; color: #e2e8f0; width: 24px; height: 24px; border-radius: 4px; cursor: pointer; font-size: 14px; line-height: 1; }
.month-nav button:hover { background: #2d3f52; }
.cal-summary { display: flex; gap: 12px; font-size: 11px; }
.cal-summary .t { color: #cbd5e1; }
.cal-summary .w { color: #22c55e; }
.cal-summary .l { color: #ef4444; }
.cal-grid-wrap { flex: 1; overflow-y: auto; padding: 0 16px 16px; }

/* 8-column grid: 7 days + weekly total */
.cal-grid { display: grid; grid-template-columns: repeat(7, 1fr) 72px; gap: 3px; }
.cal-day-header { text-align: center; color: #cbd5e1; font-size: 10px; padding: 5px 4px; text-transform: uppercase; letter-spacing: 0.06em; }
.cal-week-header { text-align: center; color: #3b82f6; font-size: 10px; padding: 5px 4px; text-transform: uppercase; letter-spacing: 0.06em; }
.cal-cell { background: #131a26; border-radius: 5px; min-height: 66px; padding: 6px 8px; transition: background 0.1s; }
.cal-cell.empty { background: transparent; min-height: 66px; }
.cal-cell.today { outline: 2px solid #3b82f6; }
.cal-cell .cal-date { font-size: 10px; color: #9ca3af; margin-bottom: 4px; font-weight: 500; }
.cal-cell .cal-pnl { font-size: 12px; font-weight: 700; line-height: 1.3; }
.cal-cell .cal-model { font-size: 9px; color: #9ca3af; margin-top: 3px; }
.cal-cell .cal-grade { display: inline-block; font-size: 9px; background: #1e2d3d; border-radius: 2px; padding: 1px 5px; margin-top: 3px; color: #cbd5e1; font-weight: 600; }
/* WIN — green tint, solid left border, subtle top accent */
.cal-cell.win  { background: rgba(34,197,94,0.10); border-left: 3px solid #22c55e; border-top: 1px solid rgba(34,197,94,0.3); }
.cal-cell.win  .cal-pnl { color: #4ade80; }
.cal-cell.win  .cal-date::after { content: ' ●'; color: #22c55e; font-size: 8px; }
/* LOSS — red tint */
.cal-cell.loss { background: rgba(239,68,68,0.10); border-left: 3px solid #ef4444; border-top: 1px solid rgba(239,68,68,0.3); }
.cal-cell.loss .cal-pnl { color: #f87171; }
.cal-cell.loss .cal-date::after { content: ' ●'; color: #ef4444; font-size: 8px; }
/* MISS — clearly distinct but muted */
.cal-cell.miss { background: rgba(255,255,255,0.03); border-left: 2px solid #374151; }
.cal-cell.miss .cal-pnl { color: #6b7280; }
/* BREAKEVEN — amber */
.cal-cell.be   { background: rgba(234,179,8,0.10);  border-left: 3px solid #eab308; border-top: 1px solid rgba(234,179,8,0.3); }
.cal-cell.be   .cal-pnl { color: #fbbf24; }
.cal-cell.be   .cal-date::after { content: ' ●'; color: #eab308; font-size: 8px; }
.cal-cell.violation { outline: 1px solid rgba(249,115,22,0.7); }

/* Weekly total cell */
.cal-week-total { background: #0d1421; border-radius: 5px; min-height: 66px; padding: 6px 8px; border: 1px solid #1e2d3d; display: flex; flex-direction: column; justify-content: center; gap: 3px; }
.cal-week-total .wk-label { font-size: 9px; color: #9ca3af; text-transform: uppercase; letter-spacing: 0.05em; }
.cal-week-total .wk-pts { font-size: 11px; font-weight: 700; }
.cal-week-total .wk-gbp { font-size: 10px; font-weight: 600; }
.cal-week-total .wk-pct { font-size: 9px; color: #9ca3af; }

/* Right panel */
.right-panel { width: 340px; flex-shrink: 0; display: flex; flex-direction: column; overflow-y: auto; }
.panel-card { padding: 14px 16px; border-bottom: 1px solid #1e2d3d; }
.panel-card h3 { font-size: 10px; color: #cbd5e1; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 10px; }
.gauge-row { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; align-items: center; }
.gauge-box { display: flex; flex-direction: column; align-items: center; }
.gauge-box canvas { max-width: 110px; }
.gauge-lbl { font-size: 10px; color: #9ca3af; margin-top: 4px; }
.ratio-box { display: flex; flex-direction: column; justify-content: center; gap: 6px; }
.ratio-big { font-size: 32px; font-weight: 800; line-height: 1; }
.ratio-lbl { font-size: 10px; color: #9ca3af; }
.ratio-bar { height: 6px; border-radius: 3px; background: rgba(239,68,68,0.4); overflow: hidden; }
.ratio-bar-win { height: 100%; background: #3b82f6; border-radius: 3px; }
.net-pnl-big { font-size: 28px; font-weight: 800; margin-bottom: 10px; line-height: 1; }

/* Bottom table */
.table-section { flex-shrink: 0; border-top: 1px solid #1e2d3d; max-height: 300px; overflow-y: auto; }
.table-header { padding: 8px 16px 6px; background: #111827; position: sticky; top: 0; z-index: 10; }
.table-header h2 { font-size: 12px; font-weight: 600; color: #cbd5e1; text-transform: uppercase; letter-spacing: 0.05em; }
.trade-table { width: 100%; border-collapse: collapse; font-size: 11px; }
.trade-table th { background: #0d1421; color: #9ca3af; text-align: left; padding: 6px 10px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em; font-size: 9px; position: sticky; top: 37px; white-space: nowrap; }
.trade-table td { padding: 6px 10px; border-bottom: 1px solid #131a26; vertical-align: top; white-space: nowrap; color: #e2e8f0; }
.trade-table tr:hover td { background: #131a26; }

/* Weekly summary row */
.week-row td { background: #0f1d2e !important; color: #9ca3af; font-size: 10px; font-weight: 600; border-top: 1px solid #1e2d3d; border-bottom: 1px solid #1e2d3d; padding: 5px 10px; white-space: nowrap; }
.week-row td:first-child { color: #3b82f6; }

/* Badges */
.badge { display: inline-block; padding: 2px 6px; border-radius: 3px; font-size: 10px; font-weight: 600; }
.badge-win  { background: rgba(34,197,94,0.12);  color: #22c55e; }
.badge-loss { background: rgba(239,68,68,0.12);  color: #ef4444; }
.badge-miss { background: rgba(107,114,128,0.12); color: #9ca3af; }
.badge-be   { background: rgba(234,179,8,0.12);  color: #eab308; }
.badge-aa   { background: rgba(168,85,247,0.12); color: #a855f7; }
.badge-a    { background: rgba(59,130,246,0.12); color: #3b82f6; }
.badge-b    { background: rgba(234,179,8,0.12);  color: #eab308; }
.w-ok  { color: #22c55e; }
.w-bad { color: #ef4444; font-weight: 700; }
.mono { font-family: 'SF Mono','Fira Code',monospace; }

/* Chart links */
.chart-link { display: inline-flex; align-items: center; gap: 3px; color: #3b82f6; text-decoration: none; background: rgba(59,130,246,0.1); border: 1px solid rgba(59,130,246,0.25); border-radius: 3px; padding: 2px 6px; font-size: 10px; font-weight: 500; white-space: nowrap; }
.chart-link:hover { background: rgba(59,130,246,0.2); border-color: rgba(59,130,246,0.5); }

/* Tooltip cells — long text shown truncated with hover popup */
.tip-cell { position: relative; max-width: 140px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: #cbd5e1; font-size: 10px; cursor: default; }
.tip-cell[data-tip]:not([data-tip="—"]):not([data-tip=""]) { cursor: help; border-bottom: 1px dashed #374151; }
.tip-cell::after {
  content: attr(data-tip);
  display: none;
  position: fixed;
  z-index: 9999;
  background: #1a2535;
  border: 1px solid #3b82f6;
  border-radius: 6px;
  padding: 10px 14px;
  font-size: 11px;
  color: #e2e8f0;
  width: 300px;
  white-space: normal;
  line-height: 1.6;
  box-shadow: 0 8px 32px rgba(0,0,0,0.5);
  pointer-events: none;
  bottom: auto;
  top: var(--tip-y, 50px);
  left: var(--tip-x, 50px);
}
.tip-cell:hover::after { display: block; }

/* ── Trade detail modal ── */
.trade-id-link { font-family:'SF Mono','Fira Code',monospace; font-size:9px; color:#3b82f6; cursor:pointer; white-space:nowrap; padding:2px 5px; border-radius:3px; background:rgba(59,130,246,0.08); }
.trade-id-link:hover { background:rgba(59,130,246,0.2); text-decoration:underline; }
.modal-overlay { display:none; position:fixed; inset:0; background:rgba(0,0,0,0.88); z-index:2000; align-items:center; justify-content:center; }
.modal-overlay.open { display:flex; }
.modal-box { width:90vw; max-width:1200px; max-height:90vh; background:#111827; border:1px solid #1e2d3d; border-radius:12px; overflow:hidden; display:flex; flex-direction:column; box-shadow:0 24px 80px rgba(0,0,0,0.7); }
.modal-header { padding:14px 20px; border-bottom:1px solid #1e2d3d; display:flex; align-items:center; justify-content:space-between; flex-shrink:0; background:#0d1421; gap:12px; }
.modal-header-left { display:flex; flex-direction:column; gap:5px; }
.modal-trade-id { font-family:'SF Mono','Fira Code',monospace; font-size:10px; color:#6b7280; }
.modal-header-badges { display:flex; align-items:center; gap:8px; flex-wrap:wrap; }
.modal-header-title { font-size:18px; font-weight:800; color:#fff; }
.modal-header-sub { font-size:12px; color:#9ca3af; }
.modal-close { background:#1e2d3d; border:1px solid #374151; color:#9ca3af; width:30px; height:30px; border-radius:6px; cursor:pointer; font-size:15px; line-height:1; flex-shrink:0; }
.modal-close:hover { background:#374151; color:#fff; }
.modal-body { display:flex; flex:1; min-height:0; overflow:hidden; }
.modal-chart-panel { width:55%; border-right:1px solid #1e2d3d; overflow-y:auto; padding:16px; display:flex; flex-direction:column; gap:14px; background:#0a1120; flex-shrink:0; }
.chart-item-label { display:flex; align-items:center; justify-content:space-between; margin-bottom:6px; }
.chart-item-label span { font-size:10px; color:#6b7280; text-transform:uppercase; letter-spacing:0.05em; }
.modal-chart-img { width:100%; border-radius:8px; border:1px solid #1e2d3d; display:block; }
.no-chart-placeholder { display:flex; align-items:center; justify-content:center; min-height:160px; color:#374151; font-size:12px; border:1px dashed #1e2d3d; border-radius:8px; text-align:center; padding:16px; }
.modal-details-panel { flex:1; overflow-y:auto; padding:16px 20px; }
.det-section { margin-bottom:14px; }
.det-section h4 { font-size:9px; color:#3b82f6; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:8px; padding-bottom:5px; border-bottom:1px solid #1e2d3d; }
.det-grid { display:grid; grid-template-columns:1fr 1fr; gap:6px 16px; }
.det-item .dk { font-size:9px; color:#6b7280; text-transform:uppercase; letter-spacing:0.04em; margin-bottom:2px; }
.det-item .dv { font-size:12px; color:#e2e8f0; font-weight:500; word-break:break-word; }
.det-full { margin-top:8px; }
.det-full .dk { font-size:9px; color:#6b7280; text-transform:uppercase; letter-spacing:0.04em; margin-bottom:4px; }
.det-full .dv { font-size:12px; color:#cbd5e1; line-height:1.7; white-space:pre-wrap; word-break:break-word; }
.conf-tag { display:inline-block; background:#1e2d3d; border-radius:3px; padding:2px 8px; font-size:10px; color:#9ca3af; margin:2px 2px 2px 0; }
</style>
</head>
<body>

<div class="header">
  <h1>Trading Journal &mdash; <span>Scott Taylor Framework</span></h1>
  <span class="meta" id="meta-line"></span>
</div>
<div class="stats-strip" id="stats-strip"></div>

<div class="body">
  <!-- Calendar -->
  <div class="calendar-panel">
    <div class="cal-top">
      <div class="month-nav">
        <button onclick="prevMonth()">&#8249;</button>
        <h2 id="month-title"></h2>
        <button onclick="nextMonth()">&#8250;</button>
      </div>
      <div class="cal-summary" id="cal-summary"></div>
    </div>
    <div class="cal-grid-wrap">
      <div class="cal-grid" id="cal-grid"></div>
    </div>
  </div>

  <!-- Right charts -->
  <div class="right-panel">
    <div class="panel-card">
      <h3>Win Rate &amp; Avg Day</h3>
      <div class="gauge-row">
        <div class="gauge-box">
          <canvas id="winRateChart"></canvas>
          <div class="gauge-lbl">Win Rate</div>
        </div>
        <div class="ratio-box">
          <div class="ratio-big" id="ratio-val"></div>
          <div class="ratio-lbl">Avg Win / Avg Loss</div>
          <div class="ratio-bar"><div class="ratio-bar-win" id="ratio-seg"></div></div>
        </div>
      </div>
    </div>
    <div class="panel-card">
      <h3>Net P&amp;L (Points &mdash; recorded trades)</h3>
      <div class="net-pnl-big" id="net-pnl-val"></div>
      <canvas id="pnlChart" height="100"></canvas>
    </div>
    <div class="panel-card">
      <h3>Grade Distribution</h3>
      <canvas id="gradeChart" height="90"></canvas>
    </div>
    <div class="panel-card">
      <h3>By Day of Week</h3>
      <canvas id="dowChart" height="90"></canvas>
    </div>
  </div>
</div>

<!-- Trade Detail Modal -->
<div class="modal-overlay" id="tradeModal" onclick="closeModal(event)">
  <div class="modal-box">
    <div class="modal-header" id="modal-header"></div>
    <div class="modal-body">
      <div class="modal-chart-panel" id="modal-chart-panel"></div>
      <div class="modal-details-panel" id="modal-details-panel"></div>
    </div>
  </div>
</div>

<!-- Trade Table -->
<div class="table-section">
  <div class="table-header"><h2>Trade Log</h2></div>
  <table class="trade-table">
    <thead>
      <tr>
        <th>ID</th><th>Date</th><th>Inst</th><th>Day</th><th>Mode</th>
        <th>Outcome</th><th>Grade</th><th>Win&#x2F;dow</th>
        <th>Entry</th><th>SL</th><th>TP</th><th>R:R</th>
        <th>P&amp;L pts</th><th>P&amp;L USD</th><th>% Acct</th>
        <th>Model</th><th>Trade Reason</th><th>Lesson</th><th>Psychology</th><th>Charts</th>
      </tr>
    </thead>
    <tbody id="trade-tbody"></tbody>
  </table>
</div>

<script>
const TRADES       = __TRADE_DATA__;
const ACCOUNT_GBP  = __ACCOUNT_GBP__;
const GBPUSD       = __GBPUSD__;

// ── Helpers ───────────────────────────────────────────────────────────────────
const $ = id => document.getElementById(id);

function isoWeek(dateStr) {
  const d  = new Date(dateStr + 'T00:00:00');
  const thu = new Date(d); thu.setDate(d.getDate() - (d.getDay()+6)%7 + 3);
  const jan4 = new Date(thu.getFullYear(), 0, 4);
  return thu.getFullYear() + '-W' + String(1 + Math.round((thu - jan4) / 604800000)).padStart(2,'0');
}

function gbpStr(v)  { return v ? (v>=0?'+':'-')+'£'+Math.abs(v).toFixed(2) : '—'; }
function usdStr(v)  { return v ? (v>=0?'+':'-')+'$'+Math.abs(v*GBPUSD).toFixed(2) : '—'; }
function pctStr(v)  { return v ? (v>=0?'+':'')+v.toFixed(2)+'%'            : '—'; }
function ptsStr(v)  { return v ? (v>=0?'+':'')+v.toFixed(2)+' pts'         : '—'; }

// pnl_gbp: use stored value if set, else zero (user logs it going forward)
TRADES.forEach(t => { if (!t.pnl_gbp) t.pnl_gbp = 0; });

// ── Aggregates ────────────────────────────────────────────────────────────────
const taken  = TRADES.filter(t => ['win','loss','breakeven'].includes(t.outcome));
const wins   = TRADES.filter(t => t.outcome === 'win');
const losses = TRADES.filter(t => t.outcome === 'loss');

const wl      = wins.length + losses.length;
const winRate = wl > 0 ? wins.length / wl * 100 : 0;

const recWins  = wins.filter(t=>t.pnl_points>0).map(t=>t.pnl_points);
const recLoss  = losses.filter(t=>t.pnl_points<0).map(t=>Math.abs(t.pnl_points));
const avgWin   = recWins.length ? recWins.reduce((a,b)=>a+b)/recWins.length : 0;
const avgLoss  = recLoss.length ? recLoss.reduce((a,b)=>a+b)/recLoss.length : 0;
const ratio    = avgLoss > 0 ? avgWin / avgLoss : 0;
const netPnl   = TRADES.filter(t=>t.pnl_points!==0).reduce((s,t)=>s+t.pnl_points, 0);
const netGbp   = TRADES.filter(t=>t.pnl_gbp!==0).reduce((s,t)=>s+t.pnl_gbp, 0);
const wViol    = taken.filter(t=>!t.within_window).length;
const fomoN    = TRADES.filter(t=>t.psychology_qa&&t.psychology_qa.fomo_present).length;

// ── Meta ─────────────────────────────────────────────────────────────────────
$('meta-line').textContent =
  TRADES.length+' entries · Last: '+TRADES[TRADES.length-1].date+
  ' · Account: $'+(ACCOUNT_GBP*GBPUSD).toLocaleString('en-US',{minimumFractionDigits:0,maximumFractionDigits:0})+
  ' (£'+ACCOUNT_GBP.toLocaleString()+')'+
  ' · Generated '+new Date().toLocaleDateString('en-GB',{day:'numeric',month:'short',year:'numeric'});

// ── Stats strip ───────────────────────────────────────────────────────────────
const netGbpKnown = netGbp !== 0;
[
  { label:'Total Entries', value:TRADES.length,               cls:'blue' },
  { label:'Taken',         value:taken.length,                cls:'' },
  { label:'Wins',          value:wins.length,                 cls:'green' },
  { label:'Losses',        value:losses.length,               cls:'red' },
  { label:'Win Rate',      value:winRate.toFixed(1)+'%',      cls:winRate>=50?'green':'red' },
  { label:'Net P&L (USD)', value:usdStr(netGbp), cls:netGbp>=0?'green':'red', sub:(netPnl>=0?'+':'')+netPnl.toFixed(1)+' pts' },
  { label:'Window Viols',  value:wViol,                       cls:wViol>0?'red':'green', sub:fomoN+' FOMO entries' },
].forEach(s => {
  $('stats-strip').insertAdjacentHTML('beforeend',
    `<div class="stat-card">
      <div class="label">${s.label}</div>
      <div class="value ${s.cls||''}">${s.value}</div>
      ${s.sub?`<div class="sub">${s.sub}`:''}
    </div>`);
});

// ── Calendar ──────────────────────────────────────────────────────────────────
let calYear = 2026, calMonth = 4;

const byDate = {};
TRADES.forEach(t => { (byDate[t.date]=byDate[t.date]||[]).push(t); });

function dayClass(ts) {
  const o = ts.map(t=>t.outcome);
  if (o.includes('win'))       return 'win';
  if (o.includes('loss'))      return 'loss';
  if (o.includes('breakeven')) return 'be';
  return 'miss';
}

function renderCalendar() {
  const months=['January','February','March','April','May','June','July','August','September','October','November','December'];
  $('month-title').textContent = months[calMonth]+' '+calYear;

  const firstDow   = new Date(calYear, calMonth, 1).getDay();
  const daysInMo   = new Date(calYear, calMonth+1, 0).getDate();
  const todayStr   = new Date().toISOString().slice(0,10);

  // Month summary
  let mT=0,mW=0,mL=0,mP=0,mG=0;
  Object.entries(byDate).forEach(([d,ts]) => {
    const dt = new Date(d);
    if (dt.getFullYear()!==calYear||dt.getMonth()!==calMonth) return;
    ts.forEach(t => {
      if (['win','loss','breakeven'].includes(t.outcome)) mT++;
      if (t.outcome==='win')  mW++;
      if (t.outcome==='loss') mL++;
      mP += t.pnl_points||0;
      mG += t.pnl_gbp||0;
    });
  });
  const mPct = mG !== 0 ? (mG/ACCOUNT_GBP*100) : null;
  $('cal-summary').innerHTML =
    `<span class="t">Taken: ${mT}</span>`+
    `<span class="w">W: ${mW}</span>`+
    `<span class="l">L: ${mL}</span>`+
    `<span style="color:${mP>=0?'#22c55e':'#ef4444'};font-weight:700">${mP>=0?'+':''}${mP.toFixed(1)} pts</span>`+
    (mG?`<span style="color:${mG>=0?'#22c55e':'#ef4444'};font-weight:700">${usdStr(mG)}</span>`:'');

  const headers=['Sun','Mon','Tue','Wed','Thu','Fri','Sat','Wk'];
  let html = headers.map((d,i) =>
    i<7 ? `<div class="cal-day-header">${d}</div>`
        : `<div class="cal-week-header">${d}</div>`
  ).join('');

  // Empty cells for first row
  for (let i=0;i<firstDow;i++) html += `<div class="cal-cell empty"></div>`;

  let col = firstDow;  // column position 0-6
  let wkPts=0, wkGbp=0, wkW=0, wkL=0;

  for (let d=1; d<=daysInMo; d++) {
    const ds = `${calYear}-${String(calMonth+1).padStart(2,'0')}-${String(d).padStart(2,'0')}`;
    const ts = byDate[ds]||[];
    const isToday  = ds===todayStr;
    const hasViol  = ts.some(t=>!t.within_window&&['win','loss','breakeven'].includes(t.outcome));
    const pnl      = ts.reduce((s,t)=>s+(t.pnl_points||0),0);
    const gbp      = ts.reduce((s,t)=>s+(t.pnl_gbp||0),0);

    // Accumulate week totals
    wkPts += pnl; wkGbp += gbp;
    ts.forEach(t => { if(t.outcome==='win') wkW++; if(t.outcome==='loss') wkL++; });

    if (!ts.length) {
      html += `<div class="cal-cell${isToday?' today':''}"><div class="cal-date">${d}</div></div>`;
    } else {
      const cls    = dayClass(ts);
      const grades = [...new Set(ts.map(t=>t.execution_grade))].join('/');
      const models = [...new Set(ts.map(t=>t.entry_model).filter(Boolean))].join('/');
      const pnlStr = pnl!==0?(pnl>0?'+':'')+pnl.toFixed(1)+' pts': cls==='miss'?'MISS':cls.toUpperCase();
      const gbpLine= gbp!==0?`<div style="font-size:9px;color:${gbp>=0?'#22c55e':'#ef4444'}">${usdStr(gbp)}</div>`:'';
      html += `<div class="cal-cell ${cls}${isToday?' today':''}${hasViol?' violation':''}" title="${ds}">
        <div class="cal-date">${d}</div>
        <div class="cal-pnl">${pnlStr}</div>
        ${gbpLine}
        ${models?`<div class="cal-model">${models}</div>`:''}
        <div class="cal-grade">${grades}</div>
      </div>`;
    }

    col++;

    // End of week row (Saturday) or last day of month
    if (col===7 || d===daysInMo) {
      // Fill trailing empty cells if last partial row
      if (d===daysInMo && col<7) {
        for (let i=col;i<7;i++) html += `<div class="cal-cell empty"></div>`;
      }
      // Weekly total cell
      const wkPct  = wkGbp!==0 ? (wkGbp/ACCOUNT_GBP*100) : null;
      const pColor = wkPts>=0?'#22c55e':'#ef4444';
      const gColor = wkGbp>=0?'#22c55e':'#ef4444';
      html += `<div class="cal-week-total">
        <div class="wk-label">Week</div>
        <div class="wk-pts" style="color:${wkPts!==0?pColor:'#374151'}">${wkPts!==0?(wkPts>0?'+':'')+wkPts.toFixed(1):'—'}</div>
        <div class="wk-gbp" style="color:${wkGbp!==0?gColor:'#374151'}">${wkGbp!==0?usdStr(wkGbp):'—'}</div>
        ${wkPct!==null?`<div class="wk-pct">${pctStr(wkPct)}</div>`:''}
      </div>`;
      col=0; wkPts=0; wkGbp=0; wkW=0; wkL=0;
    }
  }

  $('cal-grid').innerHTML = html;
}

window.prevMonth = () => { calMonth--; if(calMonth<0){calMonth=11;calYear--;} renderCalendar(); };
window.nextMonth = () => { calMonth++; if(calMonth>11){calMonth=0;calYear++;} renderCalendar(); };
renderCalendar();

// ── Win Rate doughnut ─────────────────────────────────────────────────────────
Chart.defaults.color = '#9ca3af';
new Chart($('winRateChart'), {
  type:'doughnut',
  data:{ datasets:[{ data:[winRate,100-winRate], backgroundColor:['#3b82f6','#1e2d3d'], borderWidth:0 }] },
  options:{ cutout:'70%', animation:false, plugins:{ legend:{display:false}, tooltip:{enabled:false} } },
  plugins:[{ id:'center', afterDraw(c) {
    const {ctx,chartArea:{left,right,top,bottom}}=c;
    ctx.save(); ctx.font='bold 16px -apple-system,sans-serif';
    ctx.fillStyle='#fff'; ctx.textAlign='center'; ctx.textBaseline='middle';
    ctx.fillText(winRate.toFixed(1)+'%',(left+right)/2,(top+bottom)/2); ctx.restore();
  }}]
});

// ── Ratio bar ─────────────────────────────────────────────────────────────────
$('ratio-val').textContent = ratio>0?ratio.toFixed(2):'—';
$('ratio-val').style.color = ratio>=1?'#22c55e':'#ef4444';
$('ratio-seg').style.width = ((avgWin+avgLoss)>0?avgWin/(avgWin+avgLoss)*100:50)+'%';

// ── Net P&L ───────────────────────────────────────────────────────────────────
const pnlEl = $('net-pnl-val');
pnlEl.textContent = (netPnl>=0?'+':'')+netPnl.toFixed(2)+' pts'+(netGbp?' / '+gbpStr(netGbp):'');
pnlEl.style.color = netPnl>=0?'#22c55e':'#ef4444';

// ── P&L timeline ──────────────────────────────────────────────────────────────
const pnlSeries = TRADES.filter(t=>t.pnl_points!==0).sort((a,b)=>a.date.localeCompare(b.date));
new Chart($('pnlChart'), {
  type:'bar',
  data:{ labels:pnlSeries.map(t=>t.date.slice(5)),
    datasets:[{ data:pnlSeries.map(t=>t.pnl_points),
      backgroundColor:pnlSeries.map(t=>t.pnl_points>0?'rgba(59,130,246,0.75)':'rgba(239,68,68,0.75)'), borderRadius:2 }] },
  options:{ responsive:true, animation:false, plugins:{ legend:{display:false},
    tooltip:{callbacks:{label:ctx=>(ctx.parsed.y>0?'+':'')+ctx.parsed.y.toFixed(2)+' pts'}} },
    scales:{ x:{grid:{color:'#1a2535'},ticks:{color:'#9ca3af',font:{size:9}}},
              y:{grid:{color:'#1a2535'},ticks:{color:'#9ca3af',font:{size:9}},border:{dash:[2,2]}} } }
});

// ── Grade distribution ────────────────────────────────────────────────────────
const grades={'AA+':0,'A':0,'B':0,'Skip':0};
TRADES.forEach(t=>{if(grades[t.execution_grade]!==undefined) grades[t.execution_grade]++;});
new Chart($('gradeChart'), {
  type:'bar',
  data:{ labels:Object.keys(grades), datasets:[{ data:Object.values(grades),
    backgroundColor:['rgba(168,85,247,0.7)','rgba(59,130,246,0.7)','rgba(234,179,8,0.7)','rgba(107,114,128,0.5)'], borderRadius:3 }] },
  options:{ responsive:true, animation:false, plugins:{legend:{display:false}},
    scales:{ x:{grid:{display:false},ticks:{color:'#9ca3af',font:{size:11}}},
              y:{grid:{color:'#1a2535'},ticks:{color:'#9ca3af',font:{size:9},stepSize:1},beginAtZero:true} } }
});

// ── Day of week ───────────────────────────────────────────────────────────────
const dows=['Monday','Tuesday','Wednesday','Thursday','Friday'];
const dowW={},dowL={};
dows.forEach(d=>{dowW[d]=0;dowL[d]=0;});
taken.forEach(t=>{ if(dowW[t.day_of_week]!==undefined){ if(t.outcome==='win')dowW[t.day_of_week]++; if(t.outcome==='loss')dowL[t.day_of_week]++; } });
new Chart($('dowChart'), {
  type:'bar',
  data:{ labels:dows.map(d=>d.slice(0,3)),
    datasets:[
      {label:'Wins',  data:dows.map(d=>dowW[d]),backgroundColor:'rgba(59,130,246,0.7)',borderRadius:3},
      {label:'Losses',data:dows.map(d=>dowL[d]),backgroundColor:'rgba(239,68,68,0.7)', borderRadius:3},
    ]},
  options:{ responsive:true, animation:false,
    plugins:{legend:{labels:{color:'#9ca3af',font:{size:10},boxWidth:10}}},
    scales:{ x:{grid:{display:false},ticks:{color:'#9ca3af'}},
              y:{grid:{color:'#1a2535'},ticks:{color:'#9ca3af',stepSize:1},beginAtZero:true} } }
});

// ── Trade detail modal ────────────────────────────────────────────────────────

function tvImg(url) {
  const m = url.match(/tradingview\.com\/x\/([A-Za-z0-9]+)/);
  if (!m) return null;
  return 'https://s3.tradingview.com/snapshots/'+m[1][0].toLowerCase()+'/'+m[1]+'.png';
}

function yn(v) { return v ? '<span style="color:#22c55e;font-weight:600">&#10003; Yes</span>' : '<span style="color:#ef4444;font-weight:600">&#10007; No</span>'; }
function di(k,v) { return `<div class="det-item"><div class="dk">${k}</div><div class="dv">${v||'&#8212;'}</div></div>`; }
function df(k,v) { return v ? `<div class="det-full"><div class="dk">${k}</div><div class="dv">${v}</div></div>` : ''; }
function dsec(title, inner) { return `<div class="det-section"><h4>${title}</h4>${inner}</div>`; }

function openDetail(idx) {
  const t = TRADES[idx];
  const M = id => document.getElementById(id);

  // Header
  const scottBadge = t.scott_validated ? '<span class="badge" style="background:rgba(34,197,94,0.12);color:#22c55e">&#10003; Scott</span>' : '';
  const wBadge = t.within_window
    ? '<span class="badge" style="background:rgba(34,197,94,0.08);color:#22c55e;font-size:9px">In Window</span>'
    : '<span class="badge" style="background:rgba(239,68,68,0.1);color:#ef4444;font-size:9px">&#9888; Pre/Post Window</span>';
  M('modal-header').innerHTML =
    '<div class="modal-header-left">' +
      '<div class="modal-trade-id">'+t.trade_id+'</div>' +
      '<div class="modal-header-badges">' +
        '<span class="modal-header-title">'+t.instrument+'</span>' +
        '<span class="modal-header-sub">'+t.date+' &middot; '+t.day_of_week+' &middot; '+t.session+'</span>' +
        obadge(t.outcome)+' '+gbadge(t.execution_grade)+' '+wBadge+' '+scottBadge +
      '</div>' +
    '</div>' +
    '<button class="modal-close" onclick="closeTrade()">&#10005;</button>';

  // Charts
  const links = t.chart_links || [];
  let chartHtml = '';
  if (!links.length) {
    chartHtml = '<div class="no-chart-placeholder">No chart saved for this trade</div>';
  } else {
    links.forEach(function(url, j) {
      const img = tvImg(url);
      chartHtml += '<div>' +
        '<div class="chart-item-label"><span>Chart '+(j+1)+'</span>' +
        '<a href="'+url+'" target="_blank" class="chart-link">&#8599; Open</a></div>' +
        (img
          ? '<a href="'+url+'" target="_blank"><img class="modal-chart-img" src="'+img+'" alt="Chart '+(j+1)+'" onerror="imgFallback(this)">'+
            '<div class="no-chart-placeholder" style="display:none">Image unavailable &mdash; <a href="'+url+'" target="_blank" style="color:#3b82f6;margin-left:4px">open in browser</a></div></a>'
          : '<div class="no-chart-placeholder"><a href="'+url+'" target="_blank" style="color:#3b82f6">&#8599; Open chart</a></div>'
        ) + '</div>';
    });
  }
  M('modal-chart-panel').innerHTML = chartHtml;

  // Details
  let det = '';

  if (t.trade_reason) det += dsec('Trade Reason', df('', t.trade_reason));

  det += dsec('Market Context',
    '<div class="det-grid">' +
    di('Mode', t.market_mode) + di('Structure', t.structure) +
    di('Condition', t.market_condition) + di('Weekly POC', t.weekly_poc_position) +
    di('POC Migration', t.poc_migration) + di('4H Structure', t.four_hour_structure) +
    di('Opening Gap', t.opening_gap_present ? 'Yes' : 'No') +
    (t.opening_gap_present ? di('Gap Filled First', yn(t.opening_gap_filled_first)) : '') +
    '</div>');

  det += dsec('Liquidity',
    df('Identified', t.liquidity_identified) +
    '<div class="det-grid" style="margin-top:6px">' +
    di('Swept Before Entry', yn(t.liquidity_swept_before_entry)) +
    di('Sweep Result', t.sweep_result) +
    '</div>');

  det += dsec('Entry',
    '<div class="det-grid">' +
    di('Model', t.entry_model) + di('Tier', t.entry_tier) +
    di('Entry Price', t.entry_price||'&#8212;') + di('Exit Price', t.exit_price||'&#8212;') +
    di('Stop Loss', t.stop_loss||'&#8212;') + di('Take Profit', t.take_profit||'&#8212;') +
    di('Entry Time', t.entry_time) + di('Within Window', yn(t.within_window)) +
    di('Tier Matches Mode', yn(t.entry_tier_matches_mode)) +
    di('Confirmation Close', yn(t.confirmation_close_confirmed)) +
    di('Multi-Level Cleared', yn(t.multi_level_inefficiency_cleared)) +
    '</div>' + df('Trigger Type', t.trigger_type));

  if (['1MCP','2MCP','5MCP'].includes(t.entry_model)) {
    det += dsec('MCP Details',
      '<div class="det-grid">' +
      di('Formation Time', t.mcp_formation_time_minutes ? t.mcp_formation_time_minutes+' min' : '&#8212;') +
      di('Reversal &ge;50%', yn(t.mcp_reversal_reaches_50pct)) +
      di('Screamed Instantly', yn(t.mcp_screamed_instantly)) +
      '</div>');
  }

  const stack = (t.confluence_stack||[]).map(function(c){ return '<span class="conf-tag">'+c+'</span>'; }).join('');
  det += dsec('Confluence & Confidence',
    '<div style="margin-bottom:8px">'+(stack||'&#8212;')+'</div>' +
    '<div class="det-grid">' +
    di('Confidence', t.confidence_level ? t.confidence_level+'/10' : '&#8212;') +
    di('Cycle Position', t.cycle_position) +
    '</div>');

  const pColor = t.pnl_points > 0 ? '#22c55e' : t.pnl_points < 0 ? '#ef4444' : '#9ca3af';
  det += dsec('Risk, Management & Outcome',
    '<div class="det-grid">' +
    di('pts at Risk', t.pts_at_risk ? t.pts_at_risk+' pts' : '&#8212;') +
    di('R:R (planned)', t.rr_ratio||'&#8212;') +
    di('Position Size', t.position_size_lots||'&#8212;') +
    di('BE Moved', yn(t.be_moved)) +
    di('Trail Used', yn(t.trail_used)) +
    di('Manual Close', yn(t.manual_close)) +
    di('Result', obadge(t.outcome)) +
    di('Actual R:R', t.actual_rr_achieved||'&#8212;') +
    di('P&L Points', t.pnl_points ? '<span style="color:'+pColor+';font-weight:700">'+(t.pnl_points>0?'+':'')+t.pnl_points+' pts</span>' : '&#8212;') +
    di('P&L USD', t.pnl_gbp ? '<span style="color:'+(t.pnl_gbp>0?'#22c55e':'#ef4444')+';font-weight:700">'+usdStr(t.pnl_gbp)+'</span>' : '&#8212;') +
    '</div>');

  det += dsec('Post-Trade Assessment',
    '<div class="det-grid">' +
    di('Grade', gbadge(t.execution_grade)) +
    di('Session Rating', t.session_rating ? t.session_rating+'/10' : '&#8212;') +
    di('Analysis Correct', yn(t.analysis_correct)) +
    di('Entry Timing', yn(t.entry_timing_correct)) +
    di('Management', yn(t.management_correct)) +
    di('Scott Validated', yn(t.scott_validated)) +
    '</div>' + df('Lessons Learned', t.lessons_learned));

  const qa = t.psychology_qa || {};
  const isMiss = t.outcome === 'no_trade';
  det += dsec('Psychology',
    df('Summary', t.psychology) +
    '<div class="det-grid" style="margin-top:8px">' +
    (isMiss
      ? di('At Screen', yn(qa.at_screen)) + di('Alerts Set', yn(qa.alerts_set)) +
        di('Would Have Taken', yn(qa.would_have_taken)) + di('FOMO', yn(qa.fomo_present))
      : di('Calm Scale', qa.pre_entry_calm_scale ? qa.pre_entry_calm_scale+'/10' : '&#8212;') +
        di('FOMO', yn(qa.fomo_present)) + di('Followed Plan', yn(qa.followed_plan)) +
        di('Revenge Urge', yn(qa.revenge_trade_urge))
    ) + '</div>' +
    (isMiss && qa.missed_reason ? df('Missed Reason', qa.missed_reason) : '') +
    (!isMiss && qa.trading_for_right_reason ? df('Right Reason?', qa.trading_for_right_reason) : '') +
    (qa.post_trade_emotion ? df('Post-Trade Emotion', qa.post_trade_emotion) : '') +
    (qa.key_lesson_emotion ? df('Key Emotional Lesson', qa.key_lesson_emotion) : ''));

  M('modal-details-panel').innerHTML = det;
  M('tradeModal').classList.add('open');
}

function closeTrade() {
  document.getElementById('tradeModal').classList.remove('open');
}

function imgFallback(img) {
  img.style.display = 'none';
  if (img.nextElementSibling) img.nextElementSibling.style.display = 'flex';
}

function closeModal(e) {
  if (e.target === document.getElementById('tradeModal')) closeTrade();
}

document.addEventListener('keydown', function(e) {
  if (e.key === 'Escape') closeTrade();
});

// ── Trade table with weekly summary rows ──────────────────────────────────────
function obadge(o){const m={win:'badge-win',loss:'badge-loss',no_trade:'badge-miss',breakeven:'badge-be'};const l={win:'WIN',loss:'LOSS',no_trade:'MISS',breakeven:'BE'};return `<span class="badge ${m[o]||''}">${l[o]||o}</span>`;}
function gbadge(g){const m={'AA+':'badge-aa','A':'badge-a','B':'badge-b','Skip':'badge-miss'};return `<span class="badge ${m[g]||''}">${g}</span>`;}
function fp(v){return(v&&v!==0)?`<span class="mono">${v}</span>`:'<span style="color:#374151">—</span>';}
function tipCell(text){const t=(text||'').trim();return`<td class="tip-cell" data-tip="${t||'—'}">${t?t.slice(0,22)+(t.length>22?'…':''):'—'}</td>`;}

// Tooltip position tracking
document.addEventListener('mousemove', e => {
  document.querySelectorAll('.tip-cell').forEach(el => {
    el.style.setProperty('--tip-x', Math.min(e.clientX+12, window.innerWidth-320)+'px');
    el.style.setProperty('--tip-y', Math.max(e.clientY-120, 8)+'px');
  });
});

// Group by ISO week
const sorted    = [...TRADES].sort((a,b)=>a.date.localeCompare(b.date));
const weekGroups = {};
sorted.forEach(t => {
  const wk = isoWeek(t.date);
  (weekGroups[wk] = weekGroups[wk]||[]).push(t);
});

let html = '';

Object.entries(weekGroups).sort(([a],[b])=>a.localeCompare(b)).forEach(([wk, wkTrades]) => {
  // Trade rows for this week
  wkTrades.forEach(t => {
    const i = TRADES.indexOf(t);
    const charts = (t.chart_links||[]).map((l,j)=>
      `<a href="${l}" target="_blank" class="chart-link">&#128247; Chart ${j+1}</a>`).join(' ');
    const pnlCol = t.pnl_points!==0
      ?`<span style="font-weight:700;color:${t.pnl_points>0?'#22c55e':'#ef4444'}">${t.pnl_points>0?'+':''}${t.pnl_points.toFixed(2)}</span>`
      :'<span style="color:#374151">—</span>';
    const gbpCol = t.pnl_gbp && t.pnl_gbp!==0
      ?`<span style="font-weight:700;color:${t.pnl_gbp>0?'#22c55e':'#ef4444'}">${usdStr(t.pnl_gbp)}</span>`
      :'<span style="color:#374151">—</span>';
    const pctCol = t.pnl_gbp && t.pnl_gbp!==0
      ?`<span style="color:${t.pnl_gbp>0?'#22c55e':'#ef4444'}">${pctStr(t.pnl_gbp/ACCOUNT_GBP*100)}</span>`
      :'<span style="color:#374151">—</span>';
    const rrCol  = t.rr_ratio
      ?`<span style="color:${t.rr_ratio>=2?'#22c55e':'#ef4444'}">${t.rr_ratio}</span>`
      :'<span style="color:#374151">—</span>';
    html += `<tr>
      <td><span class="trade-id-link" onclick="openDetail(${i})">${t.trade_id.replace(/^\d{4}-\d{2}-\d{2}-/,'')}</span></td>
      <td style="color:#9ca3af">${t.date}</td>
      <td><b style="color:#e2e8f0">${t.instrument}</b></td>
      <td style="color:#cbd5e1">${t.day_of_week.slice(0,3)}</td>
      <td style="color:#cbd5e1">${t.market_mode}</td>
      <td>${obadge(t.outcome)}</td>
      <td>${gbadge(t.execution_grade)}</td>
      <td class="${t.within_window?'w-ok':'w-bad'}">${t.within_window?'✓':'✗'}</td>
      <td>${fp(t.entry_price)}</td>
      <td>${fp(t.stop_loss)}</td>
      <td>${fp(t.take_profit)}</td>
      <td>${rrCol}</td>
      <td>${pnlCol}</td>
      <td>${gbpCol}</td>
      <td>${pctCol}</td>
      <td style="color:#9ca3af;font-size:10px">${t.entry_model||'—'}</td>
      ${tipCell(t.trade_reason)}
      ${tipCell(t.lessons_learned)}
      ${tipCell(t.psychology)}
      <td>${charts||'<span style="color:#374151">—</span>'}</td>
    </tr>`;
  });

  // Weekly summary row
  const wkPts = wkTrades.reduce((s,t)=>s+(t.pnl_points||0),0);
  const wkGbp = wkTrades.reduce((s,t)=>s+(t.pnl_gbp||0),0);
  const wkPct = wkGbp!==0 ? wkGbp/ACCOUNT_GBP*100 : null;
  const wkW   = wkTrades.filter(t=>t.outcome==='win').length;
  const wkL   = wkTrades.filter(t=>t.outcome==='loss').length;
  const wkT   = wkTrades.filter(t=>['win','loss','breakeven'].includes(t.outcome)).length;
  const pColor= wkPts>=0?'#22c55e':'#ef4444';
  const gColor= wkGbp>=0?'#22c55e':'#ef4444';

  html += `<tr class="week-row">
    <td colspan="2">${wk}</td>
    <td colspan="3">Taken: ${wkT} &nbsp;|&nbsp; W:${wkW} L:${wkL}</td>
    <td colspan="2"></td>
    <td colspan="5"></td>
    <td style="color:${wkPts!==0?pColor:'#374151'}">${wkPts!==0?ptsStr(wkPts):'—'}</td>
    <td style="color:${wkGbp!==0?gColor:'#374151'}">${wkGbp!==0?usdStr(wkGbp):'—'}</td>
    <td style="color:${wkPct!==null?gColor:'#374151'}">${wkPct!==null?pctStr(wkPct):'—'}</td>
    <td colspan="5"></td>
  </tr>`;
});

$('trade-tbody').innerHTML = html;
</script>
</body>
</html>'''

OUT_PATH.write_text(
    HTML
    .replace('__TRADE_DATA__', data)
    .replace('__ACCOUNT_GBP__', str(ACCOUNT_GBP))
    .replace('__GBPUSD__', str(round(GBPUSD, 4)))
)
webbrowser.open_new_tab(f'file://{OUT_PATH.resolve()}')
print(f'Dashboard written → {OUT_PATH}')
