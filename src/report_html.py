import io, base64, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _fig_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=160, bbox_inches='tight')
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('ascii')


def build_html_report(out_dir: str, df1: pd.DataFrame, preds: pd.DataFrame, trades: pd.DataFrame, equity: pd.DataFrame, metrics: dict):
    # Basic plots
    images = {}

    if len(equity):
        eq = equity.copy()
        eq['time']=pd.to_datetime(eq['time'])
        eq=eq.sort_values('time')
        eq['equity']=pd.to_numeric(eq['equity'], errors='coerce').dropna()
        peak = eq['equity'].cummax()
        dd = eq['equity']/peak - 1

        fig=plt.figure(figsize=(10,3.2))
        ax=fig.add_subplot(111)
        ax.plot(eq['time'], eq['equity'], lw=1)
        ax.set_title('Equity curve (non-overlapping trades)')
        ax.grid(True, alpha=0.3)
        images['equity']=_fig_b64(fig)

        fig=plt.figure(figsize=(10,2.6))
        ax=fig.add_subplot(111)
        ax.plot(eq['time'], dd, lw=1, color='tab:red')
        ax.set_title('Drawdown')
        ax.set_ylim(-1.05, 0.05)
        ax.grid(True, alpha=0.3)
        images['dd']=_fig_b64(fig)

    if len(trades):
        tr=trades.copy()
        tr['time']=pd.to_datetime(tr['time'])
        tr=tr.sort_values('time')
        tr['ret']=pd.to_numeric(tr['ret'], errors='coerce')
        tr=tr.dropna(subset=['ret'])

        monthly = tr.set_index('time')['ret'].resample('MS').sum()
        fig=plt.figure(figsize=(10,3.0))
        ax=fig.add_subplot(111)
        ax.bar(monthly.index, monthly.values, width=20, color=['tab:red' if x<0 else 'tab:blue' for x in monthly.values])
        ax.set_title('Monthly sum of returns')
        ax.axhline(0, color='black', alpha=0.3)
        ax.grid(True, axis='y', alpha=0.3)
        images['monthly']=_fig_b64(fig)

        fig=plt.figure(figsize=(10,3.0))
        ax=fig.add_subplot(111)
        ax.hist(tr['ret'].values, bins=120, color='tab:purple', alpha=0.85)
        ax.set_title('Per-trade return distribution')
        ax.grid(True, alpha=0.3)
        q1, q99 = tr['ret'].quantile(0.01), tr['ret'].quantile(0.99)
        ax.set_xlim(q1, q99)
        images['rethist']=_fig_b64(fig)

    # KPIs
    kpi={}
    if len(trades):
        kpi['trades']=int(len(trades))
        kpi['win_rate']=float((pd.to_numeric(trades['ret'], errors='coerce')>0).mean())
        kpi['mean_ret']=float(pd.to_numeric(trades['ret'], errors='coerce').mean())
    if len(equity):
        kpi['equity_end']=float(pd.to_numeric(equity['equity'], errors='coerce').dropna().iloc[-1])

    html = f"""<!doctype html>
<html>
<head>
  <meta charset='utf-8'/>
  <meta name='viewport' content='width=device-width,initial-scale=1'/>
  <title>LPM v2 Report</title>
  <style>
    body {{ font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial; margin: 24px; color:#111; }}
    .grid {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 16px; }}
    .card {{ border:1px solid #ddd; border-radius: 12px; padding: 14px; background:#fff; }}
    img {{ max-width: 100%; height: auto; border-radius: 10px; border:1px solid #eee; }}
    code {{ background:#f6f8fa; padding:2px 6px; border-radius:6px; }}
    pre {{ background:#f6f8fa; padding: 10px; border-radius:10px; overflow:auto; }}
  </style>
</head>
<body>
  <h1>LPM v2 — Walk-forward + Purging + FD</h1>
  <p>Outputs are in <code>{out_dir}</code>.</p>

  <div class='card'>
    <h2>KPIs</h2>
    <pre>{json.dumps(kpi, indent=2)}</pre>
  </div>

  <div class='card'>
    <h2>Model metrics</h2>
    <pre>{json.dumps(metrics, indent=2)}</pre>
  </div>

  <div class='grid'>
    {"<div class='card'><h3>Equity</h3><img src='data:image/png;base64," + images.get('equity','') + "'/></div>" if images.get('equity') else ''}
    {"<div class='card'><h3>Drawdown</h3><img src='data:image/png;base64," + images.get('dd','') + "'/></div>" if images.get('dd') else ''}
    {"<div class='card'><h3>Monthly</h3><img src='data:image/png;base64," + images.get('monthly','') + "'/></div>" if images.get('monthly') else ''}
    {"<div class='card'><h3>Return dist</h3><img src='data:image/png;base64," + images.get('rethist','') + "'/></div>" if images.get('rethist') else ''}
  </div>
</body>
</html>"""

    return html
