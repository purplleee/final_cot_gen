"""
CoT Generation Run Analyzer
Parses log files + JSONL outputs from all test runs and produces a dashboard.

Runs configured:
  - OmniDroid 50 imbalanced
  - CICMalDroid 1k (imbalanced)
  - CICMalDroid 3k (imbalanced)
  - CICMalDroid 5k (balanced)
"""

import re
import json
import io
from pathlib import Path
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

# ──────────────────────────────────────────────
# CONFIG — edit paths if needed
# ──────────────────────────────────────────────
RUNS = {
    "OmniDroid\n50 samples": {
        "log":   r"E:\PFE_CoT_outputs_omnidroid\data\test_50_imbalance\50_imbalance_log.txt",
        "jsonl": r"E:\PFE_CoT_outputs_omnidroid\data\test_50_imbalance\groq_llama33_shard2.jsonl",
        "color": "#4ecdc4",
        "dataset": "OmniDroid",
    },
    "CICMalDroid\n1k events": {
        "log":   r"E:\PFE_CoT_outputs\data\1k_cot\cici_cot_gen_log_1k.txt",
        "jsonl": r"E:\PFE_CoT_outputs\data\1k_cot\groq_llama33_shard2.jsonl",
        "color": "#ff6b6b",
        "dataset": "CICMalDroid",
    },
    "CICMalDroid\n3k events": {
        "log":   r"E:\PFE_CoT_outputs\data\3k_cot\cicidrod_cot_gen_log_3k.txt",
        "jsonl": r"E:\PFE_CoT_outputs\data\3k_cot\groq_llama33_shard2.jsonl",
        "color": "#ffa94d",
        "dataset": "CICMalDroid",
    },
    "CICMalDroid\n5k balanced": {
        "log":   r"E:\PFE_CoT_outputs\data\5k_cot_balanced\5k_cot_log.txt",
        "jsonl": r"E:\PFE_CoT_outputs\data\5k_cot_balanced\groq_llama33_shard2.jsonl",
        "color": "#f06595",
        "dataset": "CICMalDroid",
    },
}

OUTPUT_DIR     = Path(r"E:\Test_CoT_outputs")
ALL_CATEGORIES = ["SMS", "Adware", "Banking", "Benign", "Riskware"]

# ──────────────────────────────────────────────
# PARSERS
# ──────────────────────────────────────────────

def parse_log(log_path):
    samples    = {}
    failed_ids = set()
    try:
        with open(log_path, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"[WARN] Log not found: {log_path}")
        return {}

    for line in lines:
        sid_match = re.search(r'sample_id=(\d+)', line)
        if not sid_match:
            continue
        sid = sid_match.group(1)
        if line.startswith('[OK]'):
            verdict = re.search(r'verdict=(\w+)', line)
            if verdict:
                samples[sid] = {'outcome': 'OK', 'predicted': verdict.group(1), 'expected': verdict.group(1)}
        elif line.startswith('[FAILED]'):
            failed_ids.add(sid)
        elif line.startswith('[MISMATCH]') and sid not in samples:
            pred = re.search(r'predicted=(\w+)', line)
            exp  = re.search(r'expected=(\w+)', line)
            if pred and exp:
                samples[sid] = {'outcome': 'MISMATCH', 'predicted': pred.group(1), 'expected': exp.group(1)}

    for sid in failed_ids:
        if sid in samples:
            samples[sid]['outcome'] = 'FAILED'
    return samples


def parse_jsonl(jsonl_path):
    records = []
    try:
        with open(jsonl_path, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    except FileNotFoundError:
        print(f"[WARN] JSONL not found: {jsonl_path}")
    return records


def compute_stats(samples):
    if not samples:
        return None
    total    = len(samples)
    ok       = sum(1 for s in samples.values() if s['outcome'] == 'OK')
    failed   = sum(1 for s in samples.values() if s['outcome'] == 'FAILED')
    mismatch = total - ok - failed
    per_class = defaultdict(lambda: {'correct': 0, 'total': 0})
    for s in samples.values():
        per_class[s['expected']]['total'] += 1
        if s['outcome'] == 'OK':
            per_class[s['expected']]['correct'] += 1
    pred_dist = Counter(s['predicted'] for s in samples.values())
    confusion  = defaultdict(Counter)
    for s in samples.values():
        confusion[s['expected']][s['predicted']] += 1
    return {
        'total': total, 'ok': ok, 'failed': failed, 'mismatch': mismatch,
        'agreement_rate': ok / total * 100,
        'per_class': dict(per_class),
        'pred_dist': pred_dist,
        'confusion': confusion,
    }

# ──────────────────────────────────────────────
# LOAD ALL RUNS
# ──────────────────────────────────────────────

run_stats = {}
for run_name, cfg in RUNS.items():
    samples = parse_log(cfg['log'])
    if not samples:
        records = parse_jsonl(cfg['jsonl'])
        samples = {}
        for r in records:
            sid       = str(r.get('sample_id', r.get('original_id', '')))
            predicted = r.get('predicted_verdict', r.get('verdict', ''))
            expected  = r.get('true_label', r.get('expected_verdict', ''))
            if predicted and expected:
                is_ok = predicted.strip().lower() == expected.strip().lower()
                samples[sid] = {
                    'outcome': 'OK' if is_ok else 'FAILED',
                    'predicted': predicted,
                    'expected': expected,
                }
    stats = compute_stats(samples)
    if stats:
        run_stats[run_name] = {'stats': stats, 'color': cfg['color'], 'dataset': cfg['dataset']}
        print(f"[OK] {run_name.replace(chr(10), ' ')}: {stats['total']} samples, {stats['agreement_rate']:.1f}% agreement")
    else:
        print(f"[SKIP] {run_name.replace(chr(10), ' ')}: no data found")

if not run_stats:
    print("No data loaded. Check your file paths.")
    exit(1)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────
# STYLE HELPERS
# ──────────────────────────────────────────────
LABEL_COLOR = '#e6edf3'
GRID_COLOR  = '#21262d'
BG_PANEL    = '#161b22'
BG_FIG      = '#0d1117'

run_names  = list(run_stats.keys())
colors     = [run_stats[r]['color'] for r in run_names]
agreements = [run_stats[r]['stats']['agreement_rate'] for r in run_names]
n_runs     = len(run_names)
n_cats     = len(ALL_CATEGORIES)
width      = 0.8 / n_runs
xs         = np.arange(n_cats)

def style_ax(ax):
    ax.set_facecolor(BG_PANEL)
    ax.tick_params(colors=LABEL_COLOR)
    ax.spines[:].set_color(GRID_COLOR)
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)

def save_fig(fig, filename):
    path = OUTPUT_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f"  Saved: {filename}")
    return path

# ──────────────────────────────────────────────
# BUILD & SAVE INDIVIDUAL PLOTS
# ──────────────────────────────────────────────
saved_figs = []   # keep references for dashboard

# 1. Overall Agreement Rate
fig1, ax1 = plt.subplots(figsize=(9, 5))
fig1.patch.set_facecolor(BG_FIG)
style_ax(ax1)
bars = ax1.bar(range(n_runs), agreements, color=colors, width=0.55, zorder=3)
for bar, val in zip(bars, agreements):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f'{val:.1f}%', ha='center', va='bottom', color=LABEL_COLOR, fontsize=10, fontweight='bold')
ax1.set_xticks(range(n_runs))
ax1.set_xticklabels([r.replace('\n', ' ') for r in run_names], fontsize=9, color=LABEL_COLOR, rotation=15, ha='right')
ax1.set_ylabel('Agreement Rate (%)', color=LABEL_COLOR, fontsize=10)
ax1.set_title('Overall Agreement Rate', color=LABEL_COLOR, fontsize=13, fontweight='bold', pad=12)
ax1.set_ylim(0, 110)
save_fig(fig1, "01_agreement_rate.png")
saved_figs.append(fig1)

# 2. Outcome Breakdown
fig2, ax2 = plt.subplots(figsize=(9, 5))
fig2.patch.set_facecolor(BG_FIG)
style_ax(ax2)
ok_vals   = [run_stats[r]['stats']['ok']       / run_stats[r]['stats']['total'] * 100 for r in run_names]
mm_vals   = [run_stats[r]['stats']['mismatch'] / run_stats[r]['stats']['total'] * 100 for r in run_names]
fail_vals = [run_stats[r]['stats']['failed']   / run_stats[r]['stats']['total'] * 100 for r in run_names]
ax2.bar(range(n_runs), ok_vals,   color='#2ea043', label='OK',       zorder=3)
ax2.bar(range(n_runs), mm_vals,   color='#f0a500', label='Mismatch', bottom=ok_vals, zorder=3)
ax2.bar(range(n_runs), fail_vals, color='#da3633', label='Failed',
        bottom=[o+m for o, m in zip(ok_vals, mm_vals)], zorder=3)
ax2.set_xticks(range(n_runs))
ax2.set_xticklabels([r.replace('\n', ' ') for r in run_names], fontsize=9, color=LABEL_COLOR, rotation=15, ha='right')
ax2.set_ylabel('%', color=LABEL_COLOR, fontsize=10)
ax2.set_title('Outcome Breakdown', color=LABEL_COLOR, fontsize=13, fontweight='bold', pad=12)
ax2.set_ylim(0, 110)
ax2.legend(fontsize=9, facecolor=BG_PANEL, labelcolor=LABEL_COLOR, edgecolor=GRID_COLOR)
save_fig(fig2, "02_outcome_breakdown.png")
saved_figs.append(fig2)

# 3. Per-Class Accuracy
fig3, ax3 = plt.subplots(figsize=(11, 5))
fig3.patch.set_facecolor(BG_FIG)
style_ax(ax3)
for i, (run_name, color) in enumerate(zip(run_names, colors)):
    pc   = run_stats[run_name]['stats']['per_class']
    accs = [pc[cat]['correct'] / pc[cat]['total'] * 100 if cat in pc and pc[cat]['total'] > 0 else 0
            for cat in ALL_CATEGORIES]
    offset = (i - n_runs/2 + 0.5) * width
    ax3.bar(xs + offset, accs, width=width*0.9, color=color,
            label=run_name.replace('\n', ' '), zorder=3, alpha=0.9)
ax3.set_xticks(xs)
ax3.set_xticklabels(ALL_CATEGORIES, color=LABEL_COLOR, fontsize=10)
ax3.set_ylabel('Accuracy (%)', color=LABEL_COLOR, fontsize=10)
ax3.set_title('Per-Class Accuracy', color=LABEL_COLOR, fontsize=13, fontweight='bold', pad=12)
ax3.set_ylim(0, 115)
ax3.legend(fontsize=8, facecolor=BG_PANEL, labelcolor=LABEL_COLOR, edgecolor=GRID_COLOR)
save_fig(fig3, "03_per_class_accuracy.png")
saved_figs.append(fig3)

# 4-7. Confusion Matrices
for idx, run_name in enumerate(run_names):
    stats     = run_stats[run_name]['stats']
    matrix    = np.zeros((n_cats, n_cats))
    for i, exp in enumerate(ALL_CATEGORIES):
        for j, pred in enumerate(ALL_CATEGORIES):
            matrix[i][j] = stats['confusion'][exp][pred]
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    matrix_norm = matrix / row_sums * 100

    fig_c, ax_c = plt.subplots(figsize=(6, 5))
    fig_c.patch.set_facecolor(BG_FIG)
    ax_c.set_facecolor(BG_PANEL)
    ax_c.imshow(matrix_norm, cmap='YlOrRd', vmin=0, vmax=100, aspect='auto')
    ax_c.set_xticks(range(n_cats))
    ax_c.set_yticks(range(n_cats))
    ax_c.set_xticklabels(ALL_CATEGORIES, color=LABEL_COLOR, fontsize=8, rotation=30, ha='right')
    ax_c.set_yticklabels(ALL_CATEGORIES, color=LABEL_COLOR, fontsize=8)
    ax_c.set_xlabel('Predicted', color=LABEL_COLOR, fontsize=9)
    ax_c.set_ylabel('Expected', color=LABEL_COLOR, fontsize=9)
    ax_c.set_title(f'Confusion Matrix: {run_name.replace(chr(10), " ")}\n(n={stats["total"]})',
                   color=LABEL_COLOR, fontsize=11, fontweight='bold', pad=8)
    for i in range(n_cats):
        for j in range(n_cats):
            val = matrix_norm[i][j]
            if val > 0:
                text_color = 'black' if val > 55 else LABEL_COLOR
                ax_c.text(j, i, f'{val:.0f}%', ha='center', va='center',
                          fontsize=8, color=text_color, fontweight='bold')
    ax_c.tick_params(colors=LABEL_COLOR)
    ax_c.spines[:].set_color(GRID_COLOR)
    safe_name = run_name.replace('\n', '_').replace(' ', '_')
    save_fig(fig_c, f"0{4+idx}_confusion_{safe_name}.png")
    saved_figs.append(fig_c)

# 8. Prediction Distribution
fig8, ax8 = plt.subplots(figsize=(11, 5))
fig8.patch.set_facecolor(BG_FIG)
style_ax(ax8)
for i, (run_name, color) in enumerate(zip(run_names, colors)):
    pred_dist = run_stats[run_name]['stats']['pred_dist']
    total     = run_stats[run_name]['stats']['total']
    pcts      = [pred_dist.get(cat, 0) / total * 100 for cat in ALL_CATEGORIES]
    offset    = (i - n_runs/2 + 0.5) * width
    ax8.bar(xs + offset, pcts, width=width*0.9, color=color,
            label=run_name.replace('\n', ' '), zorder=3, alpha=0.9)
ax8.set_xticks(xs)
ax8.set_xticklabels(ALL_CATEGORIES, color=LABEL_COLOR, fontsize=10)
ax8.set_ylabel('% of all predictions', color=LABEL_COLOR, fontsize=10)
ax8.set_title('Prediction Distribution (class collapse check)',
              color=LABEL_COLOR, fontsize=13, fontweight='bold', pad=12)
ax8.legend(fontsize=8, facecolor=BG_PANEL, labelcolor=LABEL_COLOR, edgecolor=GRID_COLOR)
save_fig(fig8, "08_prediction_distribution.png")
saved_figs.append(fig8)

# 9. Summary Table
fig9, ax9 = plt.subplots(figsize=(10, 3))
fig9.patch.set_facecolor(BG_FIG)
ax9.set_facecolor(BG_PANEL)
ax9.axis('off')
table_data = [['Run', 'N', 'OK%', 'Mismatch%', 'Failed%', 'Agree%']]
for run_name in run_names:
    s = run_stats[run_name]['stats']
    table_data.append([
        run_name.replace('\n', ' '),
        str(s['total']),
        f"{s['ok']/s['total']*100:.1f}%",
        f"{s['mismatch']/s['total']*100:.1f}%",
        f"{s['failed']/s['total']*100:.1f}%",
        f"{s['agreement_rate']:.1f}%",
    ])
table = ax9.table(cellText=table_data[1:], colLabels=table_data[0],
                  loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2.2)
for (row, col), cell in table.get_celld().items():
    cell.set_facecolor('#1c2128' if row % 2 == 0 else BG_PANEL)
    cell.set_text_props(color=LABEL_COLOR)
    cell.set_edgecolor(GRID_COLOR)
    if row == 0:
        cell.set_facecolor('#30363d')
        cell.set_text_props(color=LABEL_COLOR, fontweight='bold')
ax9.set_title('Summary Table', color=LABEL_COLOR, fontsize=13, fontweight='bold', pad=12)
save_fig(fig9, "09_summary_table.png")
saved_figs.append(fig9)

# ──────────────────────────────────────────────
# FULL DASHBOARD (3x3 composite)
# ──────────────────────────────────────────────
print("\nBuilding full dashboard...")
fig_all = plt.figure(figsize=(22, 18))
fig_all.patch.set_facecolor(BG_FIG)

for i, src_fig in enumerate(saved_figs, start=1):
    src_fig.canvas.draw()
    buf = io.BytesIO()
    src_fig.savefig(buf, format='png', dpi=100, bbox_inches='tight',
                    facecolor=src_fig.get_facecolor())
    buf.seek(0)
    img = mpimg.imread(buf)
    ax_dst = fig_all.add_subplot(3, 3, i)
    ax_dst.imshow(img, aspect='auto')
    ax_dst.axis('off')

fig_all.suptitle('CoT Generation Quality Dashboard — CICMalDroid vs OmniDroid',
                 color=LABEL_COLOR, fontsize=16, fontweight='bold', y=0.99)
plt.tight_layout(rect=[0, 0, 1, 0.97])
save_fig(fig_all, "00_cot_dashboard_full.png")

print(f"\nDone. All plots saved to: {OUTPUT_DIR}")
plt.show()