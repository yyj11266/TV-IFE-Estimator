import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib import font_manager
from pathlib import Path
import unicodedata


def _resolve_font_family() -> list[str]:
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    preferred_fonts = [
        "PingFang SC",
        "Hiragino Sans GB",
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "Arial Unicode MS",
        "Heiti TC",
        "Songti SC",
    ]
    resolved = [font for font in preferred_fonts if font in available_fonts]
    fallback_fonts = ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"]
    resolved.extend(font for font in fallback_fonts if font not in resolved)
    return resolved


def _display_width(text: str) -> int:
    return sum(2 if unicodedata.east_asian_width(ch) in {"F", "W"} else 1 for ch in text)


def _wrap_label(text: str, max_width: int = 18, max_lines: int = 2) -> str:
    text = str(text).strip()
    if not text:
        return text

    chunks: list[str] = []
    remaining = text
    while remaining and len(chunks) < max_lines:
        width = 0
        cut = 0
        for idx, ch in enumerate(remaining):
            ch_width = 2 if unicodedata.east_asian_width(ch) in {"F", "W"} else 1
            if width + ch_width > max_width:
                break
            width += ch_width
            cut = idx + 1

        if cut == 0:
            cut = 1

        chunks.append(remaining[:cut].rstrip())
        remaining = remaining[cut:].lstrip()

    if remaining:
        last = chunks[-1].rstrip()
        while last and _display_width(last) >= max_width:
            last = last[:-1].rstrip()
        chunks[-1] = f"{last}…"

    return "\n".join(chunk for chunk in chunks if chunk)


def _format_family_tick_label(family_id: str, max_width: int = 14) -> str:
    label = str(family_id).split("||")[-1].strip()
    return _wrap_label(label, max_width=max_width, max_lines=1)


# Setup styling for premium feel
plt.style.use('seaborn-v0_8-whitegrid')
matplotlib.rcParams.update({
    "figure.facecolor": "#ffffff",
    "axes.facecolor": "#ffffff",
    "grid.color": "#e0e0e0",
    "axes.labelcolor": "#333333",
    "text.color": "#333333",
    "font.sans-serif": _resolve_font_family(),
    "font.size": 11,
    "axes.unicode_minus": False,
})

def format_month(m):
    s = str(m)
    return f"{s[:4]}-{s[4:]}" if len(s) == 6 else s

def generate_waterfall_plot(
    panel_path: Path,
    eval_path: Path,
    output_path: Path,
    top_n: int = 12,
    model_name: str | None = None,
    sort_criterion: str = 'sales' # 'sales' or 'error'
):
    """
    Generate a 3D Ridgeline / Waterfall plot comparing actuals to predictions.
    panel_path: outputs/family_month_panel.csv
    eval_path: outputs/forecast_eval.csv
    """
    df_panel = pd.read_csv(panel_path)
    df_eval = pd.read_csv(eval_path)
    
    # Default to the most advanced model if None provided
    models_avail = df_eval['model_name'].unique()
    if model_name is None:
        if "tv_ife_predictive_augmented" in models_avail:
            model_name = "tv_ife_predictive_augmented"
        else:
            model_name = models_avail[-1]
            
    df_pred = df_eval[df_eval['model_name'] == model_name].copy()
    
    # 1. Select top N products
    if sort_criterion == 'sales':
        top_families = df_panel.groupby('family_id')['sales'].sum().nlargest(top_n).index.tolist()
    else:  # 'error'
        # Group by family and sum the absolute errors
        top_families = df_pred.groupby('family_id')['abs_err'].sum().nlargest(top_n).index.tolist()
        
    df_panel_top = df_panel[df_panel['family_id'].isin(top_families)].copy()
    df_pred_top = df_pred[df_pred['family_id'].isin(top_families)].copy()
    
    # Get universe of months
    all_months = sorted(list(set(df_panel['month'].unique()) | set(df_pred['target_month'].unique())))
    month_to_x = {m: i for i, m in enumerate(all_months)}
    
    fig = plt.figure(figsize=(18, 12), dpi=200)
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=24, azim=-58)  # Open the label side a bit more to reduce overlap.
    
    ax.set_facecolor('#ffffff')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(color='#f0f0f0', linestyle='-', linewidth=0.5)

    verts_actual = []
    verts_pred = []
    
    # Reverse so top selling is in the back or front? 
    # Usually putting rank 1 in the back is better so smaller ones don't block it, or vice versa.
    # Let's put rank 1 in the back visually. The y-axis will be inverted visually if needed.
    y_step = 1.5
    y_positions = np.arange(top_n) * y_step
    
    # Create colormaps for a gradient effect across products
    cmap_base = matplotlib.colormaps['Blues']
    cmap_pred = matplotlib.colormaps['OrRd']

    # We will compute smoothed versions to make it look like density/continuous curves (like the image)
    for i, family in enumerate(reversed(top_families)):
        # -- Actuals --
        f_panel = df_panel_top[df_panel_top['family_id'] == family].sort_values('month')
        x_act = np.array([month_to_x[m] for m in f_panel['month']])
        z_act = f_panel['sales'].values
        
        # Smooth actuals slightly for the "joyplot" flowing aesthetic, but preserve scale
        if len(x_act) > 3:
            from scipy.interpolate import make_interp_spline
            try:
                x_act_smooth = np.linspace(x_act.min(), x_act.max(), 300)
                spl = make_interp_spline(x_act, z_act, k=2)
                z_act_smooth = np.maximum(spl(x_act_smooth), 0)
                
                v_act = [(x_act_smooth[0], 0)] + list(zip(x_act_smooth, z_act_smooth)) + [(x_act_smooth[-1], 0)]
            except:
                v_act = [(x_act[0], 0)] + list(zip(x_act, z_act)) + [(x_act[-1], 0)]
        elif len(x_act) > 0:
            v_act = [(x_act[0], 0)] + list(zip(x_act, z_act)) + [(x_act[-1], 0)]
        else:
            v_act = [(0,0), (0,0)]
            
        verts_actual.append(v_act)
        
        # -- Predictions --
        f_pred = df_pred_top[df_pred_top['family_id'] == family].sort_values('target_month')
        if not f_pred.empty:
            x_pr = np.array([month_to_x[m] for m in f_pred['target_month']])
            z_pr = f_pred['y_pred'].values
            
            # For a pleasing look, we also pad predictions left/right to drop down to 0
            if len(x_pr) > 1:
                # Slight smoothing
                from scipy.interpolate import make_interp_spline
                try:
                    x_pr_smooth = np.linspace(x_pr.min(), x_pr.max(), 100)
                    spl_pr = make_interp_spline(x_pr, z_pr, k=1 if len(x_pr)==2 else 2)
                    z_pr_smooth = np.maximum(spl_pr(x_pr_smooth), 0)
                    
                    # Drop bounding to zero smoothly to form a polygon
                    dist = 0.5
                    v_pr = [(x_pr_smooth[0]-dist, 0)] + list(zip(x_pr_smooth, z_pr_smooth)) + [(x_pr_smooth[-1]+dist, 0)]
                except:
                    v_pr = [(x_pr[0], 0)] + list(zip(x_pr, z_pr)) + [(x_pr[-1], 0)]
            else:
                 v_pr = [(x_pr[0]-0.5, 0), (x_pr[0], z_pr[0]), (x_pr[0]+0.5, 0)]
        else:
            v_pr = [(0,0), (0,0)]
            
        verts_pred.append(v_pr)

    colors_act = [cmap_base(0.4 + 0.5 * (i/top_n)) for i in range(top_n)]
    colors_pred = [cmap_pred(0.5 + 0.5 * (i/top_n)) for i in range(top_n)]

    # Actuals Collection
    poly_act = PolyCollection(verts_actual, facecolors=colors_act, alpha=0.6, edgecolors='white', linewidths=0.8)
    ax.add_collection3d(poly_act, zs=y_positions, zdir='y')
    
    # Preds Collection
    poly_pred = PolyCollection(verts_pred, facecolors=colors_pred, alpha=0.8, edgecolors='#ffd0d0', linewidths=1.2)
    # We add a slight offset to preds purely for z-order rendering stability in mplot3d
    ax.add_collection3d(poly_pred, zs=y_positions + 0.05, zdir='y')

    # Label formatting
    x_ticks_idx = np.arange(0, len(all_months), max(1, len(all_months)//8))
    ax.set_xticks(x_ticks_idx)
    ax.set_xticklabels([format_month(all_months[i]) for i in x_ticks_idx], rotation=30, ha='right')
    
    ax.set_yticks(y_positions)
    # the labels are reversed because we reversed the loop
    ax.set_yticklabels(
        [_format_family_tick_label(f) for f in reversed(top_families)],
        rotation=0,
        ha='left',
        va='center',
    )
    ax.tick_params(axis='x', labelsize=9, pad=1)
    ax.tick_params(axis='y', labelsize=8, pad=10)
    ax.tick_params(axis='z', labelsize=9, pad=3)
    
    # Adjust limits
    ax.set_xlim(0, len(all_months)-1)
    ax.set_ylim(-0.4, y_positions[-1] + y_step * 0.7)
    
    # Find max Z for limits
    max_z = 0
    for v in verts_actual:
        if v: max_z = max(max_z, max(i[1] for i in v))
    for v in verts_pred:
        if v: max_z = max(max_z, max(i[1] for i in v))
        
    ax.set_zlim(0, max_z * 1.1)
    
    ax.set_xlabel('\nTime (Month)', fontweight='bold')
    ax.set_ylabel('\nProduct Family', fontweight='bold', labelpad=24)
    ax.set_zlabel('Sales Vol', fontweight='bold')
    
    # Title
    title = "Actual vs Predicted Sales Waterfall Plot"
    sub_title = f"Top {top_n} Families sorted by {sort_criterion.title()} (Model: {model_name})"
    fig.suptitle(title, fontsize=18, fontweight='bold', y=0.92)
    ax.set_title(sub_title, fontsize=12, color='#555555', pad=10)

    # Custom Legend
    from matplotlib.lines import Line2D
    custom_lines = [
        PolyCollection([[(0,0)]], facecolors=cmap_base(0.7), alpha=0.6, edgecolors='white', linewidths=0.8),
        PolyCollection([[(0,0)]], facecolors=cmap_pred(0.7), alpha=0.8, edgecolors='white', linewidths=1.2)
    ]
    ax.legend(
        [
            Line2D([0], [0], color=cmap_base(0.7), lw=4, alpha=0.6), 
            Line2D([0], [0], color=cmap_pred(0.7), lw=4, alpha=0.8)
        ],
        ['Actual Sales (All History)', 'Predicted Sales (Last 4 Months)'],
        loc='upper right', bbox_to_anchor=(0.95, 0.85), frameon=True
    )

    # tight_layout is unreliable for 3D axes with long tick labels, so use explicit margins.
    fig.subplots_adjust(left=0.03, right=0.9, bottom=0.08, top=0.88)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Waterfall plot saved to {output_path}")

if __name__ == '__main__':
    project_root = Path(__file__).resolve().parent.parent.parent
    gen_dir = project_root / 'outputs'
    out_dir = gen_dir / 'report_plots'
    
    panel_f = gen_dir / 'family_month_panel.csv'
    eval_f = gen_dir / 'forecast_eval.csv'
    
    if panel_f.exists() and eval_f.exists():
        p1 = out_dir / 'waterfall_plot_by_sales.png'
        generate_waterfall_plot(panel_f, eval_f, p1, sort_criterion='sales', top_n=10)
        
        p2 = out_dir / 'waterfall_plot_by_error.png'
        generate_waterfall_plot(panel_f, eval_f, p2, sort_criterion='error', top_n=10)
    else:
        print("Required CSV files not found in outputs/ directory.")
