import matplotlib.pyplot as plt
import numpy as np


colors = ["#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F", "#EDC949", "#AF7AA1", "#FF9DA7", "#9C755F", "#BAB0AC"]

def plot_top_grams(top, name=""):
    """
    Plot a horizontal bar chart from a DataFrame with columns:
      - 'term'  : string
      - 'count' : int (absolute frequency)
      - 'share' : float (relative frequency in [0,1])

    Parameters
    ----------
    top : pd.DataFrame
        DataFrame with 'term' and 'count' columns.
    name : str, optional
        Name to use in the chart title, by default "".
    """

    # Reverse for horizontal bar so largest is at the top
    terms = top["term"].tolist()[::-1]
    values = top['count'].tolist()[::-1]

    fig, ax = plt.subplots(figsize=(5, 3))
    #ax.set_axisbelow(True)  # grid lines below bars
    ax.grid(zorder=0)        
    ax.barh(terms, values, color=colors[0], zorder=2)              # default colors; no style set
    ax.set_xlabel("Count")
    ax.set_title(f"Top 10 {name if name != "" else "terms"} by count")
    fig.tight_layout()
    #plt.show()
    return fig


def quick_snapshot_charts(data):
    """
    Expect df_scored to already have these columns (from your earlier step):
      - vader_compound
      - lm_has_neg, lm_has_uncert
      - lm_neg_share, lm_uncert_share, lm_litig_share, lm_constr_share, lm_pos_share

    Produces TWO figures:
      1) Horizontal bar: headline tone metrics (VADER compound mean, % LM-negative, % LM-uncertainty, % LM-positive)
      2) 100% stacked bar: average composition of LM category shares
    """

    # --------- 1) Snapshot metrics (level chart) ---------
    # Compute simple aggregates
    vader_mean = data['vader_mean']
    pct_neg    = data['head_w_neg']
    pct_unc    = data['head_w_unc']
    pct_pos    = data['head_w_pos']

    labels = ["VADER polarity (mean)", "% headlines with LM-neg toks", "% headlines with LM-unc toks", "% headlines with LM-pos toks"]
    values = [vader_mean, pct_neg, pct_unc, pct_pos]

    # Plot as a horizontal bar chart (sorted so largest appears at top)
    order = np.argsort(values)
    values_sorted = [values[i] for i in order]
    labels_sorted = [labels[i] for i in order]

    fig1, ax1 = plt.subplots()
    ax1.grid(axis="x",zorder=0)               # vertical gridlines for barh
    ax1.barh(labels_sorted, values_sorted, zorder=2, color=colors[0]) 
    ax1.set_title("Headline tone snapshot")
    ax1.set_xlabel("Score / Percent")
    plt.tight_layout()
    #plt.show()

    # --------- 2) 100% stacked bar of LM composition ---------
    # Average shares across headlines (each share is already in [0,1])
    comp = {
        "Negative":    data['neg_share_of_tok'],
        "Positive":    data['pos_share_of_tok'],
        "Uncertainty": data['unc_share_of_tok'],
        "Litigious":   data['litig_share_of_tok'],
        "Constraining":data['constr_share_of_tok'],
    }
    # Normalize to sum=1 in case of tiny numeric drift
    total = sum(comp.values()) or 1.0
    parts = {k: v / total for k, v in comp.items()}
    cats  = list(parts.keys())
    vals  = [parts[k] for k in cats]

    # Build a single stacked bar that totals 100%
    fig2, ax2 = plt.subplots()
    ax2.grid(axis="y", zorder=0)
    left = 0.0
    for i, v in enumerate(vals):
        ax2.bar(0, v * 100.0, bottom=left * 100.0, zorder=2, color=colors[i])  # no explicit colors per your preference
        left += v

    ax2.set_xlim(-0.6, 0.6)
    ax2.set_xticks([])
    ax2.set_ylabel("% of LM share")
    ax2.set_title("Share of tokens")

    # Add category labels centered in each segment (if segment is big enough)
    left = 0.0
    for label, v in zip(cats, vals):
        if v > 0.05:  # only annotate if segment >= 5%
            ax2.text(0, (left + v / 2) * 100.0, f"{label} {v*100:.1f}%", ha="center", va="center")
        left += v

    plt.tight_layout()
    #plt.show()

    return fig1, fig2
