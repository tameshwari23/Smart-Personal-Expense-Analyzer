# ============================================================
# GRADIO UI — Smart Personal Expense Analyzer
# ============================================================
# HOW TO RUN IN GOOGLE COLAB:
#   Step 1: Run your existing Smart_Expense_Analyzer.py first
#           (to generate .pkl model files in outputs/)
#   Step 2: !pip install gradio
#   Step 3: Run this file (or paste cells into Colab)
#
# PUBLIC LINK:  Gradio auto-generates a share link — see bottom
# ============================================================

# ── Cell 1: Install Gradio ────────────────────────────────────
# !pip install gradio --quiet

# ── Cell 2: Imports ───────────────────────────────────────────
import gradio as gr
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# ── Cell 3: Load Models (from your existing outputs/) ─────────
MODEL_DIR = "outputs"   # Same folder Smart_Expense_Analyzer.py uses

def _load_models():
    """Load pre-trained models saved by Smart_Expense_Analyzer.py"""
    try:
        lin  = joblib.load(os.path.join(MODEL_DIR, "linear_regression_model.pkl"))
        log  = joblib.load(os.path.join(MODEL_DIR, "logistic_regression_model.pkl"))
        sc   = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
        print("✅ Trained models loaded from outputs/")
        return lin, log, sc
    except FileNotFoundError:
        print("⚠️  Model files not found — using built-in fallback model.")
        print("   Run Smart_Expense_Analyzer.py first for best accuracy.")
        return None, None, None

lin_model, log_model, scaler = _load_models()


# ─────────────────────────────────────────────────────────────
# SECTION A — PREDICTION LOGIC
# Wraps your existing trained model pipeline
# ─────────────────────────────────────────────────────────────

# Encoding maps — mirror Smart_Expense_Analyzer.py LabelEncoder order
INCOME_TYPE_MAP      = {"Salaried": 2, "Self-Employed": 3, "Freelance": 1,
                        "Business": 0, "Part-Time": 4}
SCENARIO_MAP         = {"Normal": 2, "Saving": 3, "Debt Repayment": 0,
                        "Investment": 1, "Emergency": 5, "Retirement": 4}
STRESS_MAP           = {"Low": 1, "Medium": 2, "High": 0}

CATEGORY_BUDGET = {
    "Rent":           1320,
    "Groceries":       890,
    "Transportation":  540,
    "Insurance":       480,
    "Healthcare":      370,
    "Utilities":       340,
    "Education":       310,
    "Investments":     290,
    "Dining Out":      260,
    "Entertainment":   210,
}

def _fallback_predict(income, ess, disc, loan, invest):
    """Simple rule-based fallback when .pkl files are unavailable."""
    return ess + disc + loan + invest + income * 0.05

def predict_expense(
    monthly_income,
    savings_rate,
    credit_score,
    debt_to_income_ratio,
    loan_payment,
    investment_amount,
    discretionary_spending,
    essential_spending,
    transaction_count,
    income_type,
    financial_scenario,
    financial_stress_level,
):
    """
    Core prediction function — same feature order as your training code.
    Called by Gradio when the user clicks 'Analyze'.
    """
    # Encode categoricals using the same mapping as LabelEncoder
    income_type_enc      = INCOME_TYPE_MAP.get(income_type, 2)
    scenario_enc         = SCENARIO_MAP.get(financial_scenario, 2)
    stress_enc           = STRESS_MAP.get(financial_stress_level, 1)

    features = np.array([[
        monthly_income, savings_rate, credit_score,
        debt_to_income_ratio, loan_payment,
        investment_amount, discretionary_spending,
        essential_spending, transaction_count,
        income_type_enc, scenario_enc, stress_enc
    ]])

    # Scale → predict
    if scaler:
        features_scaled = scaler.transform(features)
    else:
        features_scaled = features

    if lin_model:
        predicted = float(lin_model.predict(features_scaled)[0])
    else:
        predicted = _fallback_predict(
            monthly_income, essential_spending,
            discretionary_spending, loan_payment, investment_amount
        )

    # Savings goal classification
    if log_model:
        goal_met    = int(log_model.predict(features_scaled)[0])
        probability = float(log_model.predict_proba(features_scaled)[0][goal_met])
    else:
        goal_met    = 1 if savings_rate >= 15 else 0
        probability = 0.76 if goal_met else 0.34

    surplus   = monthly_income - predicted
    savings   = monthly_income * savings_rate / 100

    return predicted, surplus, savings, goal_met, probability


# ─────────────────────────────────────────────────────────────
# SECTION B — INSIGHT TEXT
# ─────────────────────────────────────────────────────────────

def generate_insight(predicted, income, surplus, savings_rate,
                     goal_met, probability, category):
    """Return a human-readable insight string."""

    # Overspend warning
    spend_pct  = (predicted / income) * 100 if income > 0 else 100
    goal_label = "✅ Likely to meet savings goal" if goal_met else "❌ Unlikely to meet savings goal"
    conf_str   = f"{probability * 100:.1f}% confidence"

    if surplus < 0:
        spend_status = "🔴 Caution: You are likely to **overspend** this month."
    elif spend_pct > 80:
        spend_status = "🟡 Warning: Expenses consume >80% of income — consider reducing discretionary spending."
    else:
        spend_status = "🟢 Your spending is within a healthy range."

    # Category tip
    avg_budget = CATEGORY_BUDGET.get(category, None)
    if avg_budget:
        cat_tip = (f"\n💡 Average budget for **{category}**: ₹{avg_budget:,}. "
                   f"Suggested max: ₹{int(avg_budget * 0.9):,} (10% reduction).")
    else:
        cat_tip = ""

    savings_tip = ""
    if savings_rate < 10:
        savings_tip = "\n📌 Tip: Aim for at least a 10–15% savings rate for long-term financial health."
    elif savings_rate >= 20:
        savings_tip = "\n🌟 Great job! A 20%+ savings rate puts you in an excellent financial position."

    return (
        f"### 📊 Spending Insight\n\n"
        f"{spend_status}\n\n"
        f"**Predicted Monthly Expense:** ₹{predicted:,.2f}\n"
        f"**Monthly Surplus / Deficit:** ₹{surplus:+,.2f}\n"
        f"**Estimated Savings:** ₹{income * savings_rate / 100:,.2f}\n\n"
        f"**Savings Goal:** {goal_label} ({conf_str})\n"
        f"{cat_tip}"
        f"{savings_tip}"
    )


# ─────────────────────────────────────────────────────────────
# SECTION C — VISUALIZATION FUNCTIONS
# ─────────────────────────────────────────────────────────────

def plot_expense_trend(income, predicted, essential, discretionary,
                       loan, investment, savings_rate):
    """
    Graph 1 — Expense vs Income trend over 6 simulated months,
    showing how predicted expenses change with income variation.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor('#0F172A')
    for ax in axes:
        ax.set_facecolor('#1E293B')
        ax.tick_params(colors='#CBD5E1')
        ax.xaxis.label.set_color('#CBD5E1')
        ax.yaxis.label.set_color('#CBD5E1')
        ax.title.set_color('#F1F5F9')
        for spine in ax.spines.values():
            spine.set_edgecolor('#334155')

    # --- Left: 6-month simulated trend ---
    months    = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
    variation = [0.95, 1.02, 0.98, 1.05, 1.01, predicted / income if income else 1]
    inc_vals  = [income * v for v in variation]
    exp_vals  = [predicted * v for v in variation]
    sav_vals  = [max(0, i - e) for i, e in zip(inc_vals, exp_vals)]

    axes[0].plot(months, inc_vals, 'o-', color='#38BDF8', linewidth=2,
                 markersize=6, label='Income')
    axes[0].plot(months, exp_vals, 's-', color='#FB7185', linewidth=2,
                 markersize=6, label='Expense')
    axes[0].fill_between(months, exp_vals, inc_vals,
                          where=[i > e for i, e in zip(inc_vals, exp_vals)],
                          alpha=0.15, color='#4ADE80', label='Surplus zone')
    axes[0].fill_between(months, exp_vals, inc_vals,
                          where=[i <= e for i, e in zip(inc_vals, exp_vals)],
                          alpha=0.2, color='#FB7185', label='Deficit zone')
    axes[0].set_title("Simulated 6-Month Income vs Expense Trend", fontsize=11)
    axes[0].set_ylabel("Amount (₹)")
    axes[0].legend(fontsize=8, facecolor='#1E293B', labelcolor='#CBD5E1',
                   framealpha=0.6)
    axes[0].yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"₹{x:,.0f}"))
    axes[0].grid(axis='y', alpha=0.2, color='#475569')

    # --- Right: Savings accumulation bar ---
    axes[1].bar(months, sav_vals, color='#4ADE80', alpha=0.85, edgecolor='#1E293B')
    axes[1].set_title("Projected Monthly Savings", fontsize=11)
    axes[1].set_ylabel("Savings (₹)")
    axes[1].yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"₹{x:,.0f}"))
    axes[1].grid(axis='y', alpha=0.2, color='#475569')
    for bar, val in zip(axes[1].patches, sav_vals):
        axes[1].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + income * 0.005,
                     f"₹{val:,.0f}", ha='center', va='bottom',
                     fontsize=7, color='#CBD5E1')

    plt.tight_layout()
    return fig


def plot_category_spending(essential, discretionary, loan,
                           investment, income, category):
    """
    Graph 2 — Category-wise spending breakdown:
    donut chart + category benchmark bar.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor('#0F172A')
    for ax in axes:
        ax.set_facecolor('#1E293B')
        ax.tick_params(colors='#CBD5E1')
        ax.xaxis.label.set_color('#CBD5E1')
        ax.yaxis.label.set_color('#CBD5E1')
        ax.title.set_color('#F1F5F9')
        for spine in ax.spines.values():
            spine.set_edgecolor('#334155')

    # --- Left: Donut chart of expense breakdown ---
    misc  = max(0, income - essential - discretionary - loan - investment)
    sizes = [essential, discretionary, loan, investment, misc]
    lbls  = ['Essential', 'Discretionary', 'Loan\nPayments', 'Investments', 'Misc']
    colors_donut = ['#38BDF8', '#FB7185', '#FBBF24', '#4ADE80', '#A78BFA']
    # Remove zero slices
    valid = [(s, l, c) for s, l, c in zip(sizes, lbls, colors_donut) if s > 0]
    sizes_v, lbls_v, colors_v = zip(*valid) if valid else ([], [], [])

    wedges, texts, autotexts = axes[0].pie(
        sizes_v, labels=lbls_v, colors=colors_v,
        autopct='%1.1f%%', startangle=90,
        wedgeprops=dict(width=0.55, edgecolor='#0F172A', linewidth=1.5),
        textprops=dict(color='#F1F5F9', fontsize=9)
    )
    for at in autotexts:
        at.set_color('#0F172A')
        at.set_fontsize(8)
    axes[0].set_title("Expense Breakdown (Current Month)", fontsize=11)

    # --- Right: Category benchmark comparison ---
    cats    = list(CATEGORY_BUDGET.keys())
    avgs    = list(CATEGORY_BUDGET.values())
    suggest = [int(v * 0.9) for v in avgs]
    user_val = CATEGORY_BUDGET.get(category, 0)

    x = np.arange(len(cats))
    w = 0.35
    b1 = axes[1].bar(x - w/2, avgs, w, label='Avg Expense',
                     color='#64748B', alpha=0.8)
    b2 = axes[1].bar(x + w/2, suggest, w, label='Suggested Budget',
                     color='#4ADE80', alpha=0.8)

    # Highlight selected category
    if category in cats:
        idx = cats.index(category)
        b1[idx].set_color('#FB7185')
        b2[idx].set_color('#38BDF8')

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(cats, rotation=40, ha='right', fontsize=7.5)
    axes[1].set_title("Category Budget: Avg vs Suggested (–10%)", fontsize=10)
    axes[1].set_ylabel("Amount (₹)")
    axes[1].legend(fontsize=8, facecolor='#1E293B', labelcolor='#CBD5E1')
    axes[1].yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"₹{x:,.0f}"))
    axes[1].grid(axis='y', alpha=0.2, color='#475569')

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────
# SECTION D — MASTER GRADIO CALLBACK
# This is the single function Gradio calls on button click
# ─────────────────────────────────────────────────────────────

def analyze(
    monthly_income,
    savings_rate,
    credit_score,
    debt_to_income_ratio,
    loan_payment,
    investment_amount,
    discretionary_spending,
    essential_spending,
    transaction_count,
    income_type,
    financial_scenario,
    financial_stress_level,
    category,
):
    """Master callback — runs prediction + generates both graphs."""

    predicted, surplus, savings, goal_met, probability = predict_expense(
        monthly_income, savings_rate, credit_score, debt_to_income_ratio,
        loan_payment, investment_amount, discretionary_spending,
        essential_spending, transaction_count,
        income_type, financial_scenario, financial_stress_level,
    )

    insight = generate_insight(
        predicted, monthly_income, surplus, savings_rate,
        goal_met, probability, category
    )

    fig_trend    = plot_expense_trend(
        monthly_income, predicted, essential_spending,
        discretionary_spending, loan_payment, investment_amount, savings_rate
    )
    fig_category = plot_category_spending(
        essential_spending, discretionary_spending,
        loan_payment, investment_amount, monthly_income, category
    )

    return insight, fig_trend, fig_category


# ─────────────────────────────────────────────────────────────
# SECTION E — GRADIO BLOCKS UI
# ─────────────────────────────────────────────────────────────

THEME = gr.themes.Base(
    primary_hue="sky",
    secondary_hue="slate",
    neutral_hue="slate",
    font=gr.themes.GoogleFont("Inter"),
).set(
    body_background_fill="#0F172A",
    body_text_color="#F1F5F9",
    block_background_fill="#1E293B",
    block_border_color="#334155",
    block_label_text_color="#94A3B8",
    input_background_fill="#0F172A",
    button_primary_background_fill="#0EA5E9",
    button_primary_background_fill_hover="#38BDF8",
    button_primary_text_color="#0F172A",
)

CSS = """
#title-row { text-align: center; padding: 10px 0 4px; }
#title-row h1 { font-size: 2rem; font-weight: 700; margin: 0;
                background: linear-gradient(90deg,#38BDF8,#818CF8);
                -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
#title-row p  { color: #94A3B8; margin: 4px 0 0; font-size: 0.95rem; }
.section-label { font-weight: 600; color: #38BDF8 !important; font-size: 0.8rem;
                 text-transform: uppercase; letter-spacing: 0.08em; }
#analyze-btn   { width: 100%; font-size: 1rem; font-weight: 700;
                 padding: 12px; border-radius: 10px; }
"""

with gr.Blocks(theme=THEME, css=CSS, title="Smart Expense Analyzer") as demo:

    # ── Header ──────────────────────────────────────────────────
    with gr.Row(elem_id="title-row"):
        gr.HTML("""
            <div>
              <h1>💰 Smart Personal Expense Analyzer</h1>
              <p>ML-powered spending prediction &amp; budget insights</p>
            </div>
        """)

    # ── Input / Output layout ────────────────────────────────────
    with gr.Row():

        # ── LEFT COLUMN — Inputs ─────────────────────────────────
        with gr.Column(scale=4):

            gr.Markdown("### 🔹 Financial Profile", elem_classes="section-label")

            with gr.Row():
                monthly_income = gr.Slider(
                    minimum=5000, maximum=200000, value=60000, step=1000,
                    label="💵 Monthly Income (₹)",
                    info="Drag to adjust — prediction updates on Analyze"
                )

            with gr.Row():
                savings_rate = gr.Slider(
                    minimum=0, maximum=60, value=20, step=1,
                    label="💾 Savings Rate (%)"
                )
                credit_score = gr.Slider(
                    minimum=300, maximum=900, value=700, step=10,
                    label="📈 Credit Score"
                )

            with gr.Row():
                essential_spending = gr.Number(
                    value=15000, label="🏠 Essential Spending (₹)",
                    info="Rent, groceries, utilities"
                )
                discretionary_spending = gr.Number(
                    value=8000, label="🛍️ Discretionary Spending (₹)",
                    info="Dining, entertainment, shopping"
                )

            with gr.Row():
                loan_payment = gr.Number(
                    value=5000, label="🏦 Loan Payment (₹/month)"
                )
                investment_amount = gr.Number(
                    value=4000, label="📊 Investment Amount (₹/month)"
                )

            gr.Markdown("### 🔹 Lifestyle Details", elem_classes="section-label")

            with gr.Row():
                debt_to_income_ratio = gr.Slider(
                    minimum=0.0, maximum=1.0, value=0.3, step=0.01,
                    label="⚖️ Debt-to-Income Ratio"
                )
                transaction_count = gr.Slider(
                    minimum=1, maximum=60, value=20, step=1,
                    label="🔢 Monthly Transactions"
                )

            with gr.Row():
                income_type = gr.Dropdown(
                    choices=list(INCOME_TYPE_MAP.keys()),
                    value="Salaried", label="💼 Income Type"
                )
                financial_scenario = gr.Dropdown(
                    choices=list(SCENARIO_MAP.keys()),
                    value="Normal", label="📋 Financial Scenario"
                )

            with gr.Row():
                financial_stress_level = gr.Dropdown(
                    choices=list(STRESS_MAP.keys()),
                    value="Low", label="😓 Stress Level"
                )
                category = gr.Dropdown(
                    choices=list(CATEGORY_BUDGET.keys()),
                    value="Groceries", label="🏷️ Primary Spending Category"
                )

            analyze_btn = gr.Button("🔍 Analyze My Expenses",
                                    variant="primary", elem_id="analyze-btn")

        # ── RIGHT COLUMN — Outputs ───────────────────────────────
        with gr.Column(scale=5):

            insight_out = gr.Markdown(
                value="*Fill in your details and click **Analyze** to see your personalized insight.*",
                label="📊 Insight"
            )

            trend_plot = gr.Plot(label="📈 Expense Trend (6-Month Simulation)")
            cat_plot   = gr.Plot(label="🏷️ Category-wise Spending Benchmark")

    # ── Wire button → analyze() ──────────────────────────────────
    analyze_btn.click(
        fn=analyze,
        inputs=[
            monthly_income, savings_rate, credit_score, debt_to_income_ratio,
            loan_payment, investment_amount, discretionary_spending,
            essential_spending, transaction_count,
            income_type, financial_scenario, financial_stress_level,
            category,
        ],
        outputs=[insight_out, trend_plot, cat_plot],
    )

    # ── Footer ───────────────────────────────────────────────────
    gr.HTML("""
        <div style="text-align:center;padding:14px 0 4px;color:#475569;font-size:0.8rem;">
          Smart Personal Expense Analyzer · Data Science Mini Project 2025–26
          · Powered by scikit-learn &amp; Gradio
        </div>
    """)


# ─────────────────────────────────────────────────────────────
# SECTION F — LAUNCH
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo.launch(
        share=True,          # ← Generates a public https://xxxxx.gradio.live link
        debug=False,
        server_name="0.0.0.0",
        server_port=7860,
    )

# ── For Google Colab: paste this as the last cell ─────────────
# demo.launch(share=True)
