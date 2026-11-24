"""Build the 12-slide PDF deck for the COMP3522 sprint challenge.

Each slide is a matplotlib figure (16:9) with a title bar, optional embedded
image(s) and a bullet text panel. Writes slides/deck.pdf.

Run from the repo root inside .venv:
    .venv\\Scripts\\python.exe scripts\\build_deck.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "outputs" / "figures"
OUTPUTS = ROOT / "outputs"
SLIDES_DIR = ROOT / "slides"
DECK = SLIDES_DIR / "deck.pdf"
RESULTS = json.loads((OUTPUTS / "results.json").read_text(encoding="utf-8"))

# Colours
NAVY = "#0B2545"
TEAL = "#1B998B"
CORAL = "#EE6C4D"
LIGHT = "#F5F5F5"
WHITE = "#FFFFFF"
GREY = "#5C5C5C"

FIG_W, FIG_H = 13.333, 7.5  # 16:9 inches


def new_slide(title: str, subtitle: str | None = None):
    fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=100)
    fig.patch.set_facecolor(WHITE)
    # Title bar
    bar = fig.add_axes([0, 0.90, 1, 0.10])
    bar.set_axis_off()
    bar.add_patch(FancyBboxPatch((0.02, 0.1), 0.96, 0.8,
                                 boxstyle="round,pad=0.01,rounding_size=0.02",
                                 fc=NAVY, ec="none", transform=bar.transAxes))
    bar.text(0.04, 0.5, title, color=WHITE, fontsize=22, fontweight="bold",
             va="center", ha="left", transform=bar.transAxes)
    if subtitle:
        bar.text(0.96, 0.5, subtitle, color="#B8C4D6", fontsize=13,
                 va="center", ha="right", transform=bar.transAxes)
    return fig


def add_image(fig, path, rect, title=None):
    import matplotlib.image as mpimg
    ax = fig.add_axes(rect)
    img = mpimg.imread(path)
    ax.imshow(img)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=13, color=NAVY, pad=6)
    return ax


def add_text_panel(fig, rect, bullets, fontsize=14, title=None):
    ax = fig.add_axes(rect)
    ax.set_axis_off()
    y = 0.98
    if title:
        ax.text(0.0, y, title, fontsize=16, fontweight="bold", color=NAVY,
                transform=ax.transAxes, va="top")
        y -= 0.10
    for b in bullets:
        # simple bullet: prefix •
        ax.text(0.0, y, f"•  {b}", fontsize=fontsize, color="#222222",
                transform=ax.transAxes, va="top", wrap=True)
        # estimate height: rough char-based wrap
        chars = max(1, int(len(b) / 52) + 1)
        y -= 0.052 * chars + 0.015
    return ax


def add_footer(fig, text):
    fig.text(0.5, 0.02, text, ha="center", fontsize=9, color=GREY, style="italic")


def cover():
    fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=100)
    fig.patch.set_facecolor(NAVY)
    fig.text(0.08, 0.62, "COMP3522 Sprint Challenge", fontsize=20, color="#B8C4D6",
             fontweight="bold")
    fig.text(0.08, 0.48, "Smoking Baby's Guide to Safe Days:\nAir Quality Trends & Weather",
             fontsize=34, color=WHITE, fontweight="bold", va="top")
    fig.text(0.08, 0.30,
             "Time-series analysis of Hong Kong hourly AQHI (2021-2025)\n"
             "linked to Open-Meteo weather covariates (2023-2025)",
             fontsize=15, color="#B8C4D6")
    fig.text(0.08, 0.12, "COMP3522  •  24 November 2026", fontsize=13, color="#7A8BA8")
    return fig


def references():
    fig = new_slide("References & Data Sources")
    bullets = [
        "AQHI data: HK EPD, Past Records of Air Quality Health Index (2021-2026). "
        "data.gov.hk / aqhi.gov.hk",
        "AQHI health-risk categories: EPD AQHI (Low 1-3, Moderate 4-6, High 7, "
        "Very High 8-10, Serious 10+).",
        "Weather covariate: Open-Meteo hourly historical, published as "
        "NotASI/COMP3522-HKO on Hugging Face.",
        "AQHI data dictionary: aqhi.gov.hk/psi/dd/hk_aqhien_en.pdf",
        "Smoking Baby scenario: COMP3522 Sprint Challenge (2026-27 Sem 1).",
    ]
    add_text_panel(fig, [0.08, 0.10, 0.84, 0.72], bullets, fontsize=15,
                   title="Sources")
    return fig


def main() -> int:
    SLIDES_DIR.mkdir(parents=True, exist_ok=True)
    q1, q2, m = RESULTS["q1"], RESULTS["q2"], RESULTS.get("q2_model", {})
    pages = []

    pages.append(cover())

    # --- Slide: Agenda / data & method
    fig = new_slide("Data & Method", "Agenda")
    bullets = [
        "Q1: historical trends in hourly AQHI 2021-2025 (18 stations).",
        "Q2: link AQHI to 6 Open-Meteo weather variables (2023-2025).",
        "Q3: safe-day conditions + a simple logistic risk score.",
        "Cleaning: strip '*' flags, cap 10+ at 10, drop missing hours.",
        "Poor air = AQHI >= 7 (EPD High+) for at-risk people.",
        "Time split: train 2023-2024, validate 2025.",
    ]
    add_text_panel(fig, [0.08, 0.12, 0.84, 0.68], bullets, fontsize=17)
    pages.append(fig)

    # --- Q1 annual trend
    fig = new_slide("Q1: Annual trend", "Territory mean AQHI 2021-2025")
    add_image(fig, str(FIG / "q1_annual_trend.png"), [0.05, 0.10, 0.55, 0.72])
    add_text_panel(fig, [0.63, 0.12, 0.34, 0.70], [
        f"Cleanest {q1['year_min']}: {q1['annual_mean'][str(q1['year_min'])]}",
        f"Worst {q1['year_max']}: {q1['annual_mean'][str(q1['year_max'])]}",
        "Year-to-year variation is modest (~0.2 AQHI).",
        "Season dominates over the multi-year trend.",
    ], fontsize=14)
    pages.append(fig)

    # --- Q1 seasonal
    fig = new_slide("Q1: Seasonal pattern", "Mean AQHI by month of year")
    add_image(fig, str(FIG / "q1_seasonal_month.png"), [0.05, 0.10, 0.55, 0.72])
    pm = q1["month_of_year_mean"]
    peak_m = max(pm, key=pm.get)
    low_m = min(pm, key=pm.get)
    add_text_panel(fig, [0.63, 0.12, 0.34, 0.70], [
        f"Peak in month {peak_m}: {pm[peak_m]}",
        f"Cleanest month {low_m}: {pm[low_m]}",
        "Winter (Dec-Mar) is the polluted season.",
        "Summer monsoon (Jun-Aug) is cleanest.",
    ], fontsize=14)
    pages.append(fig)

    # --- Q1 diurnal
    fig = new_slide("Q1: Daily (diurnal) cycle", "Mean AQHI by hour of day")
    add_image(fig, str(FIG / "q1_diurnal.png"), [0.05, 0.10, 0.55, 0.72])
    add_text_panel(fig, [0.63, 0.12, 0.34, 0.70], [
        f"Peak ~{q1['peak_hour']}:00, trough ~{q1['trough_hour']}:00",
        f"Diurnal range {q1['diurnal_range']} AQHI points.",
        "Rush-hour + daytime activity raise AQHI.",
        "Early morning is the cleanest window.",
    ], fontsize=14)
    pages.append(fig)

    # --- Q1 spatial
    fig = new_slide("Q1: Spatial pattern", "Station x month heatmap")
    add_image(fig, str(FIG / "q1_station_month_heatmap.png"), [0.05, 0.08, 0.55, 0.76])
    rm = q1["region_mean"]
    add_text_panel(fig, [0.63, 0.12, 0.34, 0.70], [
        "Every station follows the same seasonal wave.",
        f"Highest region: {max(rm, key=rm.get)} ({max(rm.values())})",
        f"Lowest region: {min(rm, key=rm.get)} ({min(rm.values())})",
        "New Territories (Tuen Mun, Tung Chung) often higher.",
    ], fontsize=14)
    pages.append(fig)

    # --- Q1 categories + daily series
    fig = new_slide("Q1: Risk categories & daily series", "AQHI category share + 2021-2025 daily")
    add_image(fig, str(FIG / "q1_categories.png"), [0.05, 0.10, 0.42, 0.72])
    add_image(fig, str(FIG / "q1_daily_timeseries.png"), [0.50, 0.10, 0.48, 0.72])
    add_text_panel(fig, [0.05, 0.02, 0.90, 0.08], [
        f"Only {q1['poor_share_pct']}% of station-hours are poor (AQHI>=7); "
        "most hours are Low/Moderate.",
    ], fontsize=13)
    pages.append(fig)

    # --- Q1 advanced: STL, change-points, forecast
    adv = RESULTS.get("advanced_ts", {})
    fig = new_slide("Q1 (advanced): STL, breaks & forecast", "Deseasonalised structure + backtest")
    add_image(fig, str(FIG / "ats_stl_monthly.png"), [0.05, 0.30, 0.45, 0.52])
    add_image(fig, str(FIG / "ats_forecast.png"), [0.52, 0.30, 0.45, 0.52])
    mk = adv.get("mann_kendall", {}).get("monthly", {})
    fc = adv.get("forecast", {})
    ets = fc.get("ets_holt_winters", {})
    sn = fc.get("seasonal_naive", {})
    cp = adv.get("changepoints", {})
    st = adv.get("stationarity", {}).get("daily_level", {})
    add_text_panel(fig, [0.05, 0.04, 0.92, 0.24], [
        f"STL: seasonal amplitude {adv.get('stl',{}).get('seasonal_amplitude')} AQHI, small trend, resid std {adv.get('stl',{}).get('resid_std')}.",
        f"Mann-Kendall: NO significant monotonic trend (monthly p={mk.get('p')}); ADF/KPSS agree the series is stationary.",
        f"PELT change-points on the deseasonalised trend: {cp.get('break_months')} ({cp.get('n_breaks')} breaks).",
        f"Periodogram peak ~{adv.get('periodogram',{}).get('dominant_period_days')} days = annual cycle.",
        f"Forecast backtest (train<=2023, test 2024-25): ETS MAE {ets.get('mae')} vs seasonal-naive {sn.get('mae')}.",
    ], fontsize=12)
    pages.append(fig)

    # --- Q2a correlations
    fig = new_slide("Q2(a): Weather-AQHI relationships", "Spearman rho on daily data")
    add_image(fig, str(FIG / "q2_correlation_bars.png"), [0.05, 0.10, 0.55, 0.72])
    corrs = q2["correlations"]
    top = sorted(corrs.items(), key=lambda kv: -abs(kv[1]["spearman"]))[:3]
    top_txt = "\n".join(f"{k}: {v['spearman']}" for k, v in top)
    add_text_panel(fig, [0.63, 0.12, 0.34, 0.70], [
        f"Strongest: {q2['top_variable']} (rho={q2['top_spearman']}).",
        "Rain & humidity lower AQHI; high pressure raises it.",
        f"Weather window {q2['weather_window'][0]}..{q2['weather_window'][1]} "
        f"({q2['n_days']} days).",
        top_txt,
    ], fontsize=14)
    pages.append(fig)

    # --- Q2a scatter
    fig = new_slide("Q2(a): Scatter of top weather drivers", "Daily mean AQHI vs weather")
    add_image(fig, str(FIG / "q2_scatter_top2.png"), [0.05, 0.10, 0.90, 0.76])
    pages.append(fig)

    # --- Q2b safe-day + coefficients
    fig = new_slide("Q2(b): Conditions for a safe day", "Standardised logistic effects on P(poor)")
    add_image(fig, str(FIG / "q2_risk_coefficients.png"), [0.05, 0.10, 0.55, 0.72])
    coefs = m.get("coefficients", {})
    add_text_panel(fig, [0.63, 0.12, 0.34, 0.70], [
        "Safest days: warm, humid, wet (summer monsoon).",
        "Riskiest: cool, dry, high-pressure (winter).",
        f"AUC (2025 valid) = {m.get('valid_auc')}.",
        f"Base poor rate = {m.get('valid_base_rate')}.",
    ], fontsize=14)
    pages.append(fig)

    # --- Q2b calibration + formula
    fig = new_slide("Q2(b): Risk score & calibration", "Predicted vs observed poor-day rate")
    add_image(fig, str(FIG / "q2_risk_calibration.png"), [0.05, 0.10, 0.55, 0.72])
    add_text_panel(fig, [0.63, 0.12, 0.34, 0.70], [
        "Simple rule: caution if AQHI>=4 AND low wind AND high pressure.",
        "Avoid outdoor exertion if AQHI>=7 (EPD High+).",
        f"Safest quintile poor rate: {m.get('safest_observed_poor_rate')}.",
        f"Riskiest quintile poor rate: {m.get('riskiest_observed_poor_rate')}.",
    ], fontsize=14)
    pages.append(fig)

    # --- Conclusions
    fig = new_slide("Conclusions & limitations", "Takeaways")
    bullets = [
        "AQHI is strongly seasonal (STL amplitude ~2.2), stationary, with NO significant "
        "long-term trend (Mann-Kendall p=0.57) and two deseasonalised breaks (2022, 2023).",
        "Diurnal peak ~16:00, trough ~06:00; winter polluted, summer clean.",
        "Precipitation and humidity are the strongest weather signals; high pressure "
        "and dry air mark the worst days.",
        "Safe-day advice: go out on warm, humid, wet summer days, early morning; "
        "avoid winter high-pressure dry days and AQHI>=7 hours.",
        "A seasonal ETS model beats seasonal-naive on a 2024-25 backtest (MAE 0.37 vs 0.51); "
        "the logistic score is a transparent screening tool, not a medical device.",
        "Limitations: territory mean unweighted; missing hours dropped; 10+ capped at 10; "
        "weather 2023-2025 only; associations not causal.",
    ]
    add_text_panel(fig, [0.08, 0.10, 0.84, 0.72], bullets, fontsize=15)
    pages.append(fig)

    pages.append(references())

    DECK.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(DECK) as pdf:
        for fig in pages:
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
    print("Wrote", DECK, "with", len(pages), "pages")
    return 0


if __name__ == "__main__":
    sys.exit(main())