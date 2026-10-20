"""Shared clinical-style design tokens and CSS for the Streamlit interface."""

APP_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
  --clinical-navy: #0F172A;
  --clinical-blue: #2563EB;
  --clinical-sky: #0EA5E9;
  --clinical-muted: #64748B;
  --clinical-border: #E2E8F0;
  --clinical-surface: var(--secondary-background-color, #FFFFFF);
  --clinical-page: var(--background-color, #F5F7FB);
  --clinical-ink: var(--text-color, #0F172A);
}

html, body, [class*="css"] {
  font-family: 'Inter', sans-serif;
}
.stApp {
  background: var(--clinical-page);
}
[data-testid="stHeader"], footer, #MainMenu, [data-testid="stToolbar"],
[data-testid="stDecoration"], .stDeployButton, [data-testid="stStatusWidget"] {
  visibility: hidden;
  height: 0;
}
[data-testid="stMainBlockContainer"] {
  max-width: 1480px;
  padding-top: 1.3rem;
  padding-bottom: 2.5rem;
}
[data-testid="stVerticalBlockBorderWrapper"] {
  border-radius: 14px;
  border-color: var(--clinical-border);
  background: var(--clinical-surface);
  box-shadow: 0 4px 16px rgba(15, 23, 42, 0.045);
  transition: box-shadow 160ms ease, transform 160ms ease;
}
[data-testid="stVerticalBlockBorderWrapper"]:hover {
  box-shadow: 0 8px 24px rgba(15, 23, 42, 0.085);
  transform: translateY(-1px);
}
.hero {
  display: flex;
  align-items: center;
  gap: 1.25rem;
  color: #fff;
  padding: 2rem 2.2rem;
  margin: 0 0 1.1rem;
  border-radius: 16px;
  background: linear-gradient(120deg, #0F172A 0%, #132B4D 55%, #075985 100%);
  box-shadow: 0 12px 28px rgba(15, 23, 42, 0.18);
}
.hero-mark {
  flex: 0 0 66px;
  width: 66px;
  height: 66px;
  display: grid;
  place-items: center;
  border: 1px solid rgba(255,255,255,.22);
  border-radius: 18px;
  background: rgba(255,255,255,.1);
  font-size: 2rem;
}
.eyebrow {
  color: #7DD3FC;
  font-size: .72rem;
  font-weight: 700;
  letter-spacing: .13em;
  margin-bottom: .45rem;
}
.hero h1 {
  color: #fff;
  font-size: clamp(1.7rem, 3vw, 2.35rem);
  font-weight: 800;
  letter-spacing: -.04em;
  line-height: 1.13;
  margin: 0;
}
.hero p {
  color: #CBD5E1;
  margin: .6rem 0 .9rem;
  font-size: .98rem;
}
.hero-badges {
  display: flex;
  flex-wrap: wrap;
  gap: .5rem;
}
.hero-badges span {
  color: #E0F2FE;
  background: rgba(255,255,255,.1);
  border: 1px solid rgba(255,255,255,.16);
  border-radius: 999px;
  padding: .27rem .68rem;
  font-size: .74rem;
  font-weight: 600;
}
.clinical-notice {
  display: flex;
  align-items: center;
  gap: .7rem;
  background: rgba(14, 165, 233, .075);
  border: 1px solid rgba(14, 165, 233, .2);
  border-radius: 11px;
  padding: .72rem 1rem;
  margin-bottom: 1.4rem;
  color: var(--clinical-ink);
  font-size: .86rem;
}
.clinical-notice > span {
  color: var(--clinical-sky);
  font-size: 1.2rem;
  font-weight: 700;
}
.result-summary {
  display: flex;
  align-items: center;
  gap: .9rem;
  padding: .4rem 0 .6rem;
}
.result-summary.normal { color: #15803D; }
.result-summary.abnormal { color: #DC2626; }
.result-icon {
  width: 48px;
  height: 48px;
  display: grid;
  place-items: center;
  border-radius: 14px;
  background: currentColor;
  color: #fff;
  font-size: 1.5rem;
  font-weight: 800;
}
.result-kicker {
  font-size: .68rem;
  letter-spacing: .1em;
  font-weight: 700;
  color: var(--clinical-muted);
}
.result-label {
  font-size: 1.65rem;
  font-weight: 800;
  line-height: 1.2;
}
.status-chip {
  display: inline-block;
  color: #92400E;
  background: #FEF3C7;
  border: 1px solid #FCD34D;
  border-radius: 999px;
  padding: .35rem .72rem;
  font-size: .78rem;
  font-weight: 700;
}
.attention-legend {
  display: flex;
  align-items: center;
  gap: .7rem;
  justify-content: center;
  margin: .8rem 0 1rem;
  color: var(--clinical-muted);
  font-size: .75rem;
}
.attention-gradient {
  width: min(340px, 45vw);
  height: 9px;
  border-radius: 99px;
  background: linear-gradient(90deg, #313695, #74add1, #ffffbf, #f46d43, #a50026);
}
.empty-state {
  text-align: center;
  margin: 1.2rem 0;
  padding: 3.2rem 1.5rem;
  border: 1px dashed #94A3B8;
  border-radius: 16px;
  background: var(--clinical-surface);
}
.empty-icon { font-size: 2.7rem; }
.empty-state h3 {
  margin: .7rem 0 .3rem;
  color: var(--clinical-ink);
}
.empty-state p, .empty-state span {
  color: var(--clinical-muted);
}
.empty-state span { font-size: .78rem; }
.app-footer {
  text-align: center;
  color: var(--clinical-muted);
  border-top: 1px solid var(--clinical-border);
  padding: 1.2rem .5rem .2rem;
  margin-top: 2.2rem;
  font-size: .76rem;
}
button[kind="primary"], [data-testid="stDownloadButton"] button {
  border-radius: 10px;
  font-weight: 600;
  transition: transform 140ms ease, box-shadow 140ms ease;
}
button[kind="primary"]:hover, [data-testid="stDownloadButton"] button:hover {
  transform: translateY(-1px);
  box-shadow: 0 5px 14px rgba(37, 99, 235, .18);
}
[data-testid="stFileUploader"] section {
  border: 1.5px dashed #7DD3FC;
  border-radius: 14px;
  background: rgba(14, 165, 233, .035);
}
[data-testid="stTabs"] button[role="tab"] {
  font-weight: 600;
}
[data-testid="stSidebar"] {
  background: #0F172A;
}
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3, [data-testid="stSidebar"] label,
[data-testid="stSidebar"] p, [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
  color: #E2E8F0;
}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
  color: #94A3B8;
}
[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"] {
  background: rgba(255,255,255,.055);
  border-color: rgba(255,255,255,.13);
  box-shadow: none;
}
[data-testid="stSidebar"] [data-baseweb="select"] > div,
[data-testid="stSidebar"] [data-testid="stSlider"] {
  color: #F8FAFC;
}
@media (max-width: 700px) {
  .hero { align-items: flex-start; padding: 1.35rem; gap: .85rem; }
  .hero-mark { flex-basis: 48px; width: 48px; height: 48px; font-size: 1.5rem; border-radius: 14px; }
  .hero-break { display: none; }
  .hero p { font-size: .88rem; }
  .clinical-notice { align-items: flex-start; }
}
"""
