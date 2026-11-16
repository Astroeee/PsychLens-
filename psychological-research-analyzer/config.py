"""Central configuration: paths, colors, navigation, and default settings."""
from pathlib import Path

BASE_DIR = Path(__file__).parent
ASSETS_DIR = BASE_DIR / "assets"
DATA_DIR = BASE_DIR / "data"
SAMPLE_FILE = DATA_DIR / "sample_psych_study.csv"

APP_NAME = "PsychLens"
APP_TAGLINE = "Psychological research analytics"
SAMPLE_DATASET_NAME = "Sample - Student Wellbeing Study (simulated)"

COLORS = {
    "primary": "#1D66D6",
    "primary_soft": "#EAF1FC",
    "accent": "#2E9E6B",
    "bg": "#F7F8FA",
    "border": "#E8EAEE",
    "text": "#1A1D23",
    "muted": "#6B7280",
    "danger": "#D6453D",
    "warning": "#E0A030",
}

# Colorway used for every chart (kept short: restraint over decoration)
PALETTE = ["#1D66D6", "#2E9E6B", "#7AA7EA", "#E0A030", "#8B6FD6", "#D6453D"]

# (title, page file, material icon)
NAV_MAIN = [
    ("Dashboard", "pages/1_Dashboard.py", ":material/dashboard:"),
    ("Data Management", "pages/2_Data_Management.py", ":material/table_chart:"),
    ("Descriptive Statistics", "pages/3_Descriptive_Statistics.py", ":material/functions:"),
    ("Reliability Analysis", "pages/4_Reliability_Analysis.py", ":material/verified:"),
    ("Correlation Analysis", "pages/5_Correlation_Analysis.py", ":material/scatter_plot:"),
    ("Hypothesis Testing", "pages/6_Hypothesis_Testing.py", ":material/science:"),
    ("Data Visualization", "pages/7_Data_Visualization.py", ":material/bar_chart:"),
    ("Report Generator", "pages/8_Report_Generator.py", ":material/description:"),
]
NAV_SECONDARY = [
    ("Account", "pages/10_Account.py", ":material/person:"),
    ("Settings", "pages/9_Settings.py", ":material/settings:"),
]

DEFAULT_SETTINGS = {
    "researcher_name": "Researcher",
    "institution": "",
    "alpha": 0.05,
    "confidence": 0.95,
    "decimals": 2,
}

MAX_UPLOAD_MB = 50
