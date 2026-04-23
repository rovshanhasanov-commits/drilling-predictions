"""Custom CSS for the Streamlit app."""

CSS = """
<style>
    .block-container { padding-top: 1.5rem; }

    .stDataFrame [data-testid="stDataFrameResizable"] { width: 100%; }
    th { white-space: nowrap !important; font-size: 0.85rem !important; }
    td { font-size: 0.85rem !important; }

    [data-testid="column"] > div > div > div > div > h3 {
        text-align: center;
        border-bottom: 2px solid #4a90d9;
        padding-bottom: 0.4rem;
        margin-bottom: 0.8rem;
    }

    [data-testid="stAlert"] p { font-size: 0.9rem; }
</style>
"""
