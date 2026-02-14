"""
Styles module for Streamlit ML Application
Contains all CSS styling for the app
"""

def get_custom_css():
    """Returns custom CSS styling for the Streamlit app"""
    return """
    <style>
    /* Main title styling */
    .main-title {
        font-size: 2.8rem;
        font-weight: 700;
        color: #1e3a8a;
        text-align: center;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .subtitle {
        font-size: 1.2rem;
        color: #64748b;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Navigation styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background-color: #f8fafc;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0 2rem;
        background-color: white;
        border-radius: 8px;
        font-weight: 600;
        font-size: 1.1rem;
        color: #475569;
        border: 2px solid transparent;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: 2px solid #667eea;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.8rem;
        font-weight: 600;
        color: #1e293b;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #667eea;
    }
    
    /* Info box */
    .info-box {
        background-color: #eff6ff;
        border-left: 4px solid #3b82f6;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    /* Success box */
    .success-box {
        background-color: #f0fdf4;
        border-left: 4px solid #22c55e;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    /* Warning box */
    .warning-box {
        background-color: #fefce8;
        border-left: 4px solid #eab308;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    /* Error box */
    .error-box {
        background-color: #fef2f2;
        border-left: 4px solid #ef4444;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    /* Data table styling */
    .dataframe {
        font-size: 0.95rem !important;
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        border: none;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(102, 126, 234, 0.4);
    }
    
    /* Download button */
    .stDownloadButton>button {
        background: linear-gradient(135deg, #22c55e 0%, #16a34a 100%);
        color: white;
        font-weight: 600;
        border: none;
        padding: 0.75rem 2rem;
        font-size: 1rem;
        border-radius: 8px;
    }
    
    /* Gradient boxes */
    .gradient-box-purple {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
    }
    
    .gradient-box-pink {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
    }
    
    .gradient-box-blue {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
    }
    
    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """

# Color schemes
COLORS = {
    'primary': '#667eea',
    'secondary': '#764ba2',
    'success': '#22c55e',
    'warning': '#eab308',
    'danger': '#ef4444',
    'info': '#3b82f6',
    'best_model': '#22c55e',
    'default_model': '#3b82f6',
    'text_dark': '#1e293b',
    'text_light': '#64748b',
}

# Chart color palettes
CHART_COLORS = {
    'metrics': ['#3b82f6', '#8b5cf6', '#ec4899', '#f59e0b'],
    'auc_excellent': '#22c55e',
    'auc_good': '#eab308',
    'auc_poor': '#ef4444',
    'confusion_matrix': 'RdYlGn',
}