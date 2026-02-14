import os
import subprocess
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils import load_models, preprocess_uploaded_data, predict_with_model, calculate_metrics
from styles import get_custom_css, COLORS, CHART_COLORS
from constants import (APP_CONFIG, MODEL_DESCRIPTIONS, METRIC_INFO, METRICS_TO_HIGHLIGHT,METRIC_FORMAT, CHART_CONFIG, 
                       ANALYSIS_TEXT, UI_MESSAGES,TEST_DATA_REQUIREMENTS, FILE_UPLOAD_INFO, AUC_THRESHOLDS, 
                       TAB_NAMES, SECTION_HEADERS, PREDICTION_LABELS)
import warnings
warnings.filterwarnings('ignore')

# Page configuration - MUST BE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title=APP_CONFIG['title'],
    layout=APP_CONFIG['layout'],
    initial_sidebar_state="collapsed"
)

# Check if models exist
if not os.path.exists('models/logistic_regression.pkl'):
    st.info("Training models for the first time... This may take a few minutes.")
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    try:
        result = subprocess.run(['python', 'src/model_training.py'], 
                              capture_output=True, 
                              text=True)
        
        if result.returncode == 0:
            st.success("✅ Models trained successfully!")
            st.info("Please click the button below to load the application.")
            if st.button("🔄 Refresh Application", type="primary"):
                st.rerun()
            st.stop()
        else:
            st.error("❌ Training failed!")
            with st.expander("View Error Details"):
                st.code(result.stderr)
            st.stop()
            
    except Exception as e:
        st.error(f"Error during training: {str(e)}")
        st.stop()

# Apply custom CSS
st.markdown(get_custom_css(), unsafe_allow_html=True)

def plot_confusion_matrix(cm, title="Confusion Matrix"):
    """Plot confusion matrix with better styling"""
    config = CHART_CONFIG['confusion_matrix']
    fig, ax = plt.subplots(figsize=config['figsize'])
    sns.heatmap(cm, annot=True, fmt='d', cmap=config['cmap'], ax=ax,
                xticklabels=config['labels'], yticklabels=config['labels'],
                cbar_kws={'label': 'Count'}, annot_kws={'size': 14, 'weight': 'bold'})
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    return fig

def display_metric_cards(metrics):
    """Display metrics in beautiful cards"""
    cols = st.columns(6)
    
    for col, (label) in zip(cols, METRIC_INFO):
        with col:
            value = metrics[label]
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{value:.4f}</div>
                    <div class="metric-label">{label}</div>
                </div>
            """, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

def render_training_results(models, scaler, feature_names, saved_metrics, saved_cm):
    """Render the training results tab"""
    st.markdown(f'<h2 class="section-header">{SECTION_HEADERS["performance_overview"]}</h2>', 
                unsafe_allow_html=True)
    
    # Best model highlight
    best_model_idx = saved_metrics['Accuracy'].idxmax()
    best_model_name = saved_metrics.loc[best_model_idx, 'Model']
    best_accuracy = saved_metrics.loc[best_model_idx, 'Accuracy']
    best_auc = saved_metrics.loc[best_model_idx, 'AUC']
    
    st.markdown(f"""
        <div class="success-box">
            <h3 style="margin:0; color:#16a34a;">🏆 Best Performing Model</h3>
            <p style="font-size:1.3rem; margin:0.5rem 0; font-weight:600; color:#15803d;">
                {best_model_name}
            </p>
            <p style="margin:0; color:#166534;">
                Accuracy: <strong>{best_accuracy:.4f}</strong> | AUC: <strong>{best_auc:.4f}</strong>
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Metrics Comparison Table
    st.markdown(f'<h3 class="section-header">{SECTION_HEADERS["metrics_table"]}</h3>', 
                unsafe_allow_html=True)
    
    def highlight_max(s):
        is_max = s == s.max()
        return ['background-color: #d4edda; font-weight: bold' if v else '' for v in is_max]
    
    styled_df = saved_metrics.style.apply(highlight_max, subset=METRICS_TO_HIGHLIGHT)\
                                  .format(METRIC_FORMAT)
    
    st.dataframe(styled_df, use_container_width=True, height=280)
    
    # Model Analysis - All Models
    st.markdown(f'<h3 class="section-header">{SECTION_HEADERS["model_analysis"]}</h3>', 
                unsafe_allow_html=True)
    
    # Display all models
    for model_name in models.keys():
        st.markdown(f"---")
        st.markdown(f"### {model_name}")
        
        # Get metrics for this model
        model_metrics = saved_metrics[saved_metrics['Model'] == model_name].iloc[0]
        
        # Display metrics in cards
        display_metric_cards(model_metrics)
        
        # Model description and confusion matrix
        col1, col2 = st.columns([1.2, 1])
        
        with col1:
            st.markdown("#### Model Description & Insights")
            info = MODEL_DESCRIPTIONS[model_name]
            
            st.markdown(f"""
                <div class="info-box">
                    <p style="margin:0; font-size:1rem; line-height:1.6;">
                        <strong>Description:</strong><br>{info['desc']}
                    </p>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"**Strengths:**  \n{info['strengths']}")
            st.markdown(f"**Weaknesses:**  \n{info['weaknesses']}")
            st.markdown(f"**Best Use Case:**  \n{info['use_case']}")
        
        with col2:
            st.markdown("#### Confusion Matrix")
            if model_name in saved_cm:
                fig = plot_confusion_matrix(saved_cm[model_name], f"{model_name}")
                st.pyplot(fig)
                plt.close()
    
    # Performance Visualizations
    st.markdown(f'<h3 class="section-header">{SECTION_HEADERS["performance_viz"]}</h3>', 
                unsafe_allow_html=True)
    
    viz_col1, viz_col2 = st.columns(2)
    
    with viz_col1:
        st.markdown("#### 🎯 Model Accuracy Comparison")
        st.markdown(f'<div class="warning-box">{ANALYSIS_TEXT["accuracy"]}</div>', 
                   unsafe_allow_html=True)
        
        config = CHART_CONFIG['accuracy_comparison']
        fig1, ax1 = plt.subplots(figsize=config['figsize'])
        colors = [COLORS['best_model'] if model == best_model_name else COLORS['default_model']
                 for model in saved_metrics['Model']]
        bars = ax1.bar(saved_metrics['Model'], saved_metrics['Accuracy'], 
                      color=colors, edgecolor='black', linewidth=1.5)
        ax1.set_ylabel(config['ylabel'], fontsize=12, fontweight='bold')
        ax1.set_xlabel(config['xlabel'], fontsize=12, fontweight='bold')
        ax1.set_title(config['title'], fontsize=14, fontweight='bold', pad=15)
        ax1.set_ylim([saved_metrics['Accuracy'].min() - 0.05, 
                     saved_metrics['Accuracy'].max() + 0.05])
        plt.xticks(rotation=45, ha='right')
        
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}',
                    ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        plt.tight_layout()
        st.pyplot(fig1)
    
    with viz_col2:
        st.markdown("#### 📊 Multi-Metric Performance Radar")
        st.markdown(f'<div class="warning-box">{ANALYSIS_TEXT["multi_metric"]}</div>', 
                   unsafe_allow_html=True)
        
        config = CHART_CONFIG['multi_metric']
        fig2, ax2 = plt.subplots(figsize=config['figsize'])
        metrics_to_plot = saved_metrics.set_index('Model')[config['metrics']]
        metrics_to_plot.plot(kind='bar', ax=ax2, width=0.8, color=CHART_COLORS['metrics'])
        ax2.set_ylabel(config['ylabel'], fontsize=12, fontweight='bold')
        ax2.set_xlabel(config['xlabel'], fontsize=12, fontweight='bold')
        ax2.set_title(config['title'], fontsize=14, fontweight='bold', pad=15)
        ax2.legend(loc='lower right', framealpha=0.9, fontsize=10)
        ax2.set_ylim(config['ylim'])
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3, linestyle='--')
        plt.tight_layout()
        st.pyplot(fig2)
    
    # AUC Comparison
    st.markdown("#### 🎪 AUC Score Analysis")
    st.markdown(f'<div class="warning-box">{ANALYSIS_TEXT["auc"]}</div>', 
               unsafe_allow_html=True)
    
    config = CHART_CONFIG['auc_comparison']
    fig3, ax3 = plt.subplots(figsize=config['figsize'])
    colors_auc = [CHART_COLORS['auc_excellent'] if auc >= AUC_THRESHOLDS['excellent'] 
                  else CHART_COLORS['auc_good'] if auc >= AUC_THRESHOLDS['good'] 
                  else CHART_COLORS['auc_poor']
                  for auc in saved_metrics['AUC']]
    bars_auc = ax3.barh(saved_metrics['Model'], saved_metrics['AUC'], 
                       color=colors_auc, edgecolor='black', linewidth=1.5)
    ax3.set_xlabel(config['xlabel'], fontsize=12, fontweight='bold')
    ax3.set_title(config['title'], fontsize=14, fontweight='bold', pad=15)
    ax3.set_xlim(config['xlim'])
    
    for bar in bars_auc:
        width = bar.get_width()
        ax3.text(width, bar.get_y() + bar.get_height()/2.,
                f'{width:.4f}',
                ha='left', va='center', fontweight='bold', fontsize=11, 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    plt.grid(axis='x', alpha=0.3, linestyle='--')
    plt.tight_layout()
    st.pyplot(fig3)
    

def render_model_testing(models, scaler, feature_names):
    """Render the model testing tab"""
    st.markdown(f'<h2 class="section-header">{SECTION_HEADERS["test_models"]}</h2>', 
                unsafe_allow_html=True)
    
    # Model selection
    col1, col2 = st.columns([2, 1])
    with col1:
        test_model_name = st.selectbox(
            "Select Model for Testing",
            list(models.keys()),
            key="test_model_select"
        )
    with col2:
        st.markdown(f"""
            <div style="background-color:#eff6ff; padding:1rem; border-radius:8px; margin-top:1.7rem;">
                <p style="margin:0; font-size:0.9rem; color:#1e40af;">
                    <strong>Selected:</strong> {test_model_name}
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    selected_test_model = models[test_model_name]
    
    # File upload section
    st.markdown(f"### {SECTION_HEADERS['upload_data']}")
    st.markdown(f'<div class="info-box"><p style="margin:0;">{FILE_UPLOAD_INFO}</p></div>', 
               unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'], key="test_file")
    
    if uploaded_file is not None:
        try:
            test_df = pd.read_csv(uploaded_file)
            
            st.markdown(f"""
                <div class="success-box">
                    <p style="margin:0;">
                        {UI_MESSAGES['file_upload_success']}<br>
                        Shape: <code>{test_df.shape[0]} rows × {test_df.shape[1]} columns</code>
                    </p>
                </div>
            """, unsafe_allow_html=True)
            
            with st.expander("👀 Preview Data (First 10 rows)"):
                st.dataframe(test_df.head(10), use_container_width=True)
            
            if st.button("🚀 Run Prediction", key="predict_btn"):
                with st.spinner(UI_MESSAGES['processing']):
                    try:
                        X_test, y_true = preprocess_uploaded_data(test_df, scaler, feature_names)
                        predictions, probabilities = predict_with_model(selected_test_model, X_test)
                        
                        results_df = test_df.copy()
                        results_df['Prediction'] = [PREDICTION_LABELS[p] for p in predictions]
                        
                        if probabilities is not None:
                            results_df['Confidence_No'] = (probabilities[:, 0] * 100).round(2)
                            results_df['Confidence_Yes'] = (probabilities[:, 1] * 100).round(2)
                        
                        st.markdown(f'<h3 class="section-header">{SECTION_HEADERS["prediction_results"]}</h3>', 
                                   unsafe_allow_html=True)
                        
                        # Summary statistics
                        pred_counts = pd.Series(predictions).value_counts()
                        total = len(predictions)
                        
                        sum_col1, sum_col2, sum_col3 = st.columns(3)
                        with sum_col1:
                            st.metric("📊 Total Predictions", total)
                        with sum_col2:
                            st.metric("✅ Predicted Yes", 
                                    f"{pred_counts.get(1, 0)} ({pred_counts.get(1, 0)/total*100:.1f}%)")
                        with sum_col3:
                            st.metric("❌ Predicted No", 
                                    f"{pred_counts.get(0, 0)} ({pred_counts.get(0, 0)/total*100:.1f}%)")
                        
                        st.dataframe(results_df.head(20), use_container_width=True)
                        
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Full Predictions as CSV",
                            data=csv,
                            file_name=f"{test_model_name.replace(' ', '_')}_predictions.csv",
                            mime="text/csv"
                        )
                        
                        if y_true is not None:
                            st.markdown(f'<h3 class="section-header">{SECTION_HEADERS["model_evaluation"]}</h3>', 
                                       unsafe_allow_html=True)
                            
                            metrics, cm, cr = calculate_metrics(y_true, predictions, probabilities)
                            display_metric_cards(metrics)
                            
                            eval_col1, eval_col2 = st.columns(2)
                            
                            with eval_col1:
                                st.markdown("#### 🔢 Confusion Matrix")
                                fig = plot_confusion_matrix(cm, f"{test_model_name}")
                                st.pyplot(fig)
                            
                            with eval_col2:
                                st.markdown("#### 📋 Classification Report")
                                cr_df = pd.DataFrame(cr).transpose()
                                st.dataframe(cr_df.style.format("{:.4f}"), use_container_width=True)
                    
                    except Exception as e:
                        st.error(UI_MESSAGES['prediction_error'].format(str(e)))
                        st.info(UI_MESSAGES['ensure_features'])
        
        except Exception as e:
            st.error(UI_MESSAGES['file_load_error'].format(str(e)))
    
    else:
        req = TEST_DATA_REQUIREMENTS
        st.markdown(f"""
            <div class="info-box">
                <h4 style="margin:0 0 0.5rem 0;">{req['title']}</h4>
                <p style="margin:0;">{req['description']}</p>
                <code style="display:block; background:#f1f5f9; padding:0.5rem; border-radius:4px; margin-top:0.5rem;">
                """ + ", ".join(feature_names) + """
                </code>
                <p style="margin-top:0.5rem; font-size:0.9rem;">
                """ + req['optional_note'] + """
                </p>
            </div>
        """, unsafe_allow_html=True)

def main():
    # Header
    st.markdown(f'<h1 class="main-title">🎓 {APP_CONFIG["title"]}</h1>', 
                unsafe_allow_html=True)
    
    # Load models
    model_data = load_models()
    
    if model_data is None:
        st.error(UI_MESSAGES['models_not_found'])
        st.info(UI_MESSAGES['models_not_found_hint'])
        return
    
    models = model_data['models']
    scaler = model_data['scaler']
    feature_names = model_data['feature_names']
    saved_metrics = model_data['metrics']
    saved_cm = model_data['confusion_matrices']
    
    # Top Navigation Tabs
    tab1, tab2 = st.tabs([TAB_NAMES['training'], TAB_NAMES['testing']])
    
    with tab1:
        render_training_results(models, scaler, feature_names, saved_metrics, saved_cm)
    
    with tab2:
        render_model_testing(models, scaler, feature_names)

if __name__ == "__main__":
    main()
