# Enhanced Fake Job Detector with Professional UI
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import re
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input, LSTM, GRU, Bidirectional, Dense, Embedding, Dropout, Concatenate, Attention, MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D, GlobalMaxPooling1D, Conv1D
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

# Set page config with professional theme
st.set_page_config(
    page_title="JobFraud Shield - AI-Powered Job Scam Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS with modern design
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary: #2563eb;
        --secondary: #64748b;
        --success: #10b981;
        --warning: #f59e0b;
        --danger: #ef4444;
        --dark: #1e293b;
        --light: #f8fafc;
    }
    
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.5rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 1rem;
        padding: 1rem;
    }
    
    .sub-header {
        color: var(--secondary);
        text-align: center;
        font-size: 1.2rem;
        margin-bottom: 2rem;
        font-weight: 300;
    }
    
    /* Card styling */
    .card {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        border: 1px solid #e2e8f0;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    }
    
    /* Risk level styling */
    .risk-low {
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        border-left: 5px solid var(--success);
    }
    
    .risk-medium {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        border-left: 5px solid var(--warning);
    }
    
    .risk-high {
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        border-left: 5px solid var(--danger);
    }
    
    /* Metric cards */
    .metric-card {
        text-align: center;
        padding: 1rem;
        border-radius: 10px;
        background: white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: var(--secondary);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Button styling */
    .stButton button {
        width: 100%;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .test-button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
    }
    
    .analyze-button {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        border: none;
        font-size: 1.1rem;
        padding: 0.75rem;
    }
    
    /* Form styling */
    .stTextInput input, .stTextArea textarea {
        border-radius: 8px;
        border: 2px solid #e2e8f0;
        transition: border-color 0.3s ease;
    }
    
    .stTextInput input:focus, .stTextArea textarea:focus {
        border-color: var(--primary);
        box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1);
    }
    
    /* Progress bar */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Custom tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f1f5f9;
        border-radius: 8px 8px 0px 0px;
        gap: 1px;
        padding: 10px 20px;
        margin: 0 2px;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: var(--primary);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Advanced Model Architectures (Functionality unchanged)
class AdvancedJobDetector:
    def __init__(self):
        self.models = {}
        self.tokenizer = None
        self.max_length = 200
        self.embedding_dim = 128
        
    def create_transformer_model(self, vocab_size):
        """Create Transformer-based model"""
        text_input = Input(shape=(self.max_length,), name='text_input')
        embedding = Embedding(vocab_size, self.embedding_dim, name='embedding')(text_input)
        x = LayerNormalization()(embedding)
        x = MultiHeadAttention(num_heads=4, key_dim=self.embedding_dim)(x, x)
        x = LayerNormalization()(x + embedding)
        x = GlobalAveragePooling1D()(x)
        numeric_input = Input(shape=(7,), name='numeric_input')
        numeric_dense = Dense(16, activation='relu')(numeric_input)
        combined = Concatenate()([x, numeric_dense])
        x = Dense(64, activation='relu')(combined)
        x = Dropout(0.4)(x)
        x = Dense(32, activation='relu')(x)
        x = Dropout(0.3)(x)
        output = Dense(1, activation='sigmoid')(x)
        model = Model(inputs=[text_input, numeric_input], outputs=output)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
        return model
    
    def create_hybrid_model(self, vocab_size):
        """Create Hybrid CNN+RNN model"""
        text_input = Input(shape=(self.max_length,), name='text_input')
        embedding = Embedding(vocab_size, self.embedding_dim)(text_input)
        conv1 = Conv1D(64, 3, activation='relu', padding='same')(embedding)
        conv2 = Conv1D(64, 5, activation='relu', padding='same')(embedding)
        conv3 = Conv1D(64, 7, activation='relu', padding='same')(embedding)
        cnn_output = Concatenate()([conv1, conv2, conv3])
        cnn_output = GlobalMaxPooling1D()(cnn_output)
        rnn_output = Bidirectional(LSTM(64, return_sequences=True))(embedding)
        rnn_output = GlobalAveragePooling1D()(rnn_output)
        text_features = Concatenate()([cnn_output, rnn_output])
        numeric_input = Input(shape=(7,), name='numeric_input')
        numeric_dense = Dense(16, activation='relu')(numeric_input)
        combined = Concatenate()([text_features, numeric_dense])
        x = Dense(64, activation='relu')(combined)
        x = Dropout(0.4)(x)
        x = Dense(32, activation='relu')(x)
        x = Dropout(0.3)(x)
        output = Dense(1, activation='sigmoid')(x)
        model = Model(inputs=[text_input, numeric_input], outputs=output)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
        return model

# Enhanced text cleaning and fraud detection (Functionality unchanged)
def clean_text(text):
    if not text or pd.isna(text):
        return "", 0, 0, 0, 0
    text = str(text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\{.*?\}', '', text)
    text = re.sub(r'\(.*?\)', '', text)
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    money_phrases = len(re.findall(r'\$[\d,]+|\d+\s*(dollars|usd|money|cash|salary|compensation|pay|earning|income)', text.lower()))
    urgency_phrases = len(re.findall(r'urgent|immediate|quick|fast|hiring now|limited time|apply now|start today|instant|asap', text.lower()))
    exclamation_marks = text.count('!')
    question_marks = text.count('?')
    text = re.sub(r'[^a-zA-Z\s\.\,\!]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.lower()
    return text, money_phrases, urgency_phrases, exclamation_marks, question_marks

def rule_based_fraud_detection(job_data):
    score = 0.0
    combined_text = f"{job_data['title']} {job_data['description']} {job_data['requirements']} {job_data['company_profile']}"
    cleaned_text, money_phrases, urgency_phrases, exclamation_marks, question_marks = clean_text(combined_text)
    indicators = {
        'telecommuting': 0.3 if job_data.get('telecommuting', 0) else 0,
        'no_company_logo': 0.2 if not job_data.get('has_company_logo', 1) else 0,
        'no_questions': 0.1 if not job_data.get('has_questions', 1) else 0,
        'money_mentions': min(money_phrases * 0.15, 0.4),
        'urgency_phrases': min(urgency_phrases * 0.1, 0.3),
        'exclamation_marks': min(exclamation_marks * 0.05, 0.2),
        'short_description': 0.1 if len(cleaned_text) < 100 else 0,
        'vague_company': 0.2 if len(job_data.get('company_profile', '')) < 50 else 0
    }
    total_score = sum(indicators.values())
    probability = 1 / (1 + np.exp(-5 * (total_score - 0.5)))
    return probability, indicators

@st.cache_resource
def load_enhanced_models():
    try:
        detector = AdvancedJobDetector()
        if os.path.exists('models/tokenizer.pkl'):
            detector.tokenizer = joblib.load('models/tokenizer.pkl')
        else:
            st.warning("⚠️ Tokenizer not found. Using rule-based detection only.")
            detector.tokenizer = None
        if detector.tokenizer:
            vocab_size = min(10000, len(detector.tokenizer.word_index) + 1)
            with st.spinner("Creating AI models..."):
                detector.models['transformer'] = detector.create_transformer_model(vocab_size)
                detector.models['hybrid'] = detector.create_hybrid_model(vocab_size)
            for model_name in ['transformer', 'hybrid']:
                model_path = f'models/{model_name}_model.h5'
                if os.path.exists(model_path):
                    try:
                        detector.models[model_name].load_weights(model_path)
                    except:
                        st.warning(f"⚠️ Could not load weights for {model_name}")
        return detector
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        return None

def enhanced_predict(detector, job_data, model_type='ensemble', threshold=0.5):
    try:
        rule_probability, rule_indicators = rule_based_fraud_detection(job_data)
        if detector.tokenizer is None or not detector.models:
            return {
                'success': True, 'probability': rule_probability,
                'model_predictions': {'rule_based': rule_probability},
                'feature_importance': rule_indicators, 'is_fake': rule_probability > threshold, 'method': 'rule_based'
            }
        combined_text = f"{job_data['title']} {job_data['description']} {job_data['requirements']} {job_data['company_profile']}"
        cleaned_text, money_phrases, urgency_phrases, exclamation_marks, question_marks = clean_text(combined_text)
        sequences = detector.tokenizer.texts_to_sequences([cleaned_text])
        text_features = pad_sequences(sequences, maxlen=detector.max_length, padding='post', truncating='post')
        numeric_features = np.array([[
            job_data.get('telecommuting', 0), job_data.get('has_company_logo', 0), job_data.get('has_questions', 0),
            min(money_phrases / 5, 1.0), min(urgency_phrases / 5, 1.0), min(exclamation_marks / 10, 1.0), min(question_marks / 5, 1.0)
        ]])
        predictions = {}
        for model_name, model in detector.models.items():
            try:
                pred = model.predict([text_features, numeric_features], verbose=0)[0][0]
                predictions[model_name] = float(pred)
            except:
                predictions[model_name] = rule_probability
        if model_type == 'ensemble' and len(predictions) > 0:
            weights = {name: 1.0/len(predictions) for name in predictions.keys()}
            final_prediction = sum(predictions[model] * weights[model] for model in predictions.keys())
        elif predictions:
            final_prediction = predictions.get(model_type, rule_probability)
        else:
            final_prediction = rule_probability
        if predictions:
            final_prediction = 0.7 * final_prediction + 0.3 * rule_probability
        feature_importance = {
            'telecommuting': job_data.get('telecommuting', 0), 'has_company_logo': job_data.get('has_company_logo', 0),
            'has_questions': job_data.get('has_questions', 0), 'money_mentions': money_phrases,
            'urgency_signals': urgency_phrases, 'exclamation_marks': exclamation_marks,
            'question_marks': question_marks, 'text_length': len(cleaned_text),
            'company_profile_length': len(job_data.get('company_profile', ''))
        }
        return {
            'success': True, 'probability': final_prediction, 'model_predictions': predictions,
            'feature_importance': feature_importance, 'is_fake': final_prediction > threshold, 'method': 'ai_with_rules' if predictions else 'rule_based'
        }
    except Exception as e:
        rule_probability, rule_indicators = rule_based_fraud_detection(job_data)
        return {
            'success': True, 'probability': rule_probability,
            'model_predictions': {'rule_based_fallback': rule_probability},
            'feature_importance': rule_indicators, 'is_fake': rule_probability > threshold, 'method': 'rule_based_fallback'
        }

TEST_CASES = {
    'legitimate': {
        'title': 'Senior Software Engineer', 'telecommuting': False, 'has_company_logo': True, 'has_questions': True,
        'description': 'We are looking for a skilled Senior Software Engineer to join our dynamic team. You will design, develop, and maintain high-quality software solutions.',
        'requirements': 'Bachelor\'s degree in Computer Science, 5+ years professional experience, Proficiency in Python or Java',
        'company_profile': 'TechInnovate is a leading technology company established in 2010 with over 500 employees.'
    },
    'suspicious': {
        'title': 'Work From Home Data Entry Clerk - $5000/month', 'telecommuting': True, 'has_company_logo': False, 'has_questions': False,
        'description': 'EASY MONEY FROM HOME!!! No experience needed!! Immediate hiring!! You can earn up to $5000 per month working just 2 hours daily!!',
        'requirements': 'Basic computer skills, Internet connection, No education required, No experience needed',
        'company_profile': 'Global Data Solutions Inc. We are a fast-growing company looking for motivated individuals.'
    },
    'fake': {
        'title': 'URGENT HIRING!!! $8000 Weekly - Online Assistant', 'telecommuting': True, 'has_company_logo': False, 'has_questions': False,
        'description': '$$$ MAKE FAST CASH $$$ IMMEDIATE PAYMENT!!! We need online assistants for simple tasks. Earn $8000 per week working from your phone!!',
        'requirements': 'Must have smartphone, Must be 18+ years old, No background check, No resume needed',
        'company_profile': 'QuickCash Enterprises. International company offering amazing opportunities.'
    }
}

def create_risk_gauge(probability):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = probability,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "FRAUD RISK SCORE", 'font': {'size': 20}},
        delta = {'reference': 0.5, 'increasing': {'color': "#ef4444"}, 'decreasing': {'color': "#10b981"}},
        gauge = {
            'axis': {'range': [0, 1], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 0.3], 'color': '#d1fae5'},
                {'range': [0.3, 0.7], 'color': '#fef3c7'},
                {'range': [0.7, 1], 'color': '#fee2e2'}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 0.7}}))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
    return fig

def create_feature_radar(features):
    categories = ['Money Mentions', 'Urgency Signals', 'Exclamations', 'Telecommuting', 'No Logo', 'No Questions']
    values = [
        min(features.get('money_mentions', 0) / 5, 1),
        min(features.get('urgency_signals', 0) / 5, 1),
        min(features.get('exclamation_marks', 0) / 10, 1),
        features.get('telecommuting', 0),
        1 - features.get('has_company_logo', 1),
        1 - features.get('has_questions', 1)
    ]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=values, theta=categories, fill='toself', name='Risk Factors',
                                 line=dict(color='#2563eb'), fillcolor='rgba(37, 99, 235, 0.2)'))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=False, height=300)
    return fig

def main():
    # Header Section
    st.markdown('<h1 class="main-header">🛡️ JobFraud Shield</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Job Scam Detection & Prevention System</p>', unsafe_allow_html=True)
    
    # Initialize session state
    for key in ['title', 'description', 'requirements', 'company_profile', 'telecommuting', 'has_company_logo', 'has_questions']:
        if key not in st.session_state:
            st.session_state[key] = TEST_CASES['legitimate'][key] if key in TEST_CASES['legitimate'] else ''
    
    # Sidebar
    with st.sidebar:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### ⚙️ Detection Settings")
        model_type = st.selectbox("**Detection Engine**", ['ensemble', 'rule_based'], 
                                help="Ensemble: AI + Rules | Rule-based: Fast heuristic analysis")
        threshold = st.slider("**Risk Threshold**", 0.1, 0.9, 0.5, 0.05,
                            help="Higher values reduce false positives but may miss some scams")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 🚀 Quick Tests")
        for case_name, case_data in TEST_CASES.items():
            if st.button(f"🧪 {case_name.replace('_', ' ').title()}", key=f"sidebar_{case_name}", use_container_width=True):
                for key, value in case_data.items():
                    st.session_state[key] = value
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 📊 System Info")
        detector = load_enhanced_models()
        if detector:
            if detector.tokenizer:
                st.success("✅ AI Models: Active")
                st.info(f"🤖 Models: {len(detector.models)} loaded")
            else:
                st.warning("⚠️ AI Models: Rule-based only")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Main Content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 📝 Job Posting Analysis")
        
        tab1, tab2 = st.tabs(["🔍 Detailed Analysis", "⚡ Quick Scan"])
        
        with tab1:
            title = st.text_input("**Job Title**", value=st.session_state.title, 
                                placeholder="e.g., Senior Software Engineer")
            description = st.text_area("**Job Description**", value=st.session_state.description, height=120,
                                     placeholder="Detailed responsibilities and role description...")
            requirements = st.text_area("**Requirements**", value=st.session_state.requirements, height=100,
                                      placeholder="Required skills, experience, and qualifications...")
            company_profile = st.text_area("**Company Profile**", value=st.session_state.company_profile, height=80,
                                         placeholder="Information about the hiring company...")
            
            st.markdown("### 🔍 Risk Indicators")
            col1a, col2a, col3a = st.columns(3)
            with col1a:
                telecommuting = st.checkbox("**Remote Work**", value=st.session_state.telecommuting)
            with col2a:
                has_company_logo = st.checkbox("**Company Logo**", value=st.session_state.has_company_logo)
            with col3a:
                has_questions = st.checkbox("**Application Questions**", value=st.session_state.has_questions)
        
        with tab2:
            st.info("Paste the complete job description for quick analysis:")
            quick_text = st.text_area("**Full Job Text**", height=150,
                                    placeholder="Paste the entire job posting here for comprehensive analysis...")
            if quick_text:
                # Simple parsing for quick scan
                st.session_state.title = quick_text.split('.')[0][:100]
                st.session_state.description = quick_text
                st.session_state.requirements = "Quick scan mode"
        
        analyze_clicked = st.button("**🔍 Analyze Job Posting**", use_container_width=True, type="primary")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 📈 Live Analysis")
        
        if analyze_clicked or any(st.session_state.get(key) for key in ['title', 'description']):
            # Update session state
            job_data = {
                'title': title, 'description': description, 'requirements': requirements,
                'company_profile': company_profile or '', 'telecommuting': 1 if telecommuting else 0,
                'has_company_logo': 1 if has_company_logo else 0, 'has_questions': 1 if has_questions else 0
            }
            
            with st.spinner("🛡️ Scanning for fraud indicators..."):
                result = enhanced_predict(detector, job_data, model_type, threshold)
            
            if result['success']:
                probability = result['probability']
                
                # Risk classification
                if probability > 0.7:
                    risk_class, risk_text, emoji, color = "risk-high", "HIGH RISK", "🚨", "#ef4444"
                elif probability > 0.4:
                    risk_class, risk_text, emoji, color = "risk-medium", "MEDIUM RISK", "⚠️", "#f59e0b"
                else:
                    risk_class, risk_text, emoji, color = "risk-low", "LOW RISK", "✅", "#10b981"
                
                # Risk Gauge
                st.plotly_chart(create_risk_gauge(probability), use_container_width=True)
                
                # Metrics
                col1b, col2b, col3b = st.columns(3)
                with col1b:
                    st.markdown(f'<div class="metric-card"><div class="metric-label">Risk Level</div><div class="metric-value" style="color: {color}">{emoji}</div><div>{risk_text}</div></div>', unsafe_allow_html=True)
                with col2b:
                    confidence = (1 - abs(probability - 0.5) * 2) * 100
                    st.markdown(f'<div class="metric-card"><div class="metric-label">Confidence</div><div class="metric-value">{confidence:.1f}%</div></div>', unsafe_allow_html=True)
                with col3b:
                    st.markdown(f'<div class="metric-card"><div class="metric-label">Method</div><div class="metric-value">{"AI" if "ai" in result["method"] else "Rules"}</div></div>', unsafe_allow_html=True)
                
                # Feature Radar
                st.plotly_chart(create_feature_radar(result['feature_importance']), use_container_width=True)
                
                # Detailed Analysis
                with st.expander("📋 Detailed Risk Analysis", expanded=True):
                    features = result['feature_importance']
                    risk_factors = []
                    
                    if features.get('telecommuting', 0):
                        risk_factors.append(("Remote Work Opportunity", "High risk factor for scams", "#ef4444"))
                    if not features.get('has_company_logo', 1):
                        risk_factors.append(("No Company Logo", "Legitimate companies typically have branding", "#f59e0b"))
                    if features.get('money_mentions', 0) > 1:
                        risk_factors.append((f"Money Mentions ({features['money_mentions']})", "Excessive focus on earnings", "#f59e0b"))
                    if features.get('urgency_signals', 0) > 0:
                        risk_factors.append((f"Urgency Signals ({features['urgency_signals']})", "Pressure tactics detected", "#f59e0b"))
                    
                    if risk_factors:
                        st.warning("**🚨 Risk Factors Detected:**")
                        for factor, description, color in risk_factors:
                            st.markdown(f"<div style='color: {color}; margin: 5px 0;'>• <strong>{factor}</strong>: {description}</div>", unsafe_allow_html=True)
                    else:
                        st.success("**✅ No significant risk factors detected**")
                    
                    if probability > threshold:
                        st.error(f"**Recommendation:** Exercise extreme caution. This job posting shows {risk_text.lower()} characteristics.")
                    else:
                        st.success("**Recommendation:** This job appears legitimate. Continue with standard precautions.")
        
        else:
            st.info("👆 Enter job details and click 'Analyze' to begin scanning")
            st.plotly_chart(create_risk_gauge(0.0), use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("<div style='text-align: center; color: #64748b; font-size: 0.9rem;'>"
                "🛡️ JobFraud Shield • AI-Powered Protection • "
                f"© {datetime.now().year} All rights reserved</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()