# Enhanced Fake Job Detector with REAL Job Scraping
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import re
import requests
import json
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import time
import random
from urllib.parse import quote, urlparse, urljoin, parse_qs
import urllib3
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import warnings
warnings.filterwarnings('ignore')

# Common user agents for basic rotation (helps with some blocking)
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/122.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 13_6) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15',
]

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Model paths
MODEL_PATH = os.path.join("models", "lstm_text_model.h5")
TOKENIZER_PATH = os.path.join("models", "tokenizer.pkl")
MAX_TEXT_LEN = 200
ML_THRESHOLD = 0.4


@st.cache_resource(show_spinner=False)
def load_fraud_model_assets():
    if not (os.path.exists(MODEL_PATH) and os.path.exists(TOKENIZER_PATH)):
        return None, None
    try:
        from tensorflow.keras.models import load_model
    except Exception:
        return None, None
    tokenizer = joblib.load(TOKENIZER_PATH)
    model = load_model(MODEL_PATH)
    return model, tokenizer


def predict_fraud_score(model, tokenizer, title, description):
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    combined_text = f"{title} {description}"
    seq = tokenizer.texts_to_sequences([combined_text])
    padded = pad_sequences(seq, maxlen=MAX_TEXT_LEN, padding='post', truncating='post')
    # metadata placeholders (unknown in UI)
    numeric_features = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    prob = float(model.predict([padded, numeric_features], verbose=0)[0][0])
    return prob

def _normalize_match_text(s):
    return re.sub(r'[^a-z0-9]+', '', (s or '').lower())

def _domain_matches(candidate, allowed_domains):
    cand = (candidate or '').lower().lstrip('www.')
    for d in allowed_domains:
        base = (d or '').lower().lstrip('www.')
        if not base:
            continue
        if cand == base or cand.endswith('.' + base):
            return True
    return False

def verify_job_url(job_url, job_title, allowed_domains):
    """Verify a specific job posting URL on official domains."""
    if not job_url:
        return {"status": "inconclusive", "reason": "No job URL provided"}

    url = job_url.strip()
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    try:
        parsed = urlparse(url)
        if not parsed.netloc:
            return {"status": "blocked", "reason": "Invalid job URL"}
        if not _domain_matches(parsed.netloc, allowed_domains):
            return {"status": "blocked", "reason": "Job URL domain does not match official career domains"}

        # URL-level verification (works even if page is JS-heavy)
        q = parse_qs(parsed.query)
        title_q = q.get("title", [""])[0]
        id_q = q.get("id", [""])[0]
        job_title_norm = _normalize_match_text(job_title)
        if title_q and _normalize_match_text(title_q) == job_title_norm:
            return {"status": "verified", "reason": "Title matched in URL parameters"}
        if job_title_norm and job_title_norm in _normalize_match_text(url):
            return {"status": "verified", "reason": "Title matched in URL"}
        # Accenture-style jobdetails URL with id parameter
        if "jobdetails" in parsed.path.lower() and id_q:
            return {"status": "verified", "reason": "Official jobdetails URL with id parameter"}

        resp = requests.get(url, timeout=12, verify=False, proxies=get_active_proxies())
        if resp.status_code != 200:
            return {"status": "inconclusive", "reason": f"HTTP {resp.status_code}"}

        content_type = (resp.headers.get("Content-Type") or "").lower()
        text = resp.text or ""
        if "application/json" in content_type:
            try:
                data = resp.json()
                blob = json.dumps(data).lower()
                if _normalize_match_text(job_title) in _normalize_match_text(blob):
                    return {"status": "verified", "reason": "Title matched in JSON response"}
            except Exception:
                pass

        if len(text) < 2000 or "enable javascript" in text.lower():
            return {"status": "inconclusive", "reason": "Page appears JS-heavy; cannot verify server-side"}

        if _normalize_match_text(job_title) in _normalize_match_text(text):
            return {"status": "verified", "reason": "Title matched in page content"}

        return {"status": "inconclusive", "reason": "Title not found on job page"}
    except Exception as e:
        return {"status": "inconclusive", "reason": str(e)}

def _extract_first_url(text):
    if not text:
        return ""
    # Basic URL extraction
    m = re.search(r'(https?://[^\s)]+)', text)
    return m.group(1) if m else ""

def _extract_accenture_job_id(text):
    if not text:
        return ""
    # Match IDs like ATCI-5483975-S2003396 or ATCI-5483975-S2003396_en
    m = re.search(r'\bATCI-\d+-S\d+(?:_en)?\b', text)
    return m.group(0) if m else ""

def get_active_proxies():
    """Return proxies dict from Streamlit session state, if configured."""
    try:
        if st.session_state.get("use_proxy"):
            proxies = {}
            http_p = st.session_state.get("proxy_http", "").strip()
            https_p = st.session_state.get("proxy_https", "").strip()
            if http_p:
                proxies["http"] = http_p
            if https_p:
                proxies["https"] = https_p
            return proxies or None
    except Exception:
        pass
    return None

def _extract_email_domains(text):
    if not text:
        return []
    emails = re.findall(r'[A-Za-z0-9._%+-]+@([A-Za-z0-9.-]+\.[A-Za-z]{2,})', text)
    return list({e.lower().lstrip('www.') for e in emails})

def _extract_urls(text):
    if not text:
        return []
    return re.findall(r'https?://[^\s)]+', text)

def _normalize_domain(d):
    return (d or '').lower().lstrip('www.')

def _domain_in_list(domain, allowed):
    domain = _normalize_domain(domain)
    for a in allowed:
        base = _normalize_domain(a)
        if not base:
            continue
        if domain == base or domain.endswith('.' + base):
            return True
    return False

ATS_PROVIDERS = [
    "myworkdayjobs.com",
    "workday.com",
    "greenhouse.io",
    "lever.co",
    "icims.com",
    "oraclecloud.com",
    "taleo.net",
    "successfactors.com",
]

def verify_ats_url(job_url, company_name):
    """Check if URL belongs to a common ATS provider and contains company hints."""
    if not job_url:
        return {"status": "inconclusive", "reason": "No URL provided"}
    try:
        parsed = urlparse(job_url)
        host = _normalize_domain(parsed.netloc)
        if not host:
            return {"status": "blocked", "reason": "Invalid URL"}
        if not _domain_in_list(host, ATS_PROVIDERS):
            return {"status": "inconclusive", "reason": "Not an ATS provider domain"}
        # Try to see company name in path or query
        company_norm = _normalize_match_text(company_name)
        url_norm = _normalize_match_text(job_url)
        if company_norm and company_norm in url_norm:
            return {"status": "verified", "reason": "ATS URL contains company name"}
        return {"status": "inconclusive", "reason": "ATS URL detected; company not verified in URL"}
    except Exception as e:
        return {"status": "inconclusive", "reason": str(e)}

def build_verification_checklist(company_result, job_url, job_title, job_description):
    items = []
    # Official domain / URL checks
    allowed_domains = []
    try:
        if company_result.get("careers_url"):
            allowed_domains.append(urlparse(company_result["careers_url"]).netloc)
    except Exception:
        pass
    try:
        if company_result.get("website"):
            allowed_domains.append(urlparse(company_result["website"]).netloc)
    except Exception:
        pass

    url_check = {"status": "inconclusive", "reason": "No URL provided"}
    if job_url:
        url_check = verify_job_url(job_url, job_title, allowed_domains)
    items.append(("Official job URL", url_check))

    # ATS validation
    ats_check = {"status": "inconclusive", "reason": "No URL provided"}
    if job_url:
        ats_check = verify_ats_url(job_url, company_result.get("company_name", ""))
    items.append(("ATS provider check", ats_check))

    # Email domain check
    emails = _extract_email_domains(job_description)
    if not emails:
        items.append(("Recruiter email domain", {"status": "inconclusive", "reason": "No email found"}))
    else:
        ok = any(_domain_in_list(e, allowed_domains) for e in emails)
        status = "verified" if ok else "blocked"
        reason = "Matches official domain" if ok else "Email domain does not match official domain"
        items.append(("Recruiter email domain", {"status": status, "reason": reason, "emails": emails}))

    # Fraud keywords already handled earlier; keep as info here
    items.append(("Fraud keyword scan", {"status": "inconclusive", "reason": "See red-flag section above"}))

    return items

def compute_overall_verdict(matching_jobs, scrape_ok, checklist, fraud_score=0, company_verified=False, dynamic_site=False):
    """Return verdict string: VERIFIED, GENUINE_LIKELY, UNVERIFIED, INCONCLUSIVE, HIGH_RISK."""
    # Strong positive signals
    if matching_jobs:
        return "VERIFIED"
    if any(item[1].get("status") == "verified" for item in checklist):
        return "VERIFIED"
    # High fraud score overrides
    if fraud_score >= 70:
        return "HIGH_RISK"
    # Strong negative signals
    if any(item[1].get("status") == "blocked" for item in checklist):
        return "HIGH_RISK"
    # Company is verified but portal is JS-heavy          cannot scrape, but company is REAL
    # This is the key fix: don't label real jobs as INCONCLUSIVE just because we can't scrape
    if company_verified and dynamic_site and not scrape_ok:
        return "GENUINE_LIKELY"
    # If we scraped and did not find the role, mark unverified (not fake)
    if scrape_ok:
        return "UNVERIFIED"
    return "INCONCLUSIVE"

class FraudSignalScorer:
    """Multi-signal fraud scoring (no external APIs)."""

    def __init__(self):
        self.keywords = [
            'urgent', 'urgently hiring', 'earn $', 'earn rs', 'per day', 'per week',
            'work from phone', 'no experience needed', 'no interview', 'weekly payment',
            'daily payment', 'whatsapp', 'telegram', 'guaranteed income', 'easy money',
            'get rich', 'registration fee', 'pay to apply', 'processing fee', 'deposit required',
            'send money', 'upfront payment', 'mlm', 'joining fee', 'advance payment',
            'lottery', 'prize money', 'investment required', 'jackpot',
            'click here to apply', 'limited seats', 'limited openings', 'work from home',
            'data entry', 'typing job', 'part time', 'no skills required', 'only phone',
            'paytm wallet', 'upi', 'bank transfer', 'telegram channel', 'whatsapp group'
        ]
        self.personal_data_terms = [
            'aadhaar', 'aadhar', 'pan card', 'pan number', 'passport', 'bank account',
            'ifsc', 'upi id', 'otp', 'cvv', 'debit card', 'credit card'
        ]
        self.free_email_domains = [
            'gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com', 'live.com', 'rediffmail.com'
        ]

    def _salary_risk(self, text):
        # Very simple salary sanity check
        patterns = [
            r'(\d{4,7})\s*(?:per|/)\s*day',
            r'(\d{5,8})\s*(?:per|/)\s*month',
            r'(\d{6,9})\s*(?:per|/)\s*year',
        ]
        for pat in patterns:
            for m in re.findall(pat, text):
                try:
                    val = int(m.replace(',', ''))
                except Exception:
                    continue
                if 'day' in pat and val >= 5000:
                    return 1.0
                if 'month' in pat and val >= 500000:
                    return 1.0
                if 'year' in pat and val >= 10000000:
                    return 0.8
        return 0.0

    def score(self, company_name, job_title, job_description, allowed_domains):
        text = (job_title + " " + job_description).lower()
        score = 0.0
        breakdown = []

        # Fraud keywords
        kw_hits = [kw for kw in self.keywords if kw in text]
        kw_factor = min(1.0, len(kw_hits) / 3.0)
        score += 35 * kw_factor
        breakdown.append(("Fraud keywords", 35 * kw_factor, kw_hits[:5]))

        # Salary sanity
        sal_factor = self._salary_risk(text)
        score += 15 * sal_factor
        breakdown.append(("Salary sanity", 15 * sal_factor, []))

        # Company profile quality (very short or missing company name)
        desc_len = len(job_description.strip())
        quality_factor = 1.0 if desc_len < 300 else 0.5 if desc_len < 600 else 0.0
        if company_name and company_name.lower() not in text:
            quality_factor = max(quality_factor, 0.5)
        score += 15 * quality_factor
        breakdown.append(("Company/profile quality", 15 * quality_factor, []))

        # Contact method
        contact_factor = 1.0 if ("whatsapp" in text or "telegram" in text) else 0.0
        score += 10 * contact_factor
        breakdown.append(("Contact method", 10 * contact_factor, []))

        # Description quality
        desc_quality = 1.0 if desc_len < 200 else 0.5 if desc_len < 400 else 0.0
        score += 10 * desc_quality
        breakdown.append(("Description quality", 10 * desc_quality, []))

        # Personal data request
        pdata_hits = [t for t in self.personal_data_terms if t in text]
        pdata_factor = 1.0 if pdata_hits else 0.0
        score += 10 * pdata_factor
        breakdown.append(("Personal data requests", 10 * pdata_factor, pdata_hits[:5]))

        # Email domain check
        emails = _extract_email_domains(job_description)
        email_factor = 0.0
        if emails:
            for e in emails:
                if e in self.free_email_domains:
                    email_factor = 1.0
                    break
                if allowed_domains and not _domain_in_list(e, allowed_domains):
                    email_factor = max(email_factor, 0.5)
        score += 5 * email_factor
        breakdown.append(("Email domain", 5 * email_factor, emails[:3]))

        score = min(100, round(score, 1))
        return score, breakdown

# Set page config
st.set_page_config(
    page_title="JobVerification AI - Real Job Openings Finder",
    page_icon="                                   ",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS
st.markdown("""
<style>
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
    
    .ai-badge {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-left: 10px;
    }
    
    .learning-badge {
        background: linear-gradient(135deg, #8b5cf6 0%, #a855f7 100%);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-left: 10px;
    }
    
    .sub-header {
        color: var(--secondary);
        text-align: center;
        font-size: 1.2rem;
        margin-bottom: 2rem;
        font-weight: 300;
    }
    
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
    
    .auto-learn-card {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        border: 2px solid #0ea5e9;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    .discovery-success {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        border: 2px solid #22c55e;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    .discovery-failure {
        background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
        border: 2px solid #ef4444;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    .jobs-found-card {
        background: linear-gradient(135deg, #dbeafe 0%, #93c5fd 100%);
        border: 2px solid #3b82f6;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    .no-jobs-card {
        background: linear-gradient(135deg, #f1f5f9 0%, #e2e8f0 100%);
        border: 2px solid #94a3b8;
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
        text-align: center;
    }
    
    .job-listing {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        border: 1px solid #e2e8f0;
        transition: all 0.3s ease;
    }
    
    .job-listing:hover {
        transform: translateX(5px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    
    .job-title {
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 0.5rem;
        font-size: 1.1rem;
    }
    
    .job-meta {
        font-size: 0.9rem;
        color: #64748b;
        margin-bottom: 0.5rem;
    }
    
    .job-link {
        color: #2563eb;
        text-decoration: none;
        font-size: 0.9rem;
        font-weight: 500;
    }
    
    .job-link:hover {
        text-decoration: underline;
    }
    
    .verified { background: #d1fae5; color: #065f46; border: 1px solid #a7f3d0; }
    .unverified { background: #fef3c7; color: #92400e; border: 1px solid #fde68a; }
    .suspicious { background: #fee2e2; color: #991b1b; border: 1px solid #fecaca; }
    
    .metric-card {
        text-align: center;
        padding: 1rem;
        border-radius: 10px;
        background: white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border: 1px solid #e2e8f0;
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 0.8rem;
        color: var(--secondary);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .neuron {
        width: 20px;
        height: 20px;
        border-radius: 50%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        margin: 5px;
        display: inline-block;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); opacity: 0.7; }
        50% { transform: scale(1.1); opacity: 1; }
        100% { transform: scale(1); opacity: 0.7; }
    }
    
    .confidence-bar {
        height: 20px;
        background: #e2e8f0;
        border-radius: 10px;
        margin: 10px 0;
        overflow: hidden;
    }
    
    .confidence-fill {
        height: 100%;
        border-radius: 10px;
        transition: width 0.5s ease;
    }
    
    .high-confidence { background: linear-gradient(90deg, #10b981, #34d399); }
    .medium-confidence { background: linear-gradient(90deg, #f59e0b, #fbbf24); }
    .low-confidence { background: linear-gradient(90deg, #ef4444, #f87171); }
    
    .genuine-card {
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        border: 3px solid #10b981;
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
        text-align: center;
    }
    
    .fake-card {
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        border: 3px solid #ef4444;
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ==================== COMPANY DISCOVERY ENGINE ====================

class CompanyDiscoveryEngine:
    # Multi-strategy company discovery engine.

    EXCLUDED_DOMAINS = {
        'google.com', 'bing.com', 'duckduckgo.com', 'yahoo.com',
        'facebook.com', 'linkedin.com', 'twitter.com', 'youtube.com',
        'instagram.com', 'wikipedia.org', 'glassdoor.com', 'naukri.com',
        'indeed.com', 'monster.com', 'ambitionbox.com', 'quora.com',
        'reddit.com', 'crunchbase.com', 'bloomberg.com', 'forbes.com',
        'techcrunch.com', 'economictimes.com', 'businessstandard.com',
    }

    CAREER_PATHS = [
        '/careers', '/jobs', '/career', '/careers.html', '/jobs.html',
        '/work-with-us', '/join-us', '/hiring', '/vacancies', '/join',
        '/opportunities', '/recruitment', '/company/careers',
        '/about/careers', '/en/careers', '/us/careers', '/in/careers',
    ]

    CAREER_KEYWORDS = ['careers', 'jobs', 'employment', 'work with us',
                       'hiring', 'join us', 'openings', 'vacancies', 'apply now']

    def __init__(self):
        self.session = self._create_session()
        # Last scrape diagnostics to avoid false "fake" labels
        self.last_error = None
        self.last_status_code = None
        self.last_fetch_ok = False
        self.last_dynamic_site = False

    def _create_session(self):
        # Create a requests session with browser-like headers
        import requests
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry

        session = requests.Session()
        retry = Retry(total=2, backoff_factor=0.5,
                      status_forcelist=[429, 500, 502, 503, 504])
        adapter = HTTPAdapter(max_retries=retry)
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        session.headers.update({
            'User-Agent': (
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                'AppleWebKit/537.36 (KHTML, like Gecko) '
                'Chrome/122.0.0.0 Safari/537.36'
            ),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })
        return session

    def discover_company(self, company_name):
        # Try to find the company website + career page
        result = {
            'company_name': company_name,
            'found': False,
            'careers_url': None,
            'website': None,
            'confidence': 0,
            'discovery_method': None,
            'validation_details': [],
        }

        website_url = self._guess_website_directly(company_name)
        if website_url:
            result['discovery_method'] = 'direct_url_guess'
            result['validation_details'].append(f"Strategy 1 succeeded: {website_url}")
        else:
            result['validation_details'].append('Strategy 1 (direct URL) failed - trying DuckDuckGo...')
            website_url = self._search_duckduckgo(company_name)
            if website_url:
                result['discovery_method'] = 'duckduckgo_search'
                result['validation_details'].append(f"Strategy 2 (DuckDuckGo) found: {website_url}")
            else:
                result['validation_details'].append('Strategy 2 (DuckDuckGo) also failed.')
                return result

        result['website'] = website_url

        careers_url = self._find_careers_page(website_url)
        if not careers_url:
            result['validation_details'].append('Website found but no careers page detected.')
            return result

        result['careers_url'] = careers_url
        result['validation_details'].append(f"Careers page found: {careers_url}")

        score = self._score_careers_page(careers_url)
        result['confidence'] = round(score * 100, 1)
        result['validation_details'].append(f"Validation score: {score:.0%}")

        if score >= 0.5:
            result['found'] = True
        else:
            result['validation_details'].append('Score below 50% threshold - not confident enough.')

        return result

    def _guess_website_directly(self, company_name):
        # Construct candidate URLs from the company name and probe them directly.
        words = company_name.lower().split()
        stop_words = {'limited', 'ltd', 'pvt', 'private', 'inc', 'corp',
                      'corporation', 'technologies', 'technology', 'solutions',
                      'systems', 'services', 'group', 'india', 'global',
                      'international', 'enterprises', 'consulting', 'the'}
        core_words = [w for w in words if w not in stop_words]

        name_variants = list(dict.fromkeys([
            ''.join(core_words),
            core_words[0] if core_words else '',
            company_name.lower().replace(' ', ''),
            company_name.lower().replace(' ', '-'),
        ]))
        name_variants = [v for v in name_variants if v]

        tlds = ['.com', '.in', '.io', '.co', '.net', '.org']
        candidates = []
        for variant in name_variants:
            for tld in tlds:
                candidates.append(f'https://www.{variant}{tld}')
                candidates.append(f'https://{variant}{tld}')

        for url in candidates:
            try:
                self.session.headers['User-Agent'] = random.choice(USER_AGENTS)
                resp = self.session.get(url, timeout=(5, 10), verify=False,
                                        proxies=get_active_proxies(),
                                        allow_redirects=True)
                if resp.status_code == 200 and len(resp.content) > 500:
                    final_url = resp.url
                    if self._is_plausible_company_site(final_url, company_name, resp.text):
                        return final_url
            except Exception:
                continue

        return None

    def _is_plausible_company_site(self, url, company_name, html_text=''):
        # Heuristic: does this URL look like the company's own website?
        from urllib.parse import urlparse
        try:
            domain = urlparse(url).netloc.lower().replace('www.', '')
        except Exception:
            return False

        for excl in self.EXCLUDED_DOMAINS:
            if excl in domain:
                return False

        words = company_name.lower().split()
        stop_words = {'limited', 'ltd', 'pvt', 'private', 'inc', 'corp',
                      'corporation', 'technologies', 'technology', 'solutions',
                      'systems', 'services', 'group', 'india', 'global',
                      'international', 'enterprises', 'consulting', 'the'}
        core_words = [w for w in words if w not in stop_words and len(w) >= 3]

        return any(w in domain for w in core_words)

    def _search_duckduckgo(self, company_name):
        # Search DuckDuckGo (HTML version) for the company's official website.
        from urllib.parse import quote, unquote, parse_qs
        queries = [
            f'{company_name} official website',
            f'{company_name} careers site:*.com OR site:*.in OR site:*.io',
        ]

        for query in queries:
            try:
                url = f'https://duckduckgo.com/html/?q={quote(query)}'
                self.session.headers['User-Agent'] = random.choice(USER_AGENTS)
                resp = self.session.get(url, timeout=(5, 12), verify=False, proxies=get_active_proxies())
                if resp.status_code != 200:
                    continue

                from bs4 import BeautifulSoup
                soup = BeautifulSoup(resp.text, 'html.parser')

                for tag in soup.select('a.result__a, .result__url, a[href*="http"]'):
                    href = tag.get('href', '')
                    if 'uddg=' in href:
                        qs = parse_qs(href.split('?', 1)[-1])
                        href = unquote(qs.get('uddg', [''])[0])
                    if href.startswith('http') and self._is_plausible_company_site(href, company_name):
                        try:
                            self.session.headers['User-Agent'] = random.choice(USER_AGENTS)
                            r2 = self.session.get(href, timeout=8, verify=False, proxies=get_active_proxies())
                            if r2.status_code == 200 and len(r2.content) > 300:
                                return r2.url
                        except Exception:
                            continue
            except Exception:
                continue

        return None

    def _find_careers_page(self, website_url):
        # 1) Scan homepage links. 2) Probe common paths.
        from urllib.parse import urljoin, urlparse
        from bs4 import BeautifulSoup

        try:
            self.session.headers['User-Agent'] = random.choice(USER_AGENTS)
            resp = self.session.get(website_url, timeout=(5, 10), verify=False, proxies=get_active_proxies())
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, 'html.parser')
                for a in soup.find_all('a', href=True):
                    href = a.get('href', '').lower()
                    text = a.get_text(strip=True).lower()
                    if (any(kw in href for kw in ['career', 'job', 'hiring', 'join', 'work-with'])
                            or any(kw in text for kw in self.CAREER_KEYWORDS)):
                        candidate = urljoin(website_url, a.get('href', ''))
                        if self._is_careers_url_valid(candidate):
                            return candidate
        except Exception:
            pass

        base = '{uri.scheme}://{uri.netloc}'.format(uri=urlparse(website_url))
        for path in self.CAREER_PATHS:
            candidate = base + path
            if self._is_careers_url_valid(candidate):
                return candidate

        return None

    def _is_careers_url_valid(self, url):
        # Return True if the URL is accessible and looks like a careers page.
        try:
            self.session.headers['User-Agent'] = random.choice(USER_AGENTS)
            resp = self.session.get(url, timeout=(5, 8), verify=False, proxies=get_active_proxies())
            if resp.status_code == 200 and len(resp.content) > 300:
                text = resp.text.lower()
                hits = sum(1 for kw in ['job', 'career', 'apply', 'position',
                                        'opening', 'hire', 'opportunity', 'vacancy']
                           if kw in text)
                return hits >= 2
        except Exception:
            pass
        return False

    def _score_careers_page(self, careers_url):
        # Give a 0-1 confidence score that the careers_url is a real company careers page.
        from bs4 import BeautifulSoup
        score = 0.0
        try:
            self.session.headers['User-Agent'] = random.choice(USER_AGENTS)
            resp = self.session.get(careers_url, timeout=10, verify=False, proxies=get_active_proxies())
            if resp.status_code == 200:
                score += 0.4
                soup = BeautifulSoup(resp.text, 'html.parser')
                text = soup.get_text().lower()

                job_terms = ['apply', 'position', 'opening', 'vacancy', 'career',
                             'job', 'hire', 'opportunity', 'full-time', 'part-time']
                hits = sum(1 for t in job_terms if t in text)
                score += min(hits * 0.06, 0.3)

                if soup.find('nav') or len(soup.find_all('ul')) > 2:
                    score += 0.15
                if soup.find('footer'):
                    score += 0.1
                if soup.find('title') and soup.find('title').string:
                    title = soup.find('title').string.lower()
                    if any(t in title for t in ['careers', 'jobs', 'work with us']):
                        score += 0.05
        except Exception:
            pass
        return min(score, 1.0)


class RealJobScraper:
    def __init__(self):
        self.session = self._create_session()
        # Last scrape diagnostics to avoid false "fake" labels
        self.last_error = None
        self.last_status_code = None
        self.last_fetch_ok = False
        self.last_dynamic_site = False

    def _create_session(self):
        import requests
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry

        session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=0.7,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        session.headers.update({
            'User-Agent': random.choice(USER_AGENTS),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })
        return session

    def _reset_diagnostics(self):
        self.last_error = None
        self.last_status_code = None
        self.last_fetch_ok = False
        self.last_dynamic_site = False

    def _detect_dynamic_site(self, html_text):
        # Heuristic: if it is heavily JS-driven, scraping may be incomplete
        lowered = html_text.lower()
        signals = [
            'data-reactroot', 'ng-version', 'window.__initial_state__',
            'react-root', 'vue-app', 'angular', 'application/json',
        ]
        return any(s in lowered for s in signals)

    def _extract_jobs_from_jsonld(self, html_text, base_url=None):
        jobs = []
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html_text, 'html.parser')
            for script in soup.select('script[type="application/ld+json"]'):
                raw = script.string or script.get_text(strip=True)
                if not raw:
                    continue
                try:
                    data = json.loads(raw)
                except Exception:
                    continue
                items = data if isinstance(data, list) else [data]
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    if item.get('@type') != 'JobPosting':
                        continue
                    title = item.get('title') or item.get('name')
                    if not title:
                        continue
                    url = item.get('url') or base_url
                    location = None
                    job_loc = item.get('jobLocation')
                    if isinstance(job_loc, dict):
                        addr = job_loc.get('address', {}) if isinstance(job_loc.get('address'), dict) else {}
                        parts = [addr.get('addressLocality'), addr.get('addressRegion'), addr.get('addressCountry')]
                        location = ', '.join([p for p in parts if p]) if any(parts) else None
                    elif isinstance(job_loc, list) and job_loc:
                        loc0 = job_loc[0]
                        if isinstance(loc0, dict):
                            addr = loc0.get('address', {}) if isinstance(loc0.get('address'), dict) else {}
                            parts = [addr.get('addressLocality'), addr.get('addressRegion'), addr.get('addressCountry')]
                            location = ', '.join([p for p in parts if p]) if any(parts) else None

                    jobs.append({
                        'title': title.strip(),
                        'company': item.get('hiringOrganization', {}).get('name') if isinstance(item.get('hiringOrganization'), dict) else None,
                        'location': location,
                        'url': url,
                        'source': 'jsonld',
                    })
        except Exception:
            return jobs
        return jobs

    def _extract_jobs_from_links(self, soup, base_url):
        jobs = []
        from urllib.parse import urljoin
        for a in soup.find_all('a', href=True):
            href = a.get('href', '')
            text = a.get_text(strip=True)
            if not text:
                continue
            lower = (text + ' ' + href).lower()
            if any(k in lower for k in ['job', 'career', 'opening', 'position', 'apply', 'vacancy']):
                url = urljoin(base_url, href)
                jobs.append({
                    'title': text,
                    'company': None,
                    'location': None,
                    'url': url,
                    'source': 'links',
                })
        return jobs

    def _filter_by_title(self, jobs, search_title):
        if not search_title:
            return jobs
        q = re.sub(r'[^a-z0-9]+', ' ', search_title.lower()).strip()
        q_words = set(q.split())
        if not q_words:
            return jobs
        out = []
        for job in jobs:
            title = job.get('title', '')
            t = re.sub(r'[^a-z0-9]+', ' ', title.lower()).strip()
            if not t:
                continue
            t_words = set(t.split())
            if q in t or (q_words and len(q_words.intersection(t_words)) >= max(1, len(q_words)//2)):
                out.append(job)
        return out

    def scrape_real_jobs(self, company_name, careers_url, search_title=None):
        # Scrape REAL job openings from company career page
        self._reset_diagnostics()
        if not careers_url:
            self.last_error = 'No careers URL provided.'
            return []

        try:
            self.session.headers['User-Agent'] = random.choice(USER_AGENTS)
            resp = self.session.get(careers_url, timeout=(6, 14), verify=False, proxies=get_active_proxies())
            self.last_status_code = resp.status_code
            if resp.status_code != 200:
                self.last_error = f'HTTP {resp.status_code} when fetching careers page.'
                return []
        except Exception as e:
            self.last_error = str(e)
            return []

        self.last_fetch_ok = True
        html_text = resp.text or ''
        self.last_dynamic_site = self._detect_dynamic_site(html_text)

        jobs = []
        jobs.extend(self._extract_jobs_from_jsonld(html_text, base_url=resp.url))

        if not jobs:
            try:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(html_text, 'html.parser')
                jobs.extend(self._extract_jobs_from_links(soup, resp.url))
            except Exception:
                pass

        jobs = self._filter_by_title(jobs, search_title)

        # De-duplicate by title + url
        unique = []
        seen = set()
        for job in jobs:
            key = (job.get('title', '').lower(), job.get('url', ''))
            if key in seen:
                continue
            seen.add(key)
            if not job.get('company'):
                job['company'] = company_name
            unique.append(job)

        return unique


class SelfLearningCompanyVerifier:
    # Enhanced company verifier with automatic discovery capabilities
    
    def __init__(self):
        self.known_companies = self._load_company_database()
        self.discovery_engine = CompanyDiscoveryEngine()
        self.job_scraper = RealJobScraper()
        self.learning_enabled = True
        
    def _load_company_database(self):
        # Load comprehensive Indian company database
        try:
            if os.path.exists('company_database.json'):
                with open('company_database.json', 'r') as f:
                    return json.load(f)
        except:
            pass
        
        # Comprehensive Indian Companies Database
        return {
            # # ==================== IT & TECHNOLOGY COMPANIES ====================
            # 'tata consultancy services': 'https://www.tcs.com/careers',
            # 'tcs': 'https://www.tcs.com/careers',
            # 'infosys': 'https://www.infosys.com/careers',
            # 'wipro': 'https://careers.wipro.com',
            # 'wipro technologies': 'https://careers.wipro.com',
            # 'hcl technologies': 'https://www.hcltech.com/careers',
            # 'hcl': 'https://www.hcltech.com/careers',
            # 'hcltech': 'https://www.hcltech.com/careers',
            # 'tech mahindra': 'https://careers.techmahindra.com',
            
            # # Mid-tier IT Companies
            # 'ltimindtree': 'https://www.ltimindtree.com/careers',
            # 'l&t infotech': 'https://www.ltimindtree.com/careers',
            # 'lti': 'https://www.ltimindtree.com/careers',
            # 'mindtree': 'https://www.ltimindtree.com/careers',
            # 'mphasis': 'https://www.mphasis.com/careers.html',
            # 'persistent systems': 'https://www.persistent.com/careers',
            # 'persistent': 'https://www.persistent.com/careers',
            # 'coforge': 'https://www.coforge.com/careers',
            # 'niit technologies': 'https://www.coforge.com/careers',
            # 'hexaware technologies': 'https://www.hexaware.com/careers',
            # 'hexaware': 'https://www.hexaware.com/careers',
            # 'cyient': 'https://www.cyient.com/careers',
            # 'zensar technologies': 'https://www.zensar.com/careers',
            # 'zensar': 'https://www.zensar.com/careers',
            
            # # IT Consulting & Services
            # 'l&t technology services': 'https://www.lnttechservices.com/careers',
            # 'ltts': 'https://www.lnttechservices.com/careers',
            # 'tata elxsi': 'https://www.tataelxsi.com/careers',
            # 'kpit technologies': 'https://www.kpit.com/careers',
            # 'kpit': 'https://www.kpit.com/careers',
            # 'birlasoft': 'https://www.birlasoft.com/careers',
            # 'sonata software': 'https://www.sonata-software.com/careers',
            # 'newgen software': 'https://newgensoft.com/careers',
            # 'intellect design arena': 'https://www.intellectdesign.com/careers',
            # 'eclerx services': 'https://www.eclerx.com/careers',
            # 'eclerx': 'https://www.eclerx.com/careers',
            
            # # Product Companies
            # 'freshworks': 'https://www.freshworks.com/company/careers/',
            # 'zoho': 'https://www.zoho.com/careers/',
            # 'zoho corporation': 'https://www.zoho.com/careers/',
            # 'chargebee': 'https://www.chargebee.com/careers/',
            # 'postman': 'https://www.postman.com/company/careers/',
            # 'browserstack': 'https://www.browserstack.com/careers',
            # 'hasura': 'https://hasura.io/careers/',
            # 'innovaccer': 'https://innovaccer.com/careers/',
            # 'thoughtspot': 'https://www.thoughtspot.com/careers',
            # 'moglix': 'https://www.moglix.com/careers/',
            
            # # E-commerce & Internet
            # 'flipkart': 'https://www.flipkartcareers.com/',
            # 'flipkart internet': 'https://www.flipkartcareers.com/',
            # 'myntra': 'https://careers.myntra.com/',
            # 'nykaa': 'https://www.nykaa.com/careers',
            # 'nykaa fashion': 'https://www.nykaa.com/careers',
            # 'bigbasket': 'https://careers.bigbasket.com/',
            # 'snapdeal': 'https://careers.snapdeal.com/',
            # 'meesho': 'https://careers.meesho.com/',
            # 'swiggy': 'https://careers.swiggy.com/',
            # 'zomato': 'https://www.zomato.com/careers',
            # 'ola': 'https://www.olacabs.com/careers',
            # 'ola cabs': 'https://www.olacabs.com/careers',
            # 'ola electric': 'https://careers.olaelectric.com/',
            # 'rapido': 'https://www.rapido.bike/careers',
            # 'urban company': 'https://careers.urbancompany.com/',
            
            # # FinTech & Payments
            # 'paytm': 'https://paytm.com/careers/',
            # 'phonepe': 'https://www.phonepe.com/careers/',
            # 'razorpay': 'https://razorpay.com/careers/',
            # 'cred': 'https://careers.cred.club/',
            # 'billdesk': 'https://www.billdesk.com/careers/',
            # 'policybazaar': 'https://www.policybazaar.com/careers/',
            # 'bankbazaar': 'https://www.bankbazaar.com/careers.html',
            
            # # EdTech
            # 'byjus': 'https://byjus.com/careers/',
            # 'byju\'s': 'https://byjus.com/careers/',
            # 'unacademy': 'https://unacademy.com/careers/',
            # 'vedantu': 'https://www.vedantu.com/careers',
            # 'upgrad': 'https://www.upgrad.com/careers/',
            # 'great learning': 'https://www.greatlearning.in/careers',
            # 'whitehat jr': 'https://www.whitehatjr.com/careers/',
            
            # # HealthTech
            # 'practo': 'https://www.practo.com/careers',
            # 'pharmeasy': 'https://pharmeasy.in/careers/',
            # '1mg': 'https://www.1mg.com/careers',
            # 'netmeds': 'https://www.netmeds.com/careers',
            
            # # TravelTech
            # 'makemytrip': 'https://careers.makemytrip.com/',
            # 'goibibo': 'https://careers.goibibo.com/',
            # 'ixigo': 'https://www.ixigo.com/careers',
            # 'easemytrip': 'https://www.easemytrip.com/careers/',
            
            # # Media & Entertainment
            # 'hotstar': 'https://careers.hotstar.com/',
            # 'disney+ hotstar': 'https://careers.hotstar.com/',
            # 'zee entertainment': 'https://www.zee.com/careers',
            # 'sony pictures networks': 'https://www.sonypicturesnetworks.com/careers',
            # 'viacom18': 'https://www.viacom18.com/careers',
            
            # # Gaming
            # 'dream11': 'https://www.dreamsports.group/careers',
            # 'nazara technologies': 'https://www.nazara.com/careers',
            # 'mobile premier league': 'https://www.mpl.live/careers',
            # 'mpl': 'https://www.mpl.live/careers',
            
            # # ==================== BANKS & FINANCIAL INSTITUTIONS ====================
            # 'state bank of india': 'https://www.sbi.co.in/careers',
            # 'sbi': 'https://sbi.bank.in/web/careers',
            # 'punjab national bank': 'https://www.pnbindia.in/Careers.html',
            # 'pnb': 'https://www.pnbindia.in/Careers.html',
            # 'bank of baroda': 'https://www.bankofbaroda.in/careers',
            # 'bob': 'https://www.bankofbaroda.in/careers',
            # 'canara bank': 'https://canarabank.com/careers',
            # 'union bank of india': 'https://www.unionbankofindia.co.in/english/careers.aspx',
            
            # # Private Banks
            # 'hdfc bank': 'https://www.hdfcbank.com/personal/careers',
            # 'hdfc': 'https://www.hdfcbank.com/personal/careers',
            # 'icici bank': 'https://www.icicicareers.com/',
            # 'icici': 'https://www.icicicareers.com/',
            # 'axis bank': 'https://www.axisbank.com/careers',
            # 'kotak mahindra bank': 'https://www.kotak.com/en/careers.html',
            # 'kotak': 'https://www.kotak.com/en/careers.html',
            # 'yes bank': 'https://www.yesbank.in/careers',
            # 'indusind bank': 'https://www.indusind.com/in/en/careers.html',
            
            # # Financial Services
            # 'hdfc life': 'https://www.hdfclife.com/careers',
            # 'icici prudential': 'https://www.iciciprulife.com/careers/index.html',
            # 'bajaj finserv': 'https://www.bajajfinserv.in/careers',
            # 'bajaj finance': 'https://www.bajajfinserv.in/careers',
            # 'shriram finance': 'https://www.shriram.com/careers',
            # 'muthoot finance': 'https://www.muthootfinance.com/Careers',
            
            # # ==================== MANUFACTURING & CONGLOMERATES ====================
            # # Automobile
            # 'tata motors': 'https://www.tatamotors.com/careers/',
            # 'mahindra & mahindra': 'https://www.mahindra.com/careers',
            # 'mahindra': 'https://www.mahindra.com/careers',
            # 'maruti suzuki': 'https://www.marutisuzuki.com/careers',
            # 'bajaj auto': 'https://www.bajajauto.com/careers',
            # 'hero motocorp': 'https://www.heromotocorp.com/en-in/careers/',
            # 'tvs motor company': 'https://www.tvsmotor.com/careers',
            
            # # Conglomerates
            # 'reliance industries': 'https://careers.ril.com/',
            # 'reliance': 'https://careers.ril.com/',
            # 'reliance jio': 'https://careers.ril.com/jio/',
            # 'jio': 'https://careers.ril.com/jio/',
            # 'adani group': 'https://www.adanigroup.com/careers',
            # 'adani': 'https://www.adanigroup.com/careers',
            # 'vedanta resources': 'https://www.vedantaresources.com/careers.aspx',
            # 'vedanta': 'https://www.vedantaresources.com/careers.aspx',
            # 'larsen & toubro': 'https://www.larsentoubro.com/corporate/careers/',
            # 'l&t': 'https://www.larsentoubro.com/corporate/careers/',
            
            # # FMCG
            # 'itc limited': 'https://www.itcportal.com/careers/',
            # 'itc': 'https://www.itcportal.com/careers/',
            # 'hindustan unilever': 'https://www.hul.co.in/careers/',
            # 'hul': 'https://www.hul.co.in/careers/',
            # 'nestle india': 'https://www.nestle.in/careers',
            # 'nestle': 'https://www.nestle.in/careers',
            # 'britannia industries': 'https://www.britannia.co.in/careers',
            # 'britannia': 'https://www.britannia.co.in/careers',
            
            # # Pharma & Healthcare
            # 'sun pharmaceutical': 'https://www.sunpharma.com/careers',
            # 'sun pharma': 'https://www.sunpharma.com/careers',
            # 'dr reddys laboratories': 'https://www.drreddys.com/careers/',
            # 'dr reddys': 'https://www.drreddys.com/careers/',
            # 'cipla': 'https://www.cipla.com/careers',
            # 'lupin': 'https://www.lupin.com/careers/',
            # 'biocon': 'https://www.biocon.com/careers/',
            # 'apollo hospitals': 'https://www.apollohospitals.com/careers-2/',
            # 'fortis healthcare': 'https://www.fortishealthcare.com/india/careers',
            
            # # ==================== TELECOM & COMMUNICATION ====================
            # 'airtel': 'https://www.airtel.in/careers/',
            # 'bharti airtel': 'https://www.airtel.in/careers/',
            # 'vodafone idea': 'https://www.vodafoneidea.com/careers.html',
            # 'vi': 'https://www.vodafoneidea.com/careers.html',
            # 'bsnl': 'https://www.bsnl.co.in/Pages/Careers.aspx',
            
            # # ==================== RESEARCH & CONSULTING ====================
            # 'genpact': 'https://www.genpact.com/careers',
            # 'cognizant': 'https://careers.cognizant.com/global-en',
            # 'cognizant technology solutions': 'https://careers.cognizant.com/global-en',
            # 'capgemini': 'https://www.capgemini.com/careers/',
            # 'accenture': 'https://www.accenture.com/careers',
            # 'deloitte': 'https://www2.deloitte.com/in/en/careers.html',
            # 'deloitte india': 'https://www2.deloitte.com/in/en/careers.html',
            # 'ey': 'https://www.ey.com/en_in/careers',
            # 'ey india': 'https://www.ey.com/en_in/careers',
            # 'kpmg': 'https://home.kpmg/in/en/home/careers.html',
            # 'kpmg india': 'https://home.kpmg/in/en/home/careers.html',
            # 'pwc': 'https://www.pwc.in/careers.html',
            # 'pwc india': 'https://www.pwc.in/careers.html',
            
            # # ==================== STARTUPS & UNICORNS ====================
            # 'unikrn': 'https://www.unikrn.com/careers',
            # 'dunzo': 'https://www.dunzo.com/careers',
            # 'sharechat': 'https://sharechat.com/careers',
            # 'bharatpe': 'https://bharatpe.com/careers/',
            # 'cars24': 'https://www.cars24.com/careers/',
            # 'no broker': 'https://www.nobroker.com/careers',
            # 'lenskart': 'https://www.lenskart.com/careers',
            # 'boat lifestyle': 'https://www.boat-lifestyle.com/careers',
            # 'mamaearth': 'https://mamaearth.in/careers',
            # 'physics wallah': 'https://www.pw.live/careers',
            # 'eruditus': 'https://eruditus.com/careers/',
            
            # # ==================== GLOBAL COMPANIES WITH INDIAN PRESENCE ====================
            # 'google': 'https://careers.google.com',
            # 'microsoft': 'https://careers.microsoft.com',
            # 'amazon': 'https://www.amazon.jobs',
            # 'ibm': 'https://www.ibm.com/careers',
            # 'oracle': 'https://careers.oracle.com',
            # 'salesforce': 'https://www.salesforce.com/company/careers/',
            # 'sap': 'https://www.sap.com/india/about/careers.html',
            # 'intel': 'https://jobs.intel.com',
            # 'cisco': 'https://jobs.cisco.com',
            # 'accenture': 'https://www.accenture.com/careers',
            # 'capgemini': 'https://www.capgemini.com/careers/'

            # Tier 1 & Global Consulting

            'tcs': 'https://www.tcs.com/careers',

            'infosys': 'https://www.infosys.com/careers',

            'wipro': 'https://careers.wipro.com',

            'hcltech': 'https://www.hcltech.com/careers',

            'tech mahindra': 'https://careers.techmahindra.com',

            'cognizant': 'https://careers.cognizant.com/global-en',

            'accenture': 'https://www.accenture.com/in-en/careers',

            'capgemini': 'https://www.capgemini.com/in-en/careers/',

            'deloitte': 'https://www2.deloitte.com/in-en/careers.html',

            'pwc': 'https://www.pwc.in/careers.html',

            'ey': 'https://www.ey.com/en_in/careers',

            'kpmg': 'https://home.kpmg/in/en/home/careers.html',

            'genpact': 'https://www.genpact.com/careers',

            'dxc technology': 'https://dxc.com/in/en/careers',

            'mindtree': 'https://www.ltimindtree.com/careers',

            'ltimindtree': 'https://www.ltimindtree.com/careers',

            'mphasis': 'https://careers.mphasis.com/',

            'persistent systems': 'https://www.persistent.com/careers',

            'coforge': 'https://www.coforge.com/careers',

            'hexaware': 'https://www.hexaware.com/careers',

            'cyient': 'https://www.cyient.com/careers',

            'zensar': 'https://www.zensar.com/careers',

            'sonata software': 'https://www.sonata-software.com/careers',

            'birlasoft': 'https://www.birlasoft.com/careers',

            'tata elxsi': 'https://www.tataelxsi.com/careers',

            'kpit': 'https://www.kpit.com/careers',

            'virtusa': 'https://www.virtusa.com/careers',

            'ust global': 'https://ust.com/careers',

            'globant': 'https://careers.globant.com/',

            'nagaro': 'https://www.nagarro.com/en/careers',

            'thoughtworks': 'https://www.thoughtworks.com/careers',

            'epam': 'https://www.epam.com/careers',

            'happiest minds': 'https://www.happiestminds.com/careers/',



            # Tech Giants & R&D Centers

            'google': 'https://careers.google.com/locations/india/',

            'microsoft': 'https://careers.microsoft.com/us/en/search-results?q=India',

            'amazon': 'https://www.amazon.jobs/en/locations/india',

            'apple': 'https://jobs.apple.com/en-in/search?location=india-INDC',

            'meta': 'https://www.metacareers.com/locations/gurgaon/?p[location][0]=Gurgaon%2C%20India',

            'netflix': 'https://jobs.netflix.com/locations/mumbai-india',

            'adobe': 'https://www.adobe.com/careers.html',

            'oracle': 'https://careers.oracle.com/jobs/',

            'sap': 'https://www.sap.com/india/about/careers.html',

            'ibm': 'https://www.ibm.com/in-en/employment/',

            'cisco': 'https://jobs.cisco.com/jobs/SearchJobs/India',

            'intel': 'https://jobs.intel.com/en/location/india-jobs/599/1269750/2',

            'nvidia': 'https://www.nvidia.com/en-in/about-nvidia/careers/',

            'qualcomm': 'https://www.qualcomm.com/company/careers',

            'samsung': 'https://www.samsung.com/in/about-us/careers/',

            'atlassian': 'https://www.atlassian.com/company/careers/india',

            'salesforce': 'https://www.salesforce.com/company/careers/',

            'servicenow': 'https://www.servicenow.com/company/careers.html',

            'vmware': 'https://careers.vmware.com/main/',

            'intuit': 'https://www.intuit.com/careers/',

            'uber': 'https://www.uber.com/in/en/careers/',

            'paypal': 'https://careers.pypl.com/home/',

            'stripe': 'https://stripe.com/jobs/search?location=India',

            'walmart global tech': 'https://careers.walmart.com/results?q=&location=India',

            'target': 'https://india.target.com/careers',

            'tesco': 'https://www.tescocampus.com/careers',

            'goldman sachs': 'https://www.goldmansachs.com/careers/',

            'morgan stanley': 'https://www.morganstanley.com/about-us/careers',

            'jpmorgan chase': 'https://careers.jpmorgan.com/us/en/about-us/locations/india',

            'barclays': 'https://search.jobs.barclays/india',



            # Banking

            'sbi': 'https://sbi.bank.in/web/careers',

            'hdfc bank': 'https://www.hdfcbank.com/personal/careers',

            'icici bank': 'https://www.icicicareers.com/',

            'axis bank': 'https://www.axisbank.com/careers',

            'kotak mahindra': 'https://www.kotak.com/en/careers.html',

            'pnb': 'https://pnb.bank.in/recruitments.aspx',

            'bank of baroda': 'https://www.bankofbaroda.in/careers',

            'canara bank': 'https://canarabank.com/careers',

            'union bank': 'https://www.unionbankofindia.bank.in/en/common/recruitment',

            'idbi bank': 'https://www.idbibank.in/idbi-bank-careers.aspx',

            'yes bank': 'https://www.yesbank.in/careers',

            'idfc first bank': 'https://www.idfcfirstbank.com/careers',

            'indusind bank': 'https://www.indusind.com/in/en/careers.html',

            'standard chartered': 'https://www.sc.com/in/careers/',

            'hsbc': 'https://www.hsbc.com/careers',

            'citi': 'https://careers.citigroup.com/',

            'rbl bank': 'https://www.rblbank.com/careers',

            'federal bank': 'https://www.federalbank.co.in/careers',

            'bandhan bank': 'https://bandhanbank.com/recruitment',



            # NBFC & Insurance

            'bajaj finserv': 'https://www.bajajfinserv.in/careers',

            'muthoot finance': 'https://www.muthootfinance.com/careers',

            'manappuram finance': 'https://www.manappuram.com/careers.html',

            'lic': 'https://licindia.in/careers',

            'hdfc life': 'https://www.hdfclife.com/careers',

            'icici prudential': 'https://www.iciciprulife.com/careers/index.html',

            'sbi life': 'https://www.sbilife.co.in/en/about-us/careers',

            'bajaj allianz': 'https://www.bajajallianz.com/careers.html',

            'max life insurance': 'https://www.maxlifeinsurance.com/about-us/careers',

            'tata aia': 'https://tataaia.com/about-us/careers.html',

            'star health': 'https://www.starhealth.in/careers',

            'aditya birla capital': 'https://www.adityabirlacapital.com/careers',





            # Automobiles

            'tata motors': 'https://www.tatamotors.com/careers/',

            'mahindra': 'https://www.mahindra.com/careers',

            'maruti suzuki': 'https://www.marutisuzuki.com/careers',

            'hero motocorp': 'https://www.heromotocorp.com/en-in/careers/',

            'bajaj auto': 'https://www.bajajauto.com/careers',

            'tvs motor': 'https://www.tvsmotor.com/careers',

            'ashok leyland': 'https://www.ashokleyland.com/en/careers',

            'hyundai india': 'https://www.hyundai.com/in/en/hyundai-story/careers',

            'honda cars': 'https://www.hondacarindia.com/careers',

            'toyota kirloskar': 'https://www.toyotabharat.com/careers/',

            'royal enfield': 'https://www.royalenfield.com/in/en/our-world/careers/',

            'bosch india': 'https://www.bosch.in/careers/',

            'mercedes-benz india': 'https://www.mercedes-benz.co.in/passengercars/the-brand/careers.html',



            # Manufacturing & Heavy Engineering

            'larsen & toubro (l&t)': 'https://www.larsentoubro.com/corporate/careers/',

            'tata steel': 'https://www.tatasteel.com/careers/',

            'jsw steel': 'https://www.jsw.in/careers',

            'reliance industries': 'https://careers.ril.com/',

            'adani group': 'https://www.adanigroup.com/careers',

            'vedanta': 'https://www.vedantaresources.com/careers.aspx',

            'hindalco': 'https://www.hindalco.com/careers',

            'itc limited': 'https://www.itcportal.com/careers/',

            'asian paints': 'https://www.asianpaints.com/careers',

            'pidilite': 'https://www.pidilite.com/careers',

            'berger paints': 'https://www.bergerpaints.com/careers/',

            'ultratech cement': 'https://www.ultratechcement.com/about-us/careers',

            'siemens india': 'https://www.siemens.com/in/en/company/jobs.html',

            'abb india': 'https://new.abb.com/careers',

            'havells': 'https://www.havells.com/en/career.html',



            # FMCG & Beverages

            'hul': 'https://www.hul.co.in/careers/',

            'nestle': 'https://www.nestle.in/jobs',

            'pepsico': 'https://www.pepsicojobs.com/main/',

            'coca-cola': 'https://www.coca-colaindia.com/careers',

            'britannia': 'https://www.britannia.co.in/careers',

            'dabur': 'https://www.dabur.com/careers',

            'marico': 'https://www.marico.com/india/careers',

            'godrej consumer': 'https://www.godrejcp.com/careers',

            'amul': 'https://www.amul.com/m/careers',



            # Retail & Fashion

            'reliance retail': 'https://careers.ril.com/relianceretail/',

            'trent (westside)': 'https://trentlimited.com/pages/careers',

            'titan': 'https://www.titancompany.in/careers',

            'dmart': 'https://www.dmartindia.com/careers',

            'aditya birla fashion (abfrl)': 'https://www.abfrl.com/careers',

            'shoppers stop': 'https://www.shoppersstop.com/careers',

            'raymond': 'https://www.raymond.in/careers',



            # Healthcare & Pharma

            'apollo hospitals': 'https://www.apollohospitals.com/careers/',

            'max healthcare': 'https://www.maxhealthcare.in/careers',

            'medanta': 'https://www.medanta.org/careers',

            'fortis': 'https://www.fortishealthcare.com/careers',

            'sun pharma': 'https://www.sunpharma.com/careers',

            'dr reddys': 'https://www.drreddys.com/careers/',

            'cipla': 'https://www.cipla.com/careers',

            'lupin': 'https://www.lupin.com/careers/',

            'biocon': 'https://www.biocon.com/careers/',

            'gsk india': 'https://india-pharma.gsk.com/en-in/careers/',

            'pfizer india': 'https://www.pfizerindia.com/careers',



            'flipkart': 'https://www.flipkartcareers.com/',

            'meesho': 'https://careers.meesho.com/',

            'myntra': 'https://careers.myntra.com/',

            'zomato': 'https://www.zomato.com/careers',

            'swiggy': 'https://careers.swiggy.com/',

            'nykaa': 'https://www.nykaa.com/careers',

            'bigbasket': 'https://careers.bigbasket.com/',

            'blinkit': 'https://blinkit.com/careers',

            'zepto': 'https://www.zepto.com/careers',

            'paytm': 'https://paytm.com/careers/',

            'phonepe': 'https://www.phonepe.com/careers/',

            'razorpay': 'https://razorpay.com/careers/',

            'cred': 'https://careers.cred.club/',

            'ola': 'https://www.olacabs.com/careers',

            'urban company': 'https://careers.urbancompany.com/',

            'lenskart': 'https://www.lenskart.com/careers',

            'oyo': 'https://www.oyorooms.com/careers/',

            'makemytrip': 'https://careers.makemytrip.com/',

            'ixigo': 'https://www.ixigo.com/careers',

            'unacademy': 'https://unacademy.com/careers/',

            'upgrad': 'https://www.upgrad.com/careers/',

            'physicswallah': 'https://www.pw.live/careers',

            'delhivery': 'https://www.delhivery.com/careers/',

            'sharechat': 'https://sharechat.com/careers',

            'boat': 'https://www.boat-lifestyle.com/pages/careers',



            'jio': 'https://careers.ril.com/jio/',

            'airtel': 'https://www.airtel.in/careers/',

            'vodafone idea (vi)': 'https://www.vodafoneidea.com/careers.html',

            'tata communications': 'https://www.tatacommunications.com/careers/',

            'bsnl': 'https://www.bsnl.co.in/Pages/Careers.aspx',

            'tata power': 'https://www.tatapower.com/careers.aspx',

            'adani group': 'https://www.adanigroup.com/careers',

            'ntpc': 'https://careers.ntpc.co.in/',

            'ongc': 'https://www.ongcindia.com/wps/wcm/connect/en/career/',

            'iocl': 'https://iocl.com/pages/careers-overview',

            'bpcl': 'https://www.bharatpetroleum.in/careers/careers.aspx',

            'hpcl': 'https://www.hindustanpetroleum.com/careers',

            'gail': 'https://gailonline.com/CR-CurrentOpening.html',

            'power grid': 'https://www.powergrid.in/job-opportunities',

            'nhpc': 'https://www.nhpcindia.com/Home/Index?hc=887',



            'indigo': 'https://www.goindigo.in/careers.html',

            'air india': 'https://www.airindia.com/in/en/careers.html',

            'vistara': 'https://www.airvistara.com/in/en/careers',

            'spicejet': 'https://www.spicejet.com/careers',

            'akasa air': 'https://www.akasaair.com/careers',

            'blue dart': 'https://www.bluedart.com/careers',

            'shipping corp of india': 'https://www.shipindia.com/careers',

            'concor': 'https://concorindia.co.in/career.asp',



            'disney+ hotstar': 'https://careers.hotstar.com/',

            'zee entertainment': 'https://www.zee.com/careers',

            'sony pictures networks': 'https://www.sonypicturesnetworks.com/careers',

            'viacom18': 'https://www.viacom18.com/careers',

            'pvr inox': 'https://www.pvrcinemas.com/careers',

            'times group': 'https://timesgroup.com/careers/',

            'ndtv': 'https://www.ndtv.com/jobs',

            'dream11': 'https://www.dreamsports.group/careers',

            'nazara': 'https://www.nazara.com/careers',

            'games24x7': 'https://www.games24x7.com/careers/',

            'quess corp': 'https://www.quesscorp.com/careers/',
            'teamlease': 'https://www.teamlease.com/careers',



            # ==================== SaaS / PRODUCT TECH ====================

            'zoho': 'https://www.zoho.com/careers/',
            'zoho corporation': 'https://www.zoho.com/careers/',

            'freshworks': 'https://www.freshworks.com/company/careers/',

            'inmobi': 'https://www.inmobi.com/company/careers/',

            'browserstack': 'https://www.browserstack.com/careers',

            'postman': 'https://www.postman.com/company/careers/',

            'chargebee': 'https://www.chargebee.com/careers/',

            'clevertap': 'https://clevertap.com/company/careers/',

            'whatfix': 'https://whatfix.com/careers/',

            'druva': 'https://www.druva.com/about/careers/',

            'icertis': 'https://www.icertis.com/company/careers/',



            # ==================== FINTECH ====================

            'zerodha': 'https://zerodha.com/careers/',

            'groww': 'https://groww.in/careers',

            'policybazaar': 'https://www.policybazaar.com/careers/',

            'bharatpe': 'https://bharatpe.com/careers/',

            'pine labs': 'https://www.pinelabs.com/careers',

            'navi': 'https://navi.com/careers',

            'slice': 'https://www.sliceit.com/careers',

            'open financial technologies': 'https://open.money/careers',

            'jupiter': 'https://jupiter.money/careers/',



            # ==================== AI / ANALYTICS ====================

            'fractal analytics': 'https://fractal.ai/careers/',

            'tredence': 'https://www.tredence.com/careers/',

            'themathcompany': 'https://themathcompany.com/careers/',

            'mu sigma': 'https://www.mu-sigma.com/careers',

            'latentview analytics': 'https://www.latentview.com/careers/',

            'tiger analytics': 'https://www.tigeranalytics.com/careers/',

            'gramener': 'https://gramener.com/careers/',

            'course5 intelligence': 'https://www.course5i.com/careers/',



            # ==================== EV / NEW-AGE AUTOMOTIVE ====================

            'ather energy': 'https://www.atherenergy.com/careers',

            'ola electric': 'https://careers.olaelectric.com/',

            'tata technologies': 'https://www.tatatechnologies.com/in/careers/',

            'altigreen': 'https://altigreen.com/careers/',

            'euler motors': 'https://www.eulermotors.com/careers',



            # ==================== STARTUPS / CONSUMER INTERNET ====================

            'dunzo': 'https://www.dunzo.com/careers',

            'porter': 'https://porter.in/careers',

            'udaan': 'https://udaan.com/careers.html',

            'infra.market': 'https://www.infra.market/careers',

            'nobroker': 'https://www.nobroker.in/careers',

            'no broker': 'https://www.nobroker.in/careers',

            'cars24': 'https://www.cars24.com/careers/',

            'spinny': 'https://www.spinny.com/careers/',



            # ==================== GLOBAL CONSULTING ====================

            'mckinsey': 'https://www.mckinsey.com/careers',

            'mckinsey & company': 'https://www.mckinsey.com/careers',

            'bcg': 'https://careers.bcg.com/',

            'boston consulting group': 'https://careers.bcg.com/',

            'bain & company': 'https://www.bain.com/careers/',

            'bain': 'https://www.bain.com/careers/',

            'alvarez & marsal': 'https://www.alvarezandmarsal.com/careers',

            'alvarez and marsal': 'https://www.alvarezandmarsal.com/careers',



            # ==================== GLOBAL INVESTMENT / FINANCE ====================

            'blackrock': 'https://careers.blackrock.com/',

            'nomura': 'https://www.nomura.com/careers/',

            'deutsche bank': 'https://careers.db.com/',

            'ubs': 'https://www.ubs.com/global/en/careers.html',

            'wells fargo': 'https://www.wellsfargojobs.com/',



            # ==================== SEMICONDUCTOR / HARDWARE ====================

            'amd': 'https://careers.amd.com/',

            'micron technology': 'https://micron.wd1.myworkdayjobs.com/External',

            'micron': 'https://micron.wd1.myworkdayjobs.com/External',

            'texas instruments': 'https://careers.ti.com/',

            'broadcom': 'https://www.broadcom.com/company/careers',

            'applied materials': 'https://www.appliedmaterials.com/us/en/careers',

            'lam research': 'https://careers.lamresearch.com/',



            # ==================== GAMING / TECH PLATFORMS ====================

            'ubisoft india': 'https://www.ubisoft.com/en-gb/company/careers',

            'ubisoft': 'https://www.ubisoft.com/en-gb/company/careers',

            'rockstar games': 'https://www.rockstargames.com/careers',

            'mpl': 'https://www.mpl.live/careers',

            'mobile premier league': 'https://www.mpl.live/careers',



            # ==================== CLOUD / INFRA / DEV TOOLS ====================

            'digitalocean': 'https://www.digitalocean.com/careers/',

            'snowflake': 'https://careers.snowflake.com/',

            'databricks': 'https://www.databricks.com/company/careers',

            'cloudflare': 'https://www.cloudflare.com/careers/',

            'hashicorp': 'https://www.hashicorp.com/careers',


        }
    
    def _save_company_database(self):
        # Save updated company database
        try:
            with open('company_database.json', 'w') as f:
                json.dump(self.known_companies, f, indent=2)
        except:
            pass
    
    def verify_company(self, company_name, enable_discovery=True):
        # Verify company with optional auto-discovery
        if not company_name or len(company_name.strip()) < 2:
            return self._error_result("No company name provided")
        
        company_name_clean = company_name.strip().title()
        company_lower = company_name.lower()
        
        # Step 1: Check existing database
        existing_result = self._check_existing_companies(company_lower, company_name_clean)
        if existing_result["is_genuine"]:
            return existing_result
        
        # Step 2: Auto-discovery if enabled
        if enable_discovery and self.learning_enabled:
            discovery_result = self._attempt_auto_discovery(company_name_clean, company_lower)
            if discovery_result:
                return discovery_result
        
        return self._not_found_result(company_name_clean)
    
    def _check_existing_companies(self, company_lower, company_name_clean):
        # Check against known companies database
        if company_lower in self.known_companies:
            return {
                "is_genuine": True,
                "careers_url": self.known_companies[company_lower],
                "company_name": company_name_clean,
                "verification_method": "Verified company database",
                "details": "Found in verified companies database",
                "source": "existing_database"
            }
        
        # Partial matching
        for known_company, careers_url in self.known_companies.items():
            if known_company in company_lower:
                return {
                    "is_genuine": True,
                    "careers_url": careers_url,
                    "company_name": company_name_clean,
                    "verification_method": "Partial match in company database",
                    "details": f"Matched with known company: {known_company.title()}",
                    "source": "partial_match"
                }
        
        return {"is_genuine": False}
    
    def _attempt_auto_discovery(self, company_name_clean, company_lower):
        # Attempt to auto-discover new company
        with st.spinner(f"                                             **AI Discovery**: Learning about '{company_name_clean}'..."):
            discovery_result = self.discovery_engine.discover_company(company_name_clean)
        
        if discovery_result['found'] and discovery_result['careers_url']:
            # Add to database
            self.known_companies[company_lower] = discovery_result['careers_url']
            self._save_company_database()
            
            return {
                "is_genuine": True,
                "careers_url": discovery_result['careers_url'],
                "company_name": company_name_clean,
                "verification_method": "AI Auto-Discovery",
                "details": f"Discovered and validated via search engines (Confidence: {discovery_result['confidence']:.1f}%)",
                "source": "auto_discovery",
                "discovery_details": discovery_result
            }
        
        return None
    
    def _error_result(self, message):
        return {
            "is_genuine": False,
            "careers_url": None,
            "company_name": "Unknown",
            "verification_method": "Error",
            "details": message,
            "source": "error"
        }
    
    def _not_found_result(self, company_name_clean):
        return {
            "is_genuine": False,
            "careers_url": None,
            "company_name": company_name_clean,
            "verification_method": "All verification methods failed",
            "details": "Company not found in verified databases and auto-discovery failed",
            "source": "not_found"
        }
    
    def get_database_stats(self):
        # Get statistics about the company database
        return {
            "total_companies": len(self.known_companies),
            "auto_learned_companies": len([k for k, v in self.known_companies.items() 
                                         if not k in ['tcs', 'infosys', 'wipro', 'hcl', 
                                                     'google', 'microsoft', 'amazon']]),
            "learning_enabled": self.learning_enabled
        }

# ==================== MAIN APPLICATION ====================

def main():
    st.markdown('<h1 class="main-header">                                    JobVerification AI <span class="ai-badge">REAL JOBS ONLY</span></h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Verify Job Postings                                        Fraud Detection + Real Career Page Matching</p>', unsafe_allow_html=True)

    # Initialize components
    company_verifier = SelfLearningCompanyVerifier()
    job_scraper = RealJobScraper()

    # Database Statistics
    stats = company_verifier.get_database_stats()
    st.markdown('<div class="auto-learn-card">', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Companies", stats["total_companies"])
    with col2:
        st.metric("Auto-Learned", stats["auto_learned_companies"])
    with col3:
        st.metric("Real Job Scraping", "ACTIVE")
    with col4:
        st.metric("Fraud Detection", "ON")
    st.caption("                                             Step 1: Fraud keyword scan                                           Step 2: Company lookup                                           Step 3: Live career page match")
    st.markdown('</div>', unsafe_allow_html=True)

    #                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
    # STEP 1: INPUT FORM
    #                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
    st.markdown('<div class="card">', unsafe_allow_html=True)
    with st.expander("Network / Proxy Settings (optional)"):
        st.checkbox("Use proxy", key="use_proxy")
        st.text_input("HTTP proxy (e.g., http://user:pass@host:port)", key="proxy_http")
        st.text_input("HTTPS proxy (e.g., http://user:pass@host:port)", key="proxy_https")
    st.markdown("###                                            Enter Job Posting Details for Verification")
    col1, col2 = st.columns(2)
    with col1:
        company_name = st.text_input("**Company Name** *(required)*",
                                     placeholder="e.g., TCS, Infosys, Wipro, Google")
        job_title = st.text_input("**Job Role / Title** *(required)*",
                                  placeholder="e.g., Software Engineer, Data Analyst")
    with col2:
        job_description = st.text_area("**Job Description** *(required)*",
                                       placeholder="Paste the full job description here...",
                                       height=130)
        job_url = st.text_input("**Job Posting URL** *(optional)*",
                                placeholder="Paste the official job posting link (if available)")
        enable_ml = st.checkbox("**Enable ML Model Scoring** *(slower on first run)*", value=False,
                                help="Loads the ML model to score text risk. First run may be slow.")
        enable_discovery = st.checkbox("**Enable AI Auto-Discovery**", value=True,
                                       help="Automatically find companies not yet in the database via web scraping")
    verify_clicked = st.button("**                                          Analyze & Verify Job Posting**",
                               use_container_width=True, type="primary")
    st.markdown('</div>', unsafe_allow_html=True)

    if verify_clicked:
        status = st.empty()
        status.info("Starting verification...")
        # Validate all fields
        if not company_name.strip():
            st.warning("                                                     Please enter the **Company Name**.")
            return
        if not job_title.strip():
            st.warning("                                                     Please enter the **Job Role / Title**.")
            return
        if not job_description.strip():
            st.warning("                                                     Please paste the **Job Description**.")
            return


        # STEP 1: ML fraud score (text-only in UI)
        if enable_ml:
            with st.spinner("Loading ML model and scoring text..."):
                model, tokenizer = load_fraud_model_assets()
                ml_score = None
                if model is None or tokenizer is None:
                    st.info("ML model not loaded. Run training to enable model-based scoring.")
                else:
                    try:
                        ml_score = predict_fraud_score(model, tokenizer, job_title.strip(), job_description.strip())
                        ml_pct = round(ml_score * 100, 1)
                        risk_label = "LOW" if ml_score < 0.3 else "MEDIUM" if ml_score < 0.6 else "HIGH"
                        st.markdown(
                            f"**ML Risk Score:** {ml_pct}% (Text-only, metadata assumed neutral) | **Risk:** {risk_label}"
                        )
                    except Exception as e:
                        st.warning(f"ML scoring failed: {e}")
        else:
            st.caption("ML scoring is disabled for faster response.")

        #                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
        # STEP 2: FRAUD KEYWORD CHECK                                        runs FIRST, before any web call
        #                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             # STEP 2: FRAUD SIGNAL SCORING   runs FIRST, before any web call
        scorer = FraudSignalScorer()
        fraud_score, fraud_breakdown = scorer.score(company_name.strip(), job_title.strip(), job_description.strip(), [])

        st.markdown("### Fraud Risk Score")
        st.metric("Fraud Risk Score", str(fraud_score) + "/100")
        with st.expander("Fraud Signal Breakdown"):
            for label, pts, hints in fraud_breakdown:
                hint_text = " | " + ", ".join(hints) if hints else ""
                st.write(label + ": " + str(round(pts, 1)) + " pts" + hint_text)

        if fraud_score >= 70:
            st.markdown('<div class="fake-card">', unsafe_allow_html=True)
            st.markdown(
                "<div style='font-size:3rem;margin-bottom:0.5rem;'>&#128680;</div>"
                "<div style='font-size:2rem;font-weight:800;color:#991b1b;'>HIGH RISK JOB POSTING</div>"
                "<div style='font-size:1.1rem;color:#7f1d1d;margin-top:0.5rem;'>"
                "This job description shows multiple scam indicators."
                "<strong>Do NOT apply or share personal/financial details.</strong>"
                "</div>",
                unsafe_allow_html=True
            )
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown(
                "**What You Should Do:**\n"
                "- Do NOT pay any registration, processing, or joining fee\n"
                "- Do NOT share Aadhaar, PAN, bank details, or OTPs with the recruiter\n"
                "- Do NOT contact via WhatsApp or Telegram for job offers\n"
                "- Always verify openings on the company's official website directly"
            )
            return  # Stop   no further processing needed

        st.info("Step 2 PASSED: No high-risk fraud indicators detected. Now verifying the company...")

        #                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
        # STEP 3 / STEP 5: COMPANY LOOKUP + AUTO-DISCOVERY IF NOT IN DB
        #                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
        with st.spinner("Looking up company in verified database..."):
            company_result = company_verifier.verify_company(company_name.strip(), enable_discovery)

        if not company_result["is_genuine"] or not company_result.get("careers_url"):
            #                                                                              RESULT: COMPANY NOT VERIFIABLE                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
            st.markdown('<div class="discovery-failure">', unsafe_allow_html=True)
            st.markdown(
                "<div style='font-size:2rem;font-weight:800;color:#991b1b;'>Company Cannot Be Verified</div>"
                "<div style='font-size:1.1rem;color:#7f1d1d;margin-top:0.5rem;'>"
                "<strong>" + company_result['company_name'] + "</strong> was not found in our database "
                "or through online discovery. This may be a non-existent or fraudulent company."
                "</div>",
                unsafe_allow_html=True
            )
            st.markdown('</div>', unsafe_allow_html=True)
            st.warning("Details: " + company_result['details'])
            st.markdown(
                "**Safety Warning:**\n"
                "- This company is not in our verified database\n"
                "- No official career page was found even with AI auto-discovery\n"
                "- Do NOT apply or share personal information\n"
                "- Verify the company independently before proceeding"
            )
            return

        # Company found
        careers_url = company_result["careers_url"]
        source_label = company_result.get("verification_method", "Verified database")

        if company_result.get("source") == "auto_discovery":
            st.success(
                "AI Auto-Discovered: '" + company_result['company_name'] + "' was newly found online "
                "via web scraping and added to the database."
            )
        else:
            st.success("Company Verified: " + company_result['company_name'] + "                                        " + source_label)

        st.info("Official Career Page: [" + careers_url + "](" + careers_url + ")")

        # Optional: verify using direct job URL (best for JS-heavy portals)
        allowed_domains = []
        try:
            allowed_domains.append(urlparse(careers_url).netloc)
        except Exception:
            pass
        if company_result.get("website"):
            try:
                allowed_domains.append(urlparse(company_result["website"]).netloc)
            except Exception:
                pass

        # Update fraud score with company domain context (email/domain checks)
        fraud_score, fraud_breakdown = scorer.score(
            company_name.strip(),
            job_title.strip(),
            job_description.strip(),
            allowed_domains
        )
        st.metric("Fraud Risk Score (with company domain)", str(fraud_score) + "/100")

        # If user didn't paste a URL, try to auto-detect from description
        if not (job_url and job_url.strip()):
            auto_url = _extract_first_url(job_description)
            if auto_url:
                job_url = auto_url
                st.info("Detected a job URL from the description. Using it for verification.")

        # Accenture fallback: build job URL from job ID in description/title
        if not (job_url and job_url.strip()) and company_result.get("company_name", "").lower() == "accenture":
            acc_id = _extract_accenture_job_id(job_description + " " + job_title)
            if acc_id:
                job_url = "https://www.accenture.com/in-en/careers/jobdetails?id=" + acc_id
                st.info("Detected Accenture job ID. Constructed job URL for verification.")

        if job_url and job_url.strip():
            with st.spinner("Verifying the specific job URL..."):
                url_check = verify_job_url(job_url, job_title.strip(), allowed_domains)

            if url_check["status"] == "verified":
                st.markdown('<div class="genuine-card">', unsafe_allow_html=True)
                st.markdown(
                    "<div style='font-size:3rem;margin-bottom:0.5rem;'>&#10003;</div>"
                    "<div style='font-size:2rem;font-weight:800;color:#065f46;'>GENUINE JOB OPENING</div>"
                    "<div style='font-size:1.1rem;color:#064e3b;margin-top:0.5rem;'>"
                    "The job URL matches the official domain and the role title was found on the page.</div>",
                    unsafe_allow_html=True
                )
                st.markdown("**Job Posting URL**: [" + job_url.strip() + "](" + job_url.strip() + ")")
                st.markdown("**Official Career Page**: [" + careers_url + "](" + careers_url + ")")
                st.markdown('</div>', unsafe_allow_html=True)
                return
            elif url_check["status"] == "blocked":
                st.warning("Job URL could not be verified: " + url_check["reason"])
                # Continue with scraping as fallback
            else:
                st.info("Job URL check inconclusive: " + url_check["reason"])
                # Continue with scraping as fallback

        # Verification checklist (URL, ATS, email)
        st.markdown("### Verification Checklist")
        checklist = build_verification_checklist(
            company_result,
            job_url.strip() if job_url else "",
            job_title.strip(),
            job_description
        )
        for label, result in checklist:
            status = result.get("status", "inconclusive")
            reason = result.get("reason", "")
            if status == "verified":
                st.success(label + ": " + reason)
            elif status == "blocked":
                st.warning(label + ": " + reason)
            else:
                st.info(label + ": " + reason)

        #                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
        # STEP 3: SCRAPE CAREER PAGE                                        SEARCH FOR SPECIFIC JOB ROLE
        #                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
        st.markdown("---")
        st.markdown("### Searching for **'" + job_title.strip() + "'** on " + company_result['company_name'] + "'s Career Page...")

        with st.spinner("Scraping career page for '" + job_title.strip() + "'..."):
            matching_jobs = job_scraper.scrape_real_jobs(
                company_result['company_name'],
                careers_url,
                job_title.strip()
            )

        scrape_ok = (job_scraper.last_fetch_ok and not job_scraper.last_dynamic_site and job_scraper.last_error is None)
        verdict = compute_overall_verdict(
            matching_jobs,
            scrape_ok,
            checklist,
            fraud_score=fraud_score,
            company_verified=True,
            dynamic_site=job_scraper.last_dynamic_site
        )

        if verdict == "VERIFIED":
            st.markdown('<div class="genuine-card">', unsafe_allow_html=True)
            st.markdown(
                "<div style='font-size:3rem;margin-bottom:0.5rem;'>&#10003;</div>"
                "<div style='font-size:2rem;font-weight:800;color:#065f46;'>VERIFIED JOB OPENING</div>"
                "<div style='font-size:1.1rem;color:#064e3b;margin-top:0.5rem;'>"
                "<strong>" + company_result['company_name'] + "</strong> has a verified signal for "
                "<strong>'" + job_title.strip() + "'</strong>.</div>",
                unsafe_allow_html=True
            )
            st.markdown("**Apply Directly**: [" + careers_url + "](" + careers_url + ")")
            st.markdown('</div>', unsafe_allow_html=True)

            if matching_jobs:
                st.markdown("#### " + str(len(matching_jobs)) + " Matching Opening(s) Found on Official Career Page")
                for job in matching_jobs:
                    st.markdown(
                        '<div class="job-listing">'
                        '<div class="job-title">' + job["title"] + '</div>'
                        '<div class="job-meta">Location: ' + job["location"] + ' | Type: ' + job["type"] + ' | Posted: ' + job.get("posted", "Current") + '</div>'
                        '<a href="' + job['url'] + '" target="_blank" class="job-link">View Opening on Official Career Page</a>'
                        '</div>',
                        unsafe_allow_html=True
                    )

        elif verdict == "GENUINE_LIKELY":
            st.markdown('<div class="discovery-failure">', unsafe_allow_html=True)
            st.markdown(
                "<div style='font-size:2rem;font-weight:800;color:#92400e;'>GENUINE LIKELY</div>"
                "<div style='font-size:1.1rem;color:#7c2d12;margin-top:0.5rem;'>"
                "The company is verified, but the career portal is JS-heavy and could not be scraped. "
                "This role is likely genuine. Please verify directly on the official career page.</div>",
                unsafe_allow_html=True
            )
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown("**Verify on official career page**: [" + careers_url + "](" + careers_url + ")")

        elif verdict == "HIGH_RISK":
            st.markdown('<div class="fake-card">', unsafe_allow_html=True)
            st.markdown(
                "<div style='font-size:3rem;margin-bottom:0.5rem;'>&#10060;</div>"
                "<div style='font-size:2rem;font-weight:800;color:#991b1b;'>HIGH RISK / UNVERIFIED</div>"
                "<div style='font-size:1.1rem;color:#7f1d1d;margin-top:0.5rem;'>"
                "One or more strong warning signals were detected (e.g., non-official domain or email).</div>",
                unsafe_allow_html=True
            )
            st.markdown("**Verify on official career page**: [" + careers_url + "](" + careers_url + ")")
            st.markdown('</div>', unsafe_allow_html=True)

        elif verdict == "INCONCLUSIVE":
            st.markdown('<div class="discovery-failure">', unsafe_allow_html=True)
            st.markdown(
                "<div style='font-size:2rem;font-weight:800;color:#991b1b;'>INCONCLUSIVE</div>"
                "<div style='font-size:1.1rem;color:#7f1d1d;margin-top:0.5rem;'>"
                "We could not verify this role automatically. This does <strong>not</strong> mean it is fake."
                "</div>",
                unsafe_allow_html=True
            )
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown("**Verify on official career page**: [" + careers_url + "](" + careers_url + ")")

            if job_scraper.last_dynamic_site:
                st.warning("This career portal appears to be JavaScript-heavy, so server-side scraping may miss listings.")
            if job_scraper.last_error:
                st.warning("Scrape error: " + job_scraper.last_error)
            if job_scraper.last_status_code:
                st.caption("HTTP status: " + str(job_scraper.last_status_code))

        else:
            st.markdown('<div class="fake-card">', unsafe_allow_html=True)
            st.markdown(
                "<div style='font-size:3rem;margin-bottom:0.5rem;'>&#10060;</div>"
                "<div style='font-size:2rem;font-weight:800;color:#991b1b;'>UNVERIFIED ROLE</div>"
                "<div style='font-size:1.1rem;color:#7f1d1d;margin-top:0.5rem;'>"
                "We could not find this role on the official career page. This does not confirm it is fake."
                "</div>",
                unsafe_allow_html=True
            )
            st.markdown("**Verify on official career page**: [" + careers_url + "](" + careers_url + ")")
            st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()









