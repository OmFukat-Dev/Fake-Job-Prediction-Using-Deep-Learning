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
from urllib.parse import quote, urlparse, urljoin
import urllib3
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# DEEP LEARNING IMPORTS
import tensorflow as tf 
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import (Dense, LSTM, GRU, Bidirectional, Embedding, 
                                   Dropout, Conv1D, GlobalMaxPooling1D, GlobalAveragePooling1D,
                                   MultiHeadAttention, LayerNormalization, Input, Concatenate)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils.class_weight import compute_class_weight
import warnings
warnings.filterwarnings('ignore')

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
    model = load_model(MODEL_PATH)
    tokenizer = joblib.load(TOKENIZER_PATH)
    return model, tokenizer


def predict_fraud_score(model, tokenizer, title, description):
    combined_text = f"{title} {description}"
    seq = tokenizer.texts_to_sequences([combined_text])
    padded = pad_sequences(seq, maxlen=MAX_TEXT_LEN, padding='post', truncating='post')
    # metadata placeholders (unknown in UI)
    numeric_features = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    prob = float(model.predict([padded, numeric_features], verbose=0)[0][0])
    return prob

# Set page config
st.set_page_config(
    page_title="JobVerification AI - Real Job Openings Finder",
    page_icon="🧠",
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
    """
    Multi-strategy company discovery engine.

    Strategy 1 (fastest): Direct URL guessing from company name
        - Constructs likely URLs like zoho.com, zoho.in, zoho.io etc.
        - Hits them directly — no search engine dependency.

    Strategy 2 (fallback): DuckDuckGo HTML search
        - Only used if direct guessing fails.
        - Parses DuckDuckGo result links (more scrape-friendly than Google/Bing).

    For both strategies, once a website is found:
        - Searches the homepage for a careers/jobs link.
        - Falls back to trying common paths like /careers, /jobs.
        - Validates that the careers page actually has job-related content.
    """

    # Excluded domains that should never be treated as a company website
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

    def _create_session(self):
        """Create a requests session with browser-like headers"""
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

    # ──────────────────────────────────────────────────────────────────────────
    # PUBLIC API
    # ──────────────────────────────────────────────────────────────────────────

    def discover_company(self, company_name):
        """
        Main method: try to find the company website + career page.
        Returns a result dict with keys: found, careers_url, website, confidence, ...
        """
        result = {
            'company_name': company_name,
            'found': False,
            'careers_url': None,
            'website': None,
            'confidence': 0,
            'discovery_method': None,
            'validation_details': [],
        }

        # ── Strategy 1: Direct URL guessing ──────────────────────────────────
        website_url = self._guess_website_directly(company_name)
        if website_url:
            result['discovery_method'] = 'direct_url_guess'
            result['validation_details'].append(f"Strategy 1 succeeded: {website_url}")
        else:
            # ── Strategy 2: DuckDuckGo HTML search ───────────────────────────
            result['validation_details'].append("Strategy 1 (direct URL) failed — trying DuckDuckGo...")
            website_url = self._search_duckduckgo(company_name)
            if website_url:
                result['discovery_method'] = 'duckduckgo_search'
                result['validation_details'].append(f"Strategy 2 (DuckDuckGo) found: {website_url}")
            else:
                result['validation_details'].append("Strategy 2 (DuckDuckGo) also failed.")
                return result

        result['website'] = website_url

        # ── Find careers page from the website ───────────────────────────────
        careers_url = self._find_careers_page(website_url)
        if not careers_url:
            result['validation_details'].append("Website found but no careers page detected.")
            return result

        result['careers_url'] = careers_url
        result['validation_details'].append(f"Careers page found: {careers_url}")

        # ── Validate: is this genuinely a company careers page? ───────────────
        score = self._score_careers_page(careers_url)
        result['confidence'] = round(score * 100, 1)
        result['validation_details'].append(f"Validation score: {score:.0%}")

        if score >= 0.5:   # Threshold lowered from 0.7 to 0.5
            result['found'] = True
        else:
            result['validation_details'].append("Score below 50% threshold — not confident enough.")

        return result

    # ──────────────────────────────────────────────────────────────────────────
    # STRATEGY 1 — DIRECT URL GUESSING
    # ──────────────────────────────────────────────────────────────────────────

    def _guess_website_directly(self, company_name):
        """
        Construct candidate URLs from the company name and probe them directly.
        E.g. "Zoho Corporation"  → zoho.com, zoho.in, zoho.io, zohocorporation.com ...
        """
        # Build slug variants
        words = company_name.lower().split()
        # Remove generic suffixes so "tata motors" → "tatamotors" AND "tata"
        stop_words = {'limited', 'ltd', 'pvt', 'private', 'inc', 'corp',
                      'corporation', 'technologies', 'technology', 'solutions',
                      'systems', 'services', 'group', 'india', 'global',
                      'international', 'enterprises', 'consulting'}
        core_words = [w for w in words if w not in stop_words]

        name_variants = list(dict.fromkeys([
            ''.join(core_words),            # zohocorporation → zoho
            core_words[0] if core_words else '',   # first word only
            ''.join(words),                 # full slug
            '-'.join(core_words),           # zoho-corporation
        ]))
        name_variants = [v for v in name_variants if v]

        tlds = ['.com', '.in', '.io', '.co', '.net', '.org', '.co.in']

        candidates = []
        for variant in name_variants:
            for tld in tlds:
                candidates.append(f'https://www.{variant}{tld}')
                candidates.append(f'https://{variant}{tld}')

        for url in candidates:
            try:
                resp = self.session.get(url, timeout=8, verify=False,
                                        allow_redirects=True)
                if resp.status_code == 200 and len(resp.content) > 500:
                    final_url = resp.url  # follow redirects
                    if self._is_plausible_company_site(final_url, company_name, resp.text):
                        return final_url
            except Exception:
                continue

        return None

    def _is_plausible_company_site(self, url, company_name, html_text=''):
        """
        Heuristic: does this URL look like the company's own website?
        Returns True if at least one core word appears in the domain.
        """
        from urllib.parse import urlparse
        try:
            domain = urlparse(url).netloc.lower().replace('www.', '')
        except Exception:
            return False

        # Must not be an excluded domain
        for excl in self.EXCLUDED_DOMAINS:
            if excl in domain:
                return False

        # At least one meaningful company word must appear in the domain
        words = company_name.lower().split()
        stop_words = {'limited', 'ltd', 'pvt', 'private', 'inc', 'corp',
                      'corporation', 'technologies', 'technology', 'solutions',
                      'systems', 'services', 'group', 'india', 'global',
                      'international', 'enterprises', 'consulting', 'the'}
        core_words = [w for w in words if w not in stop_words and len(w) >= 3]

        return any(w in domain for w in core_words)

    # ──────────────────────────────────────────────────────────────────────────
    # STRATEGY 2 — DUCKDUCKGO HTML SEARCH
    # ──────────────────────────────────────────────────────────────────────────

    def _search_duckduckgo(self, company_name):
        """
        Search DuckDuckGo (HTML version) for the company's official website.
        DuckDuckGo is more scrape-tolerant than Google/Bing.
        """
        from urllib.parse import quote, urlparse
        queries = [
            f'{company_name} official website',
            f'{company_name} careers site:*.com OR site:*.in OR site:*.io',
        ]

        for query in queries:
            try:
                url = f'https://duckduckgo.com/html/?q={quote(query)}'
                resp = self.session.get(url, timeout=12, verify=False)
                if resp.status_code != 200:
                    continue

                from bs4 import BeautifulSoup
                soup = BeautifulSoup(resp.text, 'html.parser')

                # DuckDuckGo HTML results: <a class="result__a" href="...">
                for tag in soup.select('a.result__a, .result__url, a[href*="http"]'):
                    href = tag.get('href', '')
                    # DuckDuckGo wraps URLs: /l/?kh=-1&uddg=https%3A%2F%2F...
                    if 'uddg=' in href:
                        from urllib.parse import unquote, parse_qs
                        qs = parse_qs(href.split('?', 1)[-1])
                        href = unquote(qs.get('uddg', [''])[0])
                    if href.startswith('http') and self._is_plausible_company_site(href, company_name):
                        # Quick liveness check
                        try:
                            r2 = self.session.get(href, timeout=8, verify=False)
                            if r2.status_code == 200 and len(r2.content) > 500:
                                return r2.url   # return the final URL after redirects
                        except Exception:
                            continue
            except Exception:
                continue

        return None

    # ──────────────────────────────────────────────────────────────────────────
    # CAREERS PAGE FINDER
    # ──────────────────────────────────────────────────────────────────────────

    def _find_careers_page(self, website_url):
        """
        1. Fetch homepage and look for a careers/jobs link.
        2. If not found, probe common /careers, /jobs, ... paths directly.
        """
        from urllib.parse import urljoin
        from bs4 import BeautifulSoup

        # ── Step A: scan homepage links ──────────────────────────────────────
        try:
            resp = self.session.get(website_url, timeout=10, verify=False)
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

        # ── Step B: probe common career paths ───────────────────────────────
        from urllib.parse import urlparse
        base = '{uri.scheme}://{uri.netloc}'.format(uri=urlparse(website_url))
        for path in self.CAREER_PATHS:
            candidate = base + path
            if self._is_careers_url_valid(candidate):
                return candidate

        return None

    def _is_careers_url_valid(self, url):
        """Return True if the URL is accessible and looks like a careers page."""
        try:
            resp = self.session.get(url, timeout=8, verify=False)
            if resp.status_code == 200 and len(resp.content) > 300:
                text = resp.text.lower()
                hits = sum(1 for kw in ['job', 'career', 'apply', 'position',
                                        'opening', 'hire', 'opportunity', 'vacancy']
                           if kw in text)
                return hits >= 2
        except Exception:
            pass
        return False

    # ──────────────────────────────────────────────────────────────────────────
    # VALIDATION SCORER
    # ──────────────────────────────────────────────────────────────────────────

    def _score_careers_page(self, careers_url):
        """
        Give a 0–1 confidence score that the careers_url is a real company
        careers page (not a job board or aggregator).
        """
        from bs4 import BeautifulSoup
        score = 0.0
        try:
            resp = self.session.get(careers_url, timeout=10, verify=False)
            if resp.status_code == 200:
                score += 0.4   # page is live
                soup = BeautifulSoup(resp.text, 'html.parser')
                text = soup.get_text().lower()

                # Job-related terms
                job_terms = ['apply', 'position', 'opening', 'vacancy', 'career',
                             'job', 'hire', 'opportunity', 'full-time', 'part-time']
                hits = sum(1 for t in job_terms if t in text)
                score += min(hits * 0.06, 0.3)    # up to +0.3

                # Professional page indicators
                if soup.find('nav') or len(soup.find_all('ul')) > 2:
                    score += 0.15
                if soup.find('footer'):
                    score += 0.1
                if soup.find('title') and soup.find('title').string:
                    score += 0.05
        except Exception:
            pass
        return min(score, 1.0)

# ==================== 100% REAL JOB SCRAPER ====================

class RealJobScraper:
    """100% Real job scraper - Only shows ACTUAL job openings from company websites"""
    
    def __init__(self):
        self.session = self._create_session()
        
    def _create_session(self):
        """Create robust session for web scraping"""
        session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })
        return session
    
    def scrape_real_jobs(self, company_name, careers_url, search_title=None):
        """Scrape ONLY REAL job openings from company career page - NO FALLBACK DATA"""
        if not careers_url:
            return []
        
        try:
            # Show real-time scraping progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("🔍 Connecting to company career page...")
            progress_bar.progress(20)
            
            # Get the careers page
            response = self.session.get(careers_url, timeout=15, verify=False)
            if response.status_code != 200:
                st.warning(f"⚠️ Could not access {company_name} career page (Status: {response.status_code})")
                return []
            
            status_text.text("📄 Analyzing career page content...")
            progress_bar.progress(50)
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            status_text.text("🎯 Extracting real job openings...")
            progress_bar.progress(80)
            
            # Extract REAL job listings
            jobs = self._extract_real_jobs(soup, company_name, careers_url, search_title)
            
            progress_bar.progress(100)
            status_text.empty()
            
            if jobs:
                st.success(f"✅ Found {len(jobs)} REAL job openings at {company_name}")
                return jobs
            else:
                st.info(f"ℹ️ No current job openings found on {company_name}'s career page")
                return []
                
        except Exception as e:
            st.error(f"❌ Error scraping {company_name}: {str(e)}")
            return []
    
    def _extract_real_jobs(self, soup, company_name, base_url, search_title):
        """Extract ONLY REAL job listings from HTML - NO FAKE DATA"""
        jobs = []
        
        # Comprehensive job listing detection
        jobs.extend(self._find_jobs_by_selectors(soup, company_name, base_url))
        jobs.extend(self._find_jobs_by_links(soup, company_name, base_url))
        jobs.extend(self._find_jobs_by_keywords(soup, company_name, base_url))
        
        # Filter for relevance if search title provided
        if search_title:
            jobs = [job for job in jobs if self._is_relevant_to_search(job['title'], search_title)]
        
        # Remove duplicates and limit results
        unique_jobs = self._remove_duplicates(jobs)
        return unique_jobs[:10]  # Return max 10 real jobs
    
    def _find_jobs_by_selectors(self, soup, company_name, base_url):
        """Find jobs using common CSS selectors"""
        jobs = []
        
        # Comprehensive list of job listing selectors used by real companies
        job_selectors = [
            # Common career page structures
            '.job-listing', '.job-item', '.careers-item', '.position',
            '.job-post', '.opening', '.vacancy', '.role',
            '.job-card', '.career-item', '.job-opening',
            
            # Company-specific selectors
            '[data-cy="job-item"]', '[data-testid="job-item"]',
            '.jobs-list-item', '.careers-list-item',
            '.job-list-item', '.opening-list-item',
            
            # Generic job containers
            '[class*="job"]', '[class*="career"]', '[class*="position"]',
            '[class*="opening"]', '[class*="vacancy"]', '[class*="role"]',
            
            # List items that might contain jobs
            'li.job', 'li.career', 'li.position',
            'div.job', 'div.career', 'div.position'
        ]
        
        for selector in job_selectors:
            try:
                elements = soup.select(selector)
                for elem in elements:
                    job = self._parse_job_element(elem, company_name, base_url)
                    if job and self._is_valid_job(job):
                        jobs.append(job)
            except Exception as e:
                continue
        
        return jobs
    
    def _find_jobs_by_links(self, soup, company_name, base_url):
        """Find jobs by analyzing links"""
        jobs = []
        job_links = soup.find_all('a', href=True)
        
        for link in job_links:
            try:
                href = link.get('href', '').lower()
                text = link.get_text('', strip=True)
                
                # Check if this looks like a job link
                if self._is_job_link(href, text) and len(text) > 10:
                    job = {
                        'title': text,
                        'url': urljoin(base_url, href),
                        'company': company_name,
                        'location': self._extract_location_from_context(link),
                        'type': 'Full-time',
                        'source': 'Official Careers Page',
                        'posted': 'Current'
                    }
                    if self._is_valid_job(job):
                        jobs.append(job)
            except:
                continue
        
        return jobs
    
    def _find_jobs_by_keywords(self, soup, company_name, base_url):
        """Find jobs by searching for job-related keywords in the page"""
        jobs = []
        job_keywords = [
            'software engineer', 'developer', 'analyst', 'manager',
            'data scientist', 'cloud engineer', 'devops', 'qa',
            'product manager', 'business analyst', 'consultant'
        ]
        
        # Get all text elements that might contain job titles
        text_elements = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'strong', 'b', 'span', 'div'])
        
        for element in text_elements:
            text = element.get_text(strip=True)
            if text and len(text) > 10 and len(text) < 100:
                for keyword in job_keywords:
                    if keyword.lower() in text.lower():
                        job = {
                            'title': text,
                            'url': base_url,
                            'company': company_name,
                            'location': 'Multiple Locations',
                            'type': 'Full-time',
                            'source': 'Official Careers Page',
                            'posted': 'Current'
                        }
                        if self._is_valid_job(job) and job not in jobs:
                            jobs.append(job)
                            break
        
        return jobs
    
    def _parse_job_element(self, element, company_name, base_url):
        """Parse job information from HTML element"""
        try:
            # Extract title from various possible elements
            title = None
            title_elements = element.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'strong', 'b', 'span'])
            
            for title_elem in title_elements:
                text = title_elem.get_text(strip=True)
                if text and len(text) > 5 and len(text) < 100:
                    title = text
                    break
            
            if not title:
                # Try the element itself
                title = element.get_text(strip=True)
                if len(title) < 5 or len(title) > 100:
                    return None
            
            # Extract URL
            link_elem = element if element.name == 'a' else element.find('a', href=True)
            job_url = base_url
            if link_elem and link_elem.get('href'):
                job_url = urljoin(base_url, link_elem.get('href'))
            
            return {
                'title': title,
                'url': job_url,
                'company': company_name,
                'location': self._extract_location_from_element(element),
                'type': self._extract_job_type_from_element(element),
                'source': 'Official Careers Page',
                'posted': 'Current'
            }
        except:
            return None
    
    def _is_job_link(self, href, text):
        """Check if link is job-related"""
        job_url_patterns = ['/job', '/career', '/position', '/opening', '/vacancy', '/apply']
        job_text_patterns = ['job', 'career', 'position', 'opening', 'vacancy', 'apply', 'hire']
        
        href_match = any(pattern in href for pattern in job_url_patterns)
        text_match = any(pattern in text.lower() for pattern in job_text_patterns)
        
        return href_match or text_match
    
    def _is_valid_job(self, job):
        """Validate that this is a real job posting"""
        if not job or not job.get('title'):
            return False
        
        title = job['title'].lower()
        
        # Exclude navigation and footer elements
        exclude_patterns = [
            'home', 'about', 'contact', 'login', 'signup', 'privacy', 'terms',
            'blog', 'news', 'events', 'support', 'help', 'faq'
        ]
        
        if any(pattern in title for pattern in exclude_patterns):
            return False
        
        # Should be a reasonable length
        if len(job['title']) < 5 or len(job['title']) > 100:
            return False
        
        return True
    
    def _is_relevant_to_search(self, job_title, search_title):
        """Check if job is relevant to the search query"""
        if not search_title:
            return True
        
        job_lower = job_title.lower()
        search_lower = search_title.lower()
        
        # Check for direct keyword matches
        job_words = set(re.findall(r'\w+', job_lower))
        search_words = set(re.findall(r'\w+', search_lower))
        
        common_words = job_words.intersection(search_words)
        return len(common_words) >= 1
    
    def _extract_location_from_element(self, element):
        """Extract location from job element"""
        try:
            # Look for location in nearby elements
            location_selectors = ['.location', '.loc', '.place', '.city', '.country', '.office']
            for selector in location_selectors:
                loc_elem = element.select_one(selector)
                if loc_elem:
                    return loc_elem.get_text(strip=True)
            
            # Check parent and sibling elements
            parent = element.parent
            if parent:
                for selector in location_selectors:
                    loc_elem = parent.select_one(selector)
                    if loc_elem:
                        return loc_elem.get_text(strip=True)
        except:
            pass
        
        return 'Multiple Locations'
    
    def _extract_location_from_context(self, element):
        """Extract location from link context"""
        try:
            # Check sibling elements for location
            parent = element.parent
            if parent:
                location_elements = parent.find_all(class_=re.compile('location|loc|place', re.I))
                if location_elements:
                    return location_elements[0].get_text(strip=True)
        except:
            pass
        
        return 'Multiple Locations'
    
    def _extract_job_type_from_element(self, element):
        """Extract job type from element"""
        try:
            type_selectors = ['.type', '.employment-type', '.job-type', '.time-type']
            for selector in type_selectors:
                type_elem = element.select_one(selector)
                if type_elem:
                    text = type_elem.get_text(strip=True).lower()
                    if any(t in text for t in ['full', 'permanent']):
                        return 'Full-time'
                    elif 'part' in text:
                        return 'Part-time'
                    elif 'contract' in text:
                        return 'Contract'
        except:
            pass
        
        return 'Full-time'
    
    def _remove_duplicates(self, jobs):
        """Remove duplicate job listings"""
        seen_titles = set()
        unique_jobs = []
        
        for job in jobs:
            # Normalize title for comparison
            normalized_title = re.sub(r'[^a-zA-Z0-9]', '', job['title'].lower())
            if normalized_title not in seen_titles:
                unique_jobs.append(job)
                seen_titles.add(normalized_title)
        
        return unique_jobs

# ==================== SELF LEARNING COMPANY VERIFIER ====================

class SelfLearningCompanyVerifier:
    """Enhanced company verifier with automatic discovery capabilities"""
    
    def __init__(self):
        self.known_companies = self._load_company_database()
        self.discovery_engine = CompanyDiscoveryEngine()
        self.job_scraper = RealJobScraper()
        self.learning_enabled = True
        
    def _load_company_database(self):
        """Load comprehensive Indian company database with 150+ companies"""
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
        """Save updated company database"""
        try:
            with open('company_database.json', 'w') as f:
                json.dump(self.known_companies, f, indent=2)
        except:
            pass
    
    def verify_company(self, company_name, enable_discovery=True):
        """Verify company with optional auto-discovery"""
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
        """Check against known companies database"""
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
        """Attempt to auto-discover new company"""
        with st.spinner(f"🤖 **AI Discovery**: Learning about '{company_name_clean}'..."):
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
        """Get statistics about the company database"""
        return {
            "total_companies": len(self.known_companies),
            "auto_learned_companies": len([k for k, v in self.known_companies.items() 
                                         if not k in ['tcs', 'infosys', 'wipro', 'hcl', 
                                                     'google', 'microsoft', 'amazon']]),
            "learning_enabled": self.learning_enabled
        }

# ==================== MAIN APPLICATION ====================

def main():
    st.markdown('<h1 class="main-header">🧠 JobVerification AI <span class="ai-badge">REAL JOBS ONLY</span></h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Verify Job Postings — Fraud Detection + Real Career Page Matching</p>', unsafe_allow_html=True)

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
    st.caption("💡 Step 1: Fraud keyword scan → Step 2: Company lookup → Step 3: Live career page match")
    st.markdown('</div>', unsafe_allow_html=True)

    # ──────────────────────────────────────────────────────────────
    # STEP 1: INPUT FORM
    # ──────────────────────────────────────────────────────────────
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📝 Enter Job Posting Details for Verification")
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
        enable_discovery = st.checkbox("**Enable AI Auto-Discovery**", value=True,
                                       help="Automatically find companies not yet in the database via web scraping")
    verify_clicked = st.button("**🔍 Analyze & Verify Job Posting**",
                               use_container_width=True, type="primary")
    st.markdown('</div>', unsafe_allow_html=True)

    if verify_clicked:
        # Validate all fields
        if not company_name.strip():
            st.warning("⚠️ Please enter the **Company Name**.")
            return
        if not job_title.strip():
            st.warning("⚠️ Please enter the **Job Role / Title**.")
            return
        if not job_description.strip():
            st.warning("⚠️ Please paste the **Job Description**.")
            return


        # STEP 1: ML fraud score (text-only in UI)
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

        # ──────────────────────────────────────────────────────────────
        # STEP 2: FRAUD KEYWORD CHECK — runs FIRST, before any web call
        # ──────────────────────────────────────────────────────────────
        FRAUD_KEYWORDS = [
            'urgent', 'urgently hiring', 'earn $', '5000$', '$5000',
            'earn money', 'earn rs', 'earn per day', 'earn per week',
            'work from phone', 'no experience needed', 'no experience required',
            'no interview', 'weekly payment', 'daily payment',
            'whatsapp', 'telegram', 'guaranteed income', 'make money fast',
            'easy money', 'get rich', 'part time earn', 'home based earn',
            'click here to apply', 'limited seats', 'registration fee',
            'pay to apply', 'processing fee', 'deposit required',
            'send money', 'courier job', 'quick money', 'upfront payment',
            'mlm', 'referral earn', 'joining fee', 'advance payment',
            'lottery', 'prize money', 'investment required', 'jackpot'
        ]

        desc_lower = job_description.lower()
        found_fraud_keywords = [kw for kw in FRAUD_KEYWORDS if kw in desc_lower]

        if found_fraud_keywords:
            # ── RESULT: FRAUDULENT JOB POSTING ──────────────────────
            st.markdown('<div class="fake-card">', unsafe_allow_html=True)
            st.markdown("""
            <div style='font-size:3rem;margin-bottom:0.5rem;'>&#128680;</div>
            <div style='font-size:2rem;font-weight:800;color:#991b1b;'>FRAUDULENT JOB POSTING DETECTED</div>
            <div style='font-size:1.1rem;color:#7f1d1d;margin-top:0.5rem;'>
                This job description contains known fraud indicators.
                <strong>Do NOT apply or share personal/financial details.</strong>
            </div>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            st.error(
                "**Red Flag Keywords Found (" + str(len(found_fraud_keywords)) + "):**\n\n" +
                "\n".join(["- `" + kw + "`" for kw in found_fraud_keywords])
            )
            st.markdown("""
**What You Should Do:**
- Do NOT pay any registration, processing, or joining fee
- Do NOT share Aadhaar, PAN, bank details, or OTPs with the recruiter
- Do NOT contact via WhatsApp or Telegram for job offers
- Report at [cybercrime.gov.in](https://cybercrime.gov.in)
- Always verify openings on the company's official website directly
            """)
            return  # Stop — no further processing needed

        # No fraud keywords found — proceed
        st.info("Step 2 PASSED: No fraud indicators detected. Now verifying the company...")

        # ──────────────────────────────────────────────────────────────
        # STEP 3 / STEP 5: COMPANY LOOKUP + AUTO-DISCOVERY IF NOT IN DB
        # ──────────────────────────────────────────────────────────────
        with st.spinner("Looking up company in verified database..."):
            company_result = company_verifier.verify_company(company_name.strip(), enable_discovery)

        if not company_result["is_genuine"] or not company_result.get("careers_url"):
            # ── RESULT: COMPANY NOT VERIFIABLE ──────────────────────
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
            st.markdown("""
**Safety Warning:**
- This company is not in our verified database
- No official career page was found even with AI auto-discovery
- Do NOT apply or share personal information
- Verify the company independently before proceeding
            """)
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
            st.success("Company Verified: " + company_result['company_name'] + " — " + source_label)

        st.info("Official Career Page: [" + careers_url + "](" + careers_url + ")")

        # ──────────────────────────────────────────────────────────────
        # STEP 3: SCRAPE CAREER PAGE — SEARCH FOR SPECIFIC JOB ROLE
        # ──────────────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### Searching for **'" + job_title.strip() + "'** on " + company_result['company_name'] + "'s Career Page...")

        with st.spinner("Scraping career page for '" + job_title.strip() + "'..."):
            matching_jobs = job_scraper.scrape_real_jobs(
                company_result['company_name'],
                careers_url,
                job_title.strip()
            )

        if matching_jobs:
            # ── RESULT: GENUINE JOB OPENING ─────────────────────────
            st.markdown('<div class="genuine-card">', unsafe_allow_html=True)
            st.markdown(
                "<div style='font-size:3rem;margin-bottom:0.5rem;'>&#10003;</div>"
                "<div style='font-size:2rem;font-weight:800;color:#065f46;'>GENUINE JOB OPENING</div>"
                "<div style='font-size:1.1rem;color:#064e3b;margin-top:0.5rem;'>"
                "<strong>" + company_result['company_name'] + "</strong> is actively hiring for "
                "<strong>'" + job_title.strip() + "'</strong> or a closely matching role.</div>",
                unsafe_allow_html=True
            )
            st.markdown("**Apply Directly**: [" + careers_url + "](" + careers_url + ")")
            st.markdown('</div>', unsafe_allow_html=True)

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

            st.markdown("""
**Professional Advice:**
- These openings are scraped from the company's official career page
- Always apply through the official career portal — never via WhatsApp or Telegram
- A legitimate company will never ask for fees during hiring
- Verify the recruiter's email domain matches the company's official domain
            """)

        else:
            # ── STEP 4: FAKE POSTING — company exists but NOT hiring for this role
            st.markdown('<div class="fake-card">', unsafe_allow_html=True)
            st.markdown(
                "<div style='font-size:3rem;margin-bottom:0.5rem;'>&#10060;</div>"
                "<div style='font-size:2rem;font-weight:800;color:#991b1b;'>FAKE JOB POSTING</div>"
                "<div style='font-size:1.1rem;color:#7f1d1d;margin-top:0.5rem;'>"
                "<strong>" + company_result['company_name'] + "</strong> is a verified real company, "
                "but they are <strong>NOT currently hiring</strong> for "
                "<strong>'" + job_title.strip() + "'</strong> on their official career page. "
                "This posting is likely <strong>fraudulent</strong>.</div>",
                unsafe_allow_html=True
            )
            st.markdown("**Verify on official career page**: [" + careers_url + "](" + careers_url + ")")
            st.markdown('</div>', unsafe_allow_html=True)

            st.error(
                "Why This Is Marked as Fake:\n\n"
                "- The company **" + company_result['company_name'] + "** is real with a verified career page\n"
                "- No current opening for **'" + job_title.strip() + "'** was found on their official page\n"
                "- Fraudsters commonly impersonate real companies to make fake postings look credible"
            )
            st.markdown("""
**What You Should Do:**
- Visit the official career page link above to check directly
- Do NOT apply through the source that shared this posting with you
- Do NOT share personal or financial details with the recruiter
- Contact the company's HR only through their official website
            """)

if __name__ == "__main__":
    main()
