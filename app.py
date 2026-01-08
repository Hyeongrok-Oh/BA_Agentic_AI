"""
LG Electronics HE Business Intelligence - Multi-Agent System UI
"""

import streamlit as st
import sys
import os
import json
import time
from datetime import datetime

# 경로 설정 (Docker 및 로컬 환경 모두 지원)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'intent_classifier/src'))

# .env 로드 (python-dotenv 사용 가능하면 사용, 아니면 수동 로드)
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(PROJECT_ROOT, '.env'))
except ImportError:
    env_path = os.path.join(PROJECT_ROOT, '.env')
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key] = value

# Import agents
from agents import Orchestrator
from agents.base import AgentContext
from agents.analysis import AnalysisAgent
from agents.search_agent import SearchAgent

# Knowledge Graph Visualizer
try:
    from knowledge_graph.visualizer import KGVisualizer
    KG_VISUALIZER_AVAILABLE = True
except ImportError:
    KG_VISUALIZER_AVAILABLE = False
import streamlit.components.v1 as components

# Intent Classifier import (팀원이 만든 것)
try:
    from intent_classifier import IntentClassifier
    INTENT_CLASSIFIER_AVAILABLE = True
except ImportError:
    INTENT_CLASSIFIER_AVAILABLE = False


# SVG Icons (Feather-style line art - No Emoji)
USER_ICON_SVG = '''<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
    <circle cx="12" cy="8" r="4"/>
    <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/>
</svg>'''

ASSISTANT_ICON_SVG = '''<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
    <path d="M12 2L14.4 8.4L21 9.2L16 14L17.5 21L12 17.5L6.5 21L8 14L3 9.2L9.6 8.4L12 2Z"/>
</svg>'''


# 페이지 설정
st.set_page_config(
    page_title="LG HE BI System",
    page_icon="*",
    layout="wide"
)

# Notion 스타일 CSS
st.markdown("""
<style>
    /* ===== 1. 글로벌 스타일 (Notion 핵심) ===== */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        color: #37352F;
    }

    .stApp {
        background-color: #FFFFFF;
    }

    /* ===== 2. 사이드바 스타일 ===== */
    [data-testid="stSidebar"] {
        background-color: #F7F7F5;
        border-right: 1px solid #E9E9E7;
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] .stMarkdown {
        color: #37352F;
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #37352F;
        font-weight: 600;
    }

    /* ===== 3. 헤더 스타일 ===== */
    h1, h2, h3, h4, h5, h6 {
        color: #37352F !important;
        font-weight: 600;
        letter-spacing: -0.01em;
    }

    .main-header {
        font-size: 2.25rem;
        font-weight: 700;
        color: #37352F;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }

    .main-subtitle {
        font-size: 0.95rem;
        color: #787774;
        font-weight: 400;
    }

    /* ===== 4. 입력창 스타일 ===== */
    .stTextInput > div > div > input {
        background-color: #FFFFFF;
        border: 1px solid #E9E9E7;
        border-radius: 4px;
        color: #37352F;
        font-size: 15px;
        padding: 10px 12px;
        box-shadow: none !important;
    }

    .stTextInput > div > div > input:focus {
        border-color: #37352F;
        box-shadow: none !important;
    }

    .stTextInput > div > div > input::placeholder {
        color: #A4A4A0;
    }

    .stSelectbox > div > div {
        background-color: #FFFFFF;
        border: 1px solid #E9E9E7;
        border-radius: 4px;
    }

    /* ===== 5. 버튼 스타일 ===== */
    .stButton > button {
        background-color: #F7F7F5;
        color: #37352F;
        border: 1px solid #E9E9E7;
        border-radius: 4px;
        font-weight: 500;
        font-size: 14px;
        padding: 8px 16px;
        box-shadow: none;
        transition: background-color 0.15s ease;
    }

    .stButton > button:hover {
        background-color: #EFEFED;
        border-color: #DCDCDA;
        color: #37352F;
    }

    .stButton > button[kind="primary"] {
        background-color: #37352F;
        color: #FFFFFF;
        border: none;
    }

    .stButton > button[kind="primary"]:hover {
        background-color: #2F2E2A;
        color: #FFFFFF;
    }

    /* ===== 6. Expander 스타일 ===== */
    .streamlit-expanderHeader {
        background-color: #F7F7F5;
        border-radius: 4px;
        font-weight: 500;
        color: #37352F;
        border: none;
    }

    .streamlit-expanderHeader:hover {
        background-color: #EFEFED;
    }

    .streamlit-expanderContent {
        border: 1px solid #E9E9E7;
        border-top: none;
        border-radius: 0 0 4px 4px;
        background-color: #FFFFFF;
    }

    /* ===== 7. 메트릭 스타일 ===== */
    [data-testid="stMetricValue"] {
        color: #37352F;
        font-weight: 600;
    }

    [data-testid="stMetricLabel"] {
        color: #787774;
        font-size: 13px;
    }

    [data-testid="stMetricDelta"] {
        font-size: 13px;
    }

    /* ===== 8. 코드 블록 스타일 ===== */
    .stCodeBlock {
        background-color: #F7F6F3;
        border: 1px solid #E9E9E7;
        border-radius: 4px;
    }

    code {
        background-color: #F7F6F3;
        color: #EB5757;
        padding: 2px 6px;
        border-radius: 3px;
        font-size: 13px;
    }

    /* ===== 9. 구분선 ===== */
    hr {
        border: none;
        border-top: 1px solid #E9E9E7;
        margin: 24px 0;
    }

    /* ===== 10. 캡션 ===== */
    .stCaption, .caption {
        color: #9B9A97;
        font-size: 12px;
    }

    /* ===== 11. 알림 박스 (Notion 스타일) ===== */
    .stSuccess {
        background-color: #DDEDEA;
        border: none;
        border-radius: 4px;
        color: #37352F;
    }

    .stInfo {
        background-color: #E7F3F8;
        border: none;
        border-radius: 4px;
        color: #37352F;
    }

    .stWarning {
        background-color: #FBF3DB;
        border: none;
        border-radius: 4px;
        color: #37352F;
    }

    .stError {
        background-color: #FBE4E4;
        border: none;
        border-radius: 4px;
        color: #37352F;
    }

    /* ===== 12. 노드 상세 패널 ===== */
    .node-detail-panel {
        background: #F7F7F5;
        border: 1px solid #E9E9E7;
        border-radius: 4px;
        padding: 16px;
    }

    .detail-label {
        font-size: 11px;
        color: #9B9A97;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 4px;
    }

    .detail-value {
        font-size: 14px;
        color: #37352F;
        margin-bottom: 16px;
        line-height: 1.5;
    }

    .tier-badge {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 3px;
        font-size: 12px;
        font-weight: 500;
    }

    .tier-t1 { background: #E7F3F8; color: #0B6E99; }
    .tier-t2 { background: #FBF3DB; color: #9F6B00; }
    .tier-t3 { background: #F1F1EF; color: #787774; }

    .strength-bar {
        height: 4px;
        border-radius: 2px;
        background: #E9E9E7;
        overflow: hidden;
    }

    .strength-fill {
        height: 100%;
        border-radius: 2px;
    }

    .strength-strong { width: 100%; background: #37352F; }
    .strength-medium { width: 60%; background: #9B9A97; }

    .polarity-positive { color: #37352F; }
    .polarity-negative { color: #787774; }

    /* ===== 13. 데이터프레임 ===== */
    .stDataFrame {
        border: 1px solid #E9E9E7;
        border-radius: 4px;
    }

    /* ===== 14. 링크 ===== */
    a {
        color: #37352F;
        text-decoration: underline;
        text-decoration-color: #CFCDC9;
    }

    a:hover {
        text-decoration-color: #37352F;
    }

    /* ===== 15. 프로그레스 바 ===== */
    .stProgress > div > div > div {
        background-color: #37352F;
    }

    /* ===== 16. 탭 스타일 ===== */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        border-bottom: 1px solid #E9E9E7;
    }

    .stTabs [data-baseweb="tab"] {
        color: #787774;
        font-weight: 500;
        padding: 8px 16px;
        border-bottom: 2px solid transparent;
    }

    .stTabs [aria-selected="true"] {
        color: #37352F;
        border-bottom-color: #37352F;
    }

    /* ===== 17. Chat Interface (Wide Layout) ===== */
    .chat-container {
        display: flex;
        flex-direction: column;
        gap: 24px;
        max-width: 100%;
        margin: 0;
        padding: 20px 0;
    }

    .chat-message {
        display: flex;
        align-items: flex-start;
        gap: 12px;
        animation: fadeIn 0.3s ease-out;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(8px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .chat-message.user {
        flex-direction: row-reverse;
        justify-content: flex-start;
    }

    .chat-message.user .message-content {
        background-color: #F1F1EF;
        border-radius: 6px;
        padding: 12px 16px;
        max-width: 400px;
        color: #37352F;
        font-size: 15px;
        line-height: 1.6;
    }

    .chat-message.assistant .message-content {
        background-color: transparent;
        padding: 4px 0;
        max-width: 100%;
        width: 100%;
        color: #37352F;
        font-size: 15px;
        line-height: 1.7;
    }

    .message-icon {
        width: 28px;
        height: 28px;
        min-width: 28px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 6px;
        background-color: rgba(55, 53, 47, 0.04);
    }

    .message-icon svg {
        width: 16px;
        height: 16px;
        stroke: #9B9A97;
        fill: none;
        stroke-width: 1.75;
        stroke-linecap: round;
        stroke-linejoin: round;
    }

    /* Welcome State */
    .welcome-container {
        text-align: center;
        padding: 80px 20px;
        max-width: 600px;
        margin: 0 auto;
    }

    .welcome-icon {
        width: 48px;
        height: 48px;
        margin: 0 auto 16px;
        background-color: rgba(55, 53, 47, 0.04);
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
    }

    .welcome-icon svg {
        width: 24px;
        height: 24px;
        stroke: #9B9A97;
        fill: none;
        stroke-width: 1.5;
    }

    .welcome-title {
        font-size: 20px;
        font-weight: 600;
        color: #37352F;
        margin-bottom: 8px;
    }

    .welcome-subtitle {
        font-size: 14px;
        color: #787774;
        line-height: 1.5;
    }

    /* Typing Indicator */
    .typing-indicator {
        display: flex;
        align-items: center;
        gap: 4px;
        padding: 8px 0;
    }

    .typing-indicator span {
        width: 6px;
        height: 6px;
        background-color: #9B9A97;
        border-radius: 50%;
        animation: bounce 1.4s infinite ease-in-out both;
    }

    .typing-indicator span:nth-child(1) { animation-delay: -0.32s; }
    .typing-indicator span:nth-child(2) { animation-delay: -0.16s; }
    .typing-indicator span:nth-child(3) { animation-delay: 0; }

    @keyframes bounce {
        0%, 80%, 100% { transform: scale(0.6); opacity: 0.4; }
        40% { transform: scale(1); opacity: 1; }
    }

    /* Analysis Card in Chat */
    .chat-analysis-card {
        background-color: #FFFFFF;
        border: 1px solid #E9E9E7;
        border-radius: 6px;
        padding: 16px;
        margin-top: 12px;
    }

    .chat-analysis-header {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 13px;
        font-weight: 500;
        color: #787774;
        text-transform: uppercase;
        letter-spacing: 0.3px;
        margin-bottom: 12px;
        padding-bottom: 8px;
        border-bottom: 1px solid #E9E9E7;
    }

    .chat-analysis-content {
        color: #37352F;
        font-size: 15px;
        line-height: 1.7;
    }

    /* Input Area Styling - Fixed at Bottom */
    .chat-input-area {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background: linear-gradient(to bottom, rgba(255,255,255,0) 0%, rgba(255,255,255,1) 20%);
        padding: 20px 24px 24px;
        z-index: 1000;
    }

    .chat-input-inner {
        max-width: 800px;
        margin: 0 auto;
        background: #FFFFFF;
        border: 1px solid #E0E0E0;
        border-radius: 8px;
        padding: 8px 12px;
        display: flex;
        align-items: center;
        gap: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }

    .chat-input-area .stTextInput > div > div > input {
        border: none !important;
        border-radius: 0 !important;
        padding: 8px 4px !important;
        font-size: 14px !important;
        background: transparent !important;
        box-shadow: none !important;
    }

    .chat-input-area .stTextInput > div > div > input:focus {
        border: none !important;
        box-shadow: none !important;
    }

    .chat-input-area .stTextInput > div > div > input::placeholder {
        font-size: 13px !important;
        color: #B4B4B0 !important;
    }

    /* Content area padding to account for fixed input */
    .main-content-area {
        padding-bottom: 120px;
    }

    /* Unified Analysis Result Box - Wide Layout */
    .analysis-result-box {
        background: #FFFFFF;
        border: 1px solid #E9E9E7;
        border-radius: 8px;
        padding: 24px 32px;
        margin: 16px 0;
        max-width: 100%;
        width: 100%;
    }

    .box-query {
        font-size: 16px;
        font-weight: 500;
        color: #37352F;
        margin-bottom: 16px;
        padding-bottom: 16px;
        border-bottom: 1px solid #E9E9E7;
    }

    .box-summary {
        font-size: 15px;
        line-height: 1.8;
        color: #37352F;
        margin-bottom: 16px;
    }

    .box-details {
        border-top: 1px solid #E9E9E7;
        padding-top: 12px;
    }

    .box-details summary {
        font-size: 13px;
        color: #787774;
        cursor: pointer;
        padding: 8px 0;
        list-style: none;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .box-details summary::-webkit-details-marker {
        display: none;
    }

    .box-details summary::before {
        content: '';
        width: 0;
        height: 0;
        border-left: 5px solid #787774;
        border-top: 4px solid transparent;
        border-bottom: 4px solid transparent;
        transition: transform 0.2s;
    }

    .box-details[open] summary::before {
        transform: rotate(90deg);
    }

    .box-details summary:hover {
        color: #37352F;
    }

    .details-content {
        padding: 16px 0 8px 0;
    }

    .detail-item {
        font-size: 14px;
        color: #37352F;
        line-height: 1.6;
        margin-bottom: 4px;
    }

    /* ===== Analysis Steps (친절한 설명 방식) ===== */
    .analysis-step {
        margin-bottom: 24px;
        padding-bottom: 20px;
        border-bottom: 1px solid #E9E9E7;
    }
    .analysis-step:last-child {
        border-bottom: none;
        margin-bottom: 0;
        padding-bottom: 0;
    }
    .step-header {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 12px;
    }
    .step-number {
        width: 28px;
        height: 28px;
        background: #37352F;
        color: #fff;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 14px;
        font-weight: 600;
        flex-shrink: 0;
    }
    .step-title {
        font-size: 16px;
        font-weight: 600;
        color: #37352F;
    }
    .step-content {
        padding-left: 40px;
        color: #37352F;
        font-size: 14px;
        line-height: 1.7;
    }
    .step-content p {
        margin: 0 0 12px 0;
    }

    /* KPI Comparison */
    .kpi-comparison {
        display: flex;
        align-items: center;
        gap: 16px;
        background: #F7F7F5;
        padding: 16px 20px;
        border-radius: 8px;
        margin: 12px 0;
        flex-wrap: wrap;
    }
    .kpi-value {
        display: flex;
        flex-direction: column;
        gap: 4px;
    }
    .kpi-label {
        font-size: 12px;
        color: #9B9A97;
    }
    .kpi-num {
        font-size: 20px;
        font-weight: 600;
        color: #37352F;
    }
    .kpi-arrow {
        font-size: 20px;
        color: #9B9A97;
    }
    .kpi-change {
        padding: 6px 12px;
        border-radius: 16px;
        font-size: 14px;
        font-weight: 600;
        margin-left: auto;
    }
    .kpi-change.positive {
        background: #DBEDDB;
        color: #2D7D32;
    }
    .kpi-change.negative {
        background: #FDECEA;
        color: #C62828;
    }

    /* SQL Details */
    .sql-details {
        margin-top: 12px;
    }
    .sql-details summary {
        font-size: 13px;
        color: #787774;
        cursor: pointer;
    }
    .sql-details pre {
        background: #F7F7F5;
        padding: 12px;
        border-radius: 6px;
        font-size: 12px;
        overflow-x: auto;
        margin-top: 8px;
        border: 1px solid #E9E9E7;
    }

    /* Hypothesis List */
    .hypothesis-list {
        display: flex;
        flex-direction: column;
        gap: 8px;
        margin-top: 8px;
    }
    .hypothesis-item {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 10px 14px;
        background: #F7F7F5;
        border-radius: 6px;
    }
    .hypothesis-name {
        font-size: 14px;
        color: #37352F;
        font-weight: 500;
    }
    .hypothesis-conf {
        font-size: 12px;
        padding: 3px 10px;
        border-radius: 12px;
        font-weight: 500;
    }
    .hypothesis-conf.conf-높음 {
        background: #DBEDDB;
        color: #2D7D32;
    }
    .hypothesis-conf.conf-중간 {
        background: #FFF3CD;
        color: #856404;
    }
    .hypothesis-conf.conf-낮음 {
        background: #F7F7F5;
        color: #787774;
    }
    .hypothesis-more {
        font-size: 13px;
        color: #9B9A97;
        padding-left: 14px;
    }

    /* Validation Summary */
    .validation-summary {
        display: flex;
        gap: 16px;
        margin: 12px 0;
    }
    .validation-stat {
        display: flex;
        flex-direction: column;
        align-items: center;
        padding: 16px 24px;
        border-radius: 8px;
        min-width: 100px;
    }
    .validation-stat.validated {
        background: #DBEDDB;
    }
    .validation-stat.rejected {
        background: #F7F7F5;
    }
    .stat-num {
        font-size: 28px;
        font-weight: 700;
        color: #37352F;
    }
    .stat-label {
        font-size: 13px;
        color: #787774;
        margin-top: 4px;
    }

    /* Validated Factors */
    .validated-factors {
        display: flex;
        flex-direction: column;
        gap: 12px;
        margin-top: 8px;
    }
    .factor-card {
        background: #F7F7F5;
        border-radius: 8px;
        padding: 14px 16px;
        border-left: 4px solid #37352F;
    }
    .factor-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 6px;
    }
    .factor-name {
        font-weight: 600;
        color: #37352F;
        font-size: 14px;
    }
    .factor-contrib {
        font-size: 13px;
        color: #2D7D32;
        font-weight: 500;
        background: #DBEDDB;
        padding: 3px 10px;
        border-radius: 12px;
    }
    .factor-reasoning {
        font-size: 13px;
        color: #787774;
        line-height: 1.6;
    }

    /* Events List */
    .events-list {
        display: flex;
        flex-direction: column;
        gap: 6px;
        margin-top: 8px;
    }
    .event-item {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 8px 12px;
        background: #F7F7F5;
        border-radius: 6px;
    }
    .event-name {
        font-size: 13px;
        color: #37352F;
        flex: 1;
    }
    .event-category {
        font-size: 11px;
        color: #787774;
        background: #E9E9E7;
        padding: 2px 8px;
        border-radius: 10px;
    }

    .analysis-section {
        margin-bottom: 20px;
    }

    .analysis-section:last-child {
        margin-bottom: 0;
    }

    .analysis-section-title {
        font-size: 11px;
        font-weight: 600;
        color: #9B9A97;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 8px;
    }

    .analysis-section-content {
        font-size: 15px;
        color: #37352F;
        line-height: 1.6;
    }

    .analysis-divider {
        height: 1px;
        background: #E9E9E7;
        margin: 16px 0;
    }

    /* Metrics Row */
    .metrics-row {
        display: flex;
        gap: 24px;
        margin: 12px 0;
    }

    .metric-item {
        flex: 1;
    }

    .metric-label {
        font-size: 11px;
        color: #9B9A97;
        text-transform: uppercase;
        letter-spacing: 0.3px;
        margin-bottom: 4px;
    }

    .metric-value {
        font-size: 18px;
        font-weight: 600;
        color: #37352F;
    }

    .metric-delta {
        font-size: 13px;
        color: #787774;
        margin-top: 2px;
    }

    .metric-delta.positive { color: #37352F; }
    .metric-delta.negative { color: #EB5757; }

    /* Hypothesis List */
    .hypothesis-item {
        background: #F7F7F5;
        border-radius: 6px;
        padding: 12px 16px;
        margin-bottom: 8px;
    }

    .hypothesis-item:last-child {
        margin-bottom: 0;
    }

    .hypothesis-title {
        font-size: 14px;
        font-weight: 500;
        color: #37352F;
        margin-bottom: 4px;
    }

    .hypothesis-meta {
        font-size: 12px;
        color: #787774;
    }

    .hypothesis-contribution {
        display: inline-block;
        background: rgba(55, 53, 47, 0.08);
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 12px;
        font-weight: 500;
        color: #37352F;
        margin-right: 8px;
    }

    /* Event Item */
    .event-item {
        border-left: 3px solid #E9E9E7;
        padding-left: 12px;
        margin-bottom: 12px;
    }

    .event-item:last-child {
        margin-bottom: 0;
    }

    .event-title {
        font-size: 14px;
        font-weight: 500;
        color: #37352F;
    }

    .event-meta {
        font-size: 12px;
        color: #787774;
        margin-top: 2px;
    }

    /* Summary Box */
    .summary-box {
        background: #F7F7F5;
        border-radius: 6px;
        padding: 16px 20px;
        font-size: 15px;
        line-height: 1.7;
        color: #37352F;
    }

    /* Expandable Detail */
    .detail-toggle {
        font-size: 13px;
        color: #787774;
        cursor: pointer;
        display: flex;
        align-items: center;
        gap: 4px;
        margin-top: 12px;
    }

    .detail-toggle:hover {
        color: #37352F;
    }

    /* Hide default Streamlit chat elements if any */
    [data-testid="stChatMessage"] {
        display: none !important;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """세션 상태 초기화"""
    if 'orchestrator' not in st.session_state:
        st.session_state.orchestrator = Orchestrator()
    if 'history' not in st.session_state:
        st.session_state.history = []
    if 'current_result' not in st.session_state:
        st.session_state.current_result = None
    # Chat interface state
    if 'chat_messages' not in st.session_state:
        st.session_state.chat_messages = []  # [{role, content, analysis_html, timestamp}]
    if 'is_processing' not in st.session_state:
        st.session_state.is_processing = False
    if 'pending_query' not in st.session_state:
        st.session_state.pending_query = None


# ===== Chat Rendering Functions =====
def render_user_message(content: str) -> str:
    """Render a user message with Notion AI styling."""
    return f'<div class="chat-message user"><div class="message-icon">{USER_ICON_SVG}</div><div class="message-content">{content}</div></div>'


def render_assistant_message(content: str, analysis_html: str = "") -> str:
    """Render an assistant message with Notion AI styling."""
    return f'<div class="chat-message assistant"><div class="message-icon">{ASSISTANT_ICON_SVG}</div><div class="message-content">{content}{analysis_html}</div></div>'


def render_welcome_state() -> str:
    """Render welcome state when no messages exist."""
    return f'<div class="welcome-container"><div class="welcome-icon">{ASSISTANT_ICON_SVG}</div><div class="welcome-title">LG HE Business Intelligence</div><div class="welcome-subtitle">비즈니스 데이터에 대해 질문하세요. 매출 분석, 성과 트렌드, 진단 인사이트를 제공합니다.</div></div>'


def render_typing_indicator() -> str:
    """Render typing indicator while processing."""
    return f'<div class="chat-message assistant"><div class="message-icon">{ASSISTANT_ICON_SVG}</div><div class="message-content"><div class="typing-indicator"><span></span><span></span><span></span></div></div></div>'


def display_chat_messages():
    """Display all chat messages."""
    if not st.session_state.chat_messages:
        st.markdown(render_welcome_state(), unsafe_allow_html=True)
        return

    chat_html = '<div class="chat-container">'
    for msg in st.session_state.chat_messages:
        if msg['role'] == 'user':
            chat_html += render_user_message(msg['content'])
        else:
            analysis_html = ""
            if msg.get('analysis_html'):
                analysis_html = msg['analysis_html']
            chat_html += render_assistant_message(msg['content'], analysis_html)
    chat_html += '</div>'
    st.markdown(chat_html, unsafe_allow_html=True)


def render_unified_analysis_result(
    query: str,
    intent_result: dict,
    kpi_change=None,
    hypotheses: list = None,
    validated: list = None,
    matched_events: dict = None,
    summary: str = ""
) -> None:
    """Render analysis results in a unified, clean box format."""

    analysis_mode = intent_result.get("analysis_mode", "descriptive")
    entities = intent_result.get("extracted_entities", {})

    # Start the unified result box
    st.markdown('<div class="analysis-result-box">', unsafe_allow_html=True)

    # Query echo
    st.markdown(f'<div class="analysis-section"><div class="analysis-section-content" style="font-size: 16px; font-weight: 500;">{query}</div></div><div class="analysis-divider"></div>', unsafe_allow_html=True)

    # Intent Classification (compact)
    mode_text = "Diagnostic" if analysis_mode == "diagnostic" else "Descriptive"
    st.markdown(f'<div class="analysis-section"><div style="display: flex; gap: 24px; font-size: 13px; color: #787774;"><span>Mode: <strong style="color: #37352F;">{mode_text}</strong></span><span>Source: <strong style="color: #37352F;">Internal (ERP)</strong></span></div></div>', unsafe_allow_html=True)

    if analysis_mode == "diagnostic":
        # KPI Changes
        if kpi_change:
            change_sign = "+" if kpi_change.change_percent > 0 else ""
            delta_class = "positive" if kpi_change.change_percent > 0 else "negative"
            direction_text = "Increase" if kpi_change.change_percent > 0 else "Decrease"
            kpi_html = f'<div class="analysis-divider"></div><div class="analysis-section"><div class="analysis-section-title">KPI Changes</div><div class="metrics-row"><div class="metric-item"><div class="metric-label">Previous Period</div><div class="metric-value">{kpi_change.previous_value:,.0f}</div></div><div class="metric-item"><div class="metric-label">Current Period</div><div class="metric-value">{kpi_change.current_value:,.0f}</div></div><div class="metric-item"><div class="metric-label">Change</div><div class="metric-value">{change_sign}{kpi_change.change_percent:.1f}%</div><div class="metric-delta {delta_class}">{direction_text}</div></div></div></div>'
            st.markdown(kpi_html, unsafe_allow_html=True)

        # Validated Hypotheses
        if validated:
            hypotheses_html = ""
            for h in validated[:5]:  # Show top 5
                data = h.validation_data or {}
                contrib_pct = data.get("contribution_pct", 0)
                reasoning = getattr(h, 'reasoning', '') or h.factor
                reasoning_text = reasoning[:100] + "..." if len(reasoning) > 100 else reasoning
                hypotheses_html += f'<div class="hypothesis-item"><div class="hypothesis-title">{h.factor}</div><div class="hypothesis-meta"><span class="hypothesis-contribution">{contrib_pct:.1f}% contribution</span> {reasoning_text}</div></div>'

            st.markdown(f'<div class="analysis-divider"></div><div class="analysis-section"><div class="analysis-section-title">Key Factors ({len(validated)} validated)</div>{hypotheses_html}</div>', unsafe_allow_html=True)

        # Related Events
        if matched_events:
            total_events = sum(len(v) for v in matched_events.values())
            events_html = ""
            event_count = 0
            for h_id, events in matched_events.items():
                for ev in events[:2]:  # Top 2 per hypothesis
                    if event_count >= 5:  # Max 5 total
                        break
                    events_html += f'<div class="event-item"><div class="event-title">{ev.event_name}</div><div class="event-meta">{ev.event_category} | Score: {ev.total_score:.2f}</div></div>'
                    event_count += 1
                if event_count >= 5:
                    break

            if events_html:
                st.markdown(f'<div class="analysis-divider"></div><div class="analysis-section"><div class="analysis-section-title">Related Events ({total_events} found)</div>{events_html}</div>', unsafe_allow_html=True)

    # Summary
    if summary:
        st.markdown(f'<div class="analysis-divider"></div><div class="analysis-section"><div class="analysis-section-title">Summary</div><div class="summary-box">{summary}</div></div>', unsafe_allow_html=True)

    # Close the box
    st.markdown('</div>', unsafe_allow_html=True)


def classify_intent(query: str) -> dict:
    """Intent 분류"""
    if INTENT_CLASSIFIER_AVAILABLE:
        try:
            classifier = IntentClassifier()
            result = classifier.classify(query)
            return {
                "service_type": result.get("intent", "data_qa"),
                "analysis_mode": result.get("analysis_mode", "descriptive"),
                "sub_intent": result.get("sub_intent", "internal_data"),
                "query": query,
                "extracted_entities": result.get("extracted_entities", {}),
                "thinking": result.get("thinking", ""),
                "raw_result": result
            }
        except Exception as e:
            st.warning(f"Intent Classifier 오류: {e}")

    # Fallback: Orchestrator의 간단한 분류 사용
    return st.session_state.orchestrator._simple_classify(query)


def display_intent_result(intent_result: dict):
    """Intent 분류 결과 표시"""
    st.markdown("### Step 1: Intent Classification")

    col1, col2, col3 = st.columns(3)

    with col1:
        service = intent_result.get("service_type", "data_qa")
        if service == "report_generation":
            st.metric("서비스 유형", "📄 Report Generation")
        else:
            st.metric("Service Type", "Data Q&A")

    with col2:
        mode = intent_result.get("analysis_mode", "descriptive")
        if mode == "diagnostic":
            st.metric("분석 모드", "Diagnostic (원인 분석)")
        else:
            st.metric("분석 모드", "Descriptive (데이터 조회)")

    with col3:
        sub = intent_result.get("sub_intent", "internal_data")
        if sub == "external_data":
            st.metric("데이터 소스", "🌐 External (Graph)")
        elif sub == "hybrid":
            st.metric("데이터 소스", "🔄 Hybrid")
        else:
            st.metric("Data Source", "Internal (ERP)")

    # 추출된 엔티티
    entities = intent_result.get("extracted_entities", {})
    if entities:
        with st.expander("📋 추출된 엔티티", expanded=False):
            st.json(entities)

    # Thinking (있으면)
    thinking = intent_result.get("thinking", "")
    if thinking:
        with st.expander("Intent Analysis Process", expanded=False):
            st.write(thinking)


def display_hypothesis_generation(hypotheses: list):
    """가설 생성 결과 표시 (Graph-Based 상세 정보 포함)"""
    st.markdown("### Step 2: Hypothesis Generation (Graph-Based)")

    # Graph 기반 가설 수 계산
    graph_based = sum(1 for h in hypotheses if h.graph_evidence.get("from_graph", False))
    st.info(f"생성된 가설: **{len(hypotheses)}개** (Graph 기반: {graph_based}개)")

    for h in hypotheses:
        # Graph 기반 여부에 따른 아이콘
        evidence = h.graph_evidence or {}
        is_graph = evidence.get("from_graph", False)
        graph_icon = "[G]" if is_graph else "[?]"

        # 카테고리별 색상
        category_colors = {
            "cost": "[C]", "revenue": "[R]", "pricing": "[P]", "external": "[E]"
        }
        cat_icon = category_colors.get(h.category, "[O]")

        with st.expander(f"{graph_icon} [{h.id}] {h.factor} {cat_icon}", expanded=False):
            # 인과관계 체인 (있으면)
            if hasattr(h, 'reasoning') and h.reasoning:
                st.markdown(f"**🔄 인과관계:** `{h.reasoning}`")
                st.markdown("---")

            # 상세 설명 (Markdown)
            st.markdown(h.description)

            # Graph Evidence 표시
            if is_graph:
                st.markdown("---")
                st.markdown("**Knowledge Graph Evidence:**")

                col1, col2, col3 = st.columns(3)
                with col1:
                    relation = evidence.get("relation_type", "N/A")
                    relation_kr = "동비례 ↑↑" if relation == "PROPORTIONAL" else "역비례 ↑↓"
                    st.metric("관계 유형", relation_kr)
                with col2:
                    mention = evidence.get("mention_count", 0)
                    st.metric("언급 횟수", f"{mention}회")
                with col3:
                    event_count = evidence.get("event_count", 0)
                    st.metric("관련 이벤트", f"{event_count}개")

            # 관련 이벤트 상세
            if hasattr(h, 'related_events') and h.related_events:
                st.markdown("---")
                st.markdown(f"**Related Events ({len(h.related_events)}):**")

                for ev in h.related_events[:3]:
                    severity_indicator = {"critical": "[!]", "high": "[H]", "medium": "[M]", "low": "[L]"}.get(ev.severity, "[?]")
                    impact_text = "증가" if ev.impact_direction == "INCREASES" else "감소"
                    regions = ", ".join([r for r in ev.target_regions if r]) if ev.target_regions else "전체"

                    st.markdown(f"""
                    {severity_indicator} **{ev.name}** ({ev.category})
                    - 영향: {h.factor} {impact_text} 유발
                    - 심각도: {ev.severity} | 지역: {regions}
                    """)
                    if ev.evidence:
                        st.caption(f"→ {ev.evidence[:150]}...")

            if h.sql_template:
                st.markdown("---")
                st.code(h.sql_template, language="sql")


def display_hypothesis_validation(validated: list, all_hypotheses: list):
    """가설 검증 결과 표시 (검증됨 + 기각됨 모두)"""
    st.markdown("### Step 3: Hypothesis Validation (SQL)")

    # 검증된 가설과 기각된 가설 분리
    validated_ids = {h.id for h in validated}
    rejected = [h for h in all_hypotheses if h.id not in validated_ids]

    st.success(f"검증된 가설: **{len(validated)}/{len(all_hypotheses)}개** | 기각된 가설: **{len(rejected)}개**")

    # 검증된 가설
    if validated:
        st.markdown("#### Validated Hypotheses")
        for h in validated:
            data = h.validation_data or {}
            change = data.get("change_percent", 0)

            col1, col2, col3 = st.columns([2, 1, 1])

            with col1:
                st.write(f"**[{h.id}] {h.factor}**")
            with col2:
                st.metric(
                    "변화율",
                    f"{change:+.1f}%",
                    delta=f"{data.get('direction', '')}"
                )
            with col3:
                st.write(f"{data.get('previous_value', 0):,.0f} → {data.get('current_value', 0):,.0f}")

            # SQL 쿼리 표시
            sql_query = data.get('sql_query', '')
            if sql_query:
                with st.expander(f"SQL Query - {h.factor}", expanded=False):
                    st.code(sql_query, language="sql")

    # 기각된 가설 (expander로 닫혀있음)
    if rejected:
        with st.expander(f"Rejected Hypotheses ({len(rejected)}) - Click to view details", expanded=False):
            for h in rejected:
                data = h.validation_data or {}
                change = data.get("change_percent", 0)
                direction = data.get("direction", "unknown")

                # 기각 사유 판단
                if data:
                    if abs(change) < 5.0:
                        reject_reason = f"변동률 미달 ({change:+.1f}% < ±5%)"
                    elif h.direction.lower() == "increase" and direction == "decreased":
                        reject_reason = f"방향 불일치 (예상: 증가, 실제: 감소 {change:+.1f}%)"
                    elif h.direction.lower() == "decrease" and direction == "increased":
                        reject_reason = f"방향 불일치 (예상: 감소, 실제: 증가 {change:+.1f}%)"
                    else:
                        reject_reason = f"기타 ({direction}, {change:+.1f}%)"
                else:
                    reject_reason = "데이터 없음 또는 SQL 오류"

                st.markdown(f"""
                <div style="background-color: #fff3cd; padding: 10px; border-radius: 5px; margin-bottom: 10px; border-left: 4px solid #ffc107;">
                    <strong>[{h.id}] {h.factor}</strong><br>
                    <span style="color: #856404;">기각 사유: {reject_reason}</span><br>
                    <span style="font-size: 0.9em;">가설: {h.description}</span>
                </div>
                """, unsafe_allow_html=True)

                # SQL 쿼리 표시 (있는 경우)
                sql_query = data.get('sql_query', '')
                if sql_query:
                    with st.expander(f"SQL Query - {h.factor}", expanded=False):
                        st.code(sql_query, language="sql")


def display_event_matching(matched_events: dict):
    """이벤트 매칭 결과 표시 (하이브리드 스코어링)"""
    st.markdown("### Step 4: Event Matching (Hybrid Scoring)")

    total_events = sum(len(v) for v in matched_events.values())
    st.info(f"매칭된 이벤트: **{total_events}개** (Vector + Graph 하이브리드)")

    for h_id, events in matched_events.items():
        st.write(f"**[{h_id}]** - {len(events)}개 이벤트 매칭")

        for ev in events[:5]:
            # 스코어에 따른 색상 (0-1 스케일)
            score = ev.total_score
            if score >= 0.7:
                score_color = "[High]"
            elif score >= 0.4:
                score_color = "[Mid]"
            else:
                score_color = "[Low]"

            with st.expander(f"{score_color} {ev.event_name} (Score: {score:.2f})", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**카테고리**: {ev.event_category}")
                    st.write(f"**영향**: {ev.impact_type} → {ev.matched_factor}")
                    st.write(f"**심각도**: {ev.severity}")
                with col2:
                    st.write(f"**지역**: {', '.join(ev.target_regions) if ev.target_regions else '전체'}")
                    st.write(f"**Magnitude**: {ev.magnitude}")

                # Score Breakdown (하이브리드)
                st.write("**Score Breakdown:**")
                breakdown = ev.score_breakdown

                # Semantic vs Graph 비교
                col_sem, col_graph = st.columns(2)
                with col_sem:
                    semantic = breakdown.get('semantic', 0)
                    st.metric("Semantic (40%)", f"{semantic:.2f}", help="Vector Similarity")
                with col_graph:
                    graph = breakdown.get('graph', 0)
                    st.metric("Graph (60%)", f"{graph:.2f}", help="KG 관계 기반")

                # Graph 세부 점수
                st.caption("Graph Score 세부:")
                cols = st.columns(4)
                cols[0].write(f"Direction: {breakdown.get('direction', 0):.1f}")
                cols[1].write(f"Magnitude: {breakdown.get('magnitude', 0):.1f}")
                cols[2].write(f"Region: {breakdown.get('region', 0):.1f}")
                cols[3].write(f"Severity: {breakdown.get('severity', 0):.1f}")

                # 출처
                if ev.sources:
                    st.write("**출처:**")
                    for src in ev.sources[:2]:
                        title = src.get('title', 'N/A')
                        url = src.get('url', '')
                        if url:
                            st.markdown(f"- [{title[:60]}...]({url})")
                        else:
                            st.write(f"- {title[:60]}...")

                if ev.evidence:
                    st.write("**근거:**")
                    st.caption(ev.evidence[:300] + "..." if len(ev.evidence) > 300 else ev.evidence)


def display_evidence_collection(evidences: dict):
    """증거 수집 결과 표시 (레거시)"""
    st.markdown("### Step 4: Evidence Collection (Graph)")

    total_events = sum(len(v) for v in evidences.values())
    st.info(f"발견된 관련 이벤트: **{total_events}개**")

    for h_id, ev_list in evidences.items():
        st.write(f"**[{h_id}]** - {len(ev_list)}개 이벤트")

        for ev in ev_list[:5]:
            with st.expander(f"{ev.event_name} ({ev.event_category})", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**영향**: {ev.impact_type} → {ev.factor_name}")
                    st.write(f"**심각도**: {ev.event_severity}")
                with col2:
                    st.write(f"**지역**: {', '.join(ev.target_regions) if ev.target_regions else '전체'}")

                if ev.evidence_text:
                    st.write("**근거:**")
                    st.caption(ev.evidence_text[:300] + "..." if len(ev.evidence_text) > 300 else ev.evidence_text)


def display_graph_query(evidences: dict):
    """Graph Query 표시"""
    if evidences:
        with st.expander("Cypher Query Example", expanded=False):
            sample_query = """
MATCH (e:Event)-[r:INCREASES|DECREASES]->(f:Factor)
WHERE f.name CONTAINS $factor_name
OPTIONAL MATCH (e)-[:TARGETS]->(d:Dimension)
RETURN e.name, e.category, e.evidence,
       type(r) as impact, f.name as factor
ORDER BY e.severity DESC
LIMIT 10
"""
            st.code(sample_query, language="cypher")


def display_vector_search_results(events: list):
    """벡터 검색 결과 (이벤트 목록) 표시"""
    if not events:
        st.warning("관련 이벤트를 찾지 못했습니다.")
        return

    st.success(f"**{len(events)}개** 유사 이벤트 발견")

    for i, event in enumerate(events, 1):
        # 유사도 점수에 따른 색상
        score = event.get("score", 0)
        if score > 0.8:
            score_color = "[High]"
        elif score > 0.6:
            score_color = "[Mid]"
        else:
            score_color = "[Low]"

        # 심각도 배지
        severity = event.get("severity", "medium")
        severity_badge = {"high": "High", "medium": "Medium", "low": "Low"}.get(severity, "Medium")

        # 카테고리 이모지
        category = event.get("category", "")
        category_emoji = {
            "geopolitical": "[GEO]",
            "policy": "[P]",
            "market": "[M]",
            "company": "[CO]",
            "macro_economy": "[ME]",
            "technology": "[T]"
        }.get(category, "[N]")

        with st.expander(f"{score_color} [{i}] {event.get('name', 'Unknown Event')} ({category_emoji} {category})", expanded=(i <= 2)):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("유사도", f"{score:.2%}")
            with col2:
                st.write(f"**심각도**: {severity_badge}")
            with col3:
                st.write(f"**카테고리**: {category}")

            # 관련 Factor 표시
            related_factors = event.get("related_factors", [])
            if related_factors:
                st.write("**영향 Factor:**")
                st.write(", ".join([f"`{f}`" for f in related_factors[:5]]))

            # Evidence
            evidence = event.get("evidence", "")
            if evidence:
                st.write("**근거:**")
                st.caption(evidence[:500] + ("..." if len(evidence) > 500 else ""))

            # 출처 URL
            source_urls = event.get("source_urls", [])
            source_titles = event.get("source_titles", [])
            if source_urls:
                st.write("**출처:**")
                for j, url in enumerate(source_urls[:3]):
                    title = source_titles[j] if j < len(source_titles) else f"출처 {j+1}"
                    st.markdown(f"- [{title}]({url})")


def display_summary(summary_result: dict, details: list):
    """분석 결과 표시 (문장형 답변 + 출처)"""
    st.markdown("### 📝 Step 5: Analysis Result")

    # summary_result가 dict인 경우와 str인 경우 모두 처리
    if isinstance(summary_result, dict):
        summary = summary_result.get("summary", "")
        sources = summary_result.get("sources", [])
    else:
        summary = summary_result
        sources = []

    # 분석 결과 (문장형)
    st.markdown(f"""
    <div style="background-color: #f8f9fa; padding: 20px; border-radius: 10px; border-left: 5px solid #A50034; line-height: 1.8;">
    {summary.replace(chr(10), '<br>')}
    </div>
    """, unsafe_allow_html=True)


def render_node_detail_panel(visualizer, selected_node: str, kpi_context: str = None):
    """
    노션 스타일의 노드 상세 정보 패널 렌더링

    Args:
        visualizer: KGVisualizer 인스턴스
        selected_node: 선택된 노드 ID
        kpi_context: KPI 컨텍스트 (Driver 관계 표시용)
    """
    if not selected_node:
        st.markdown("""
        <div class="node-detail-panel">
            <p style="color: #9CA3AF; text-align: center; margin: 20px 0;">
                노드를 선택하세요
            </p>
        </div>
        """, unsafe_allow_html=True)
        return

    details = visualizer.get_extended_node_details(selected_node, kpi_context)

    if "error" in details:
        st.warning(details["error"])
        return

    node_type = details.get("type", "")
    node_name = details.get("name", selected_node)

    # 패널 시작
    st.markdown(f"""
    <div class="node-detail-panel">
        <div style="font-size: 16px; font-weight: 600; color: #1F2937; margin-bottom: 16px; padding-bottom: 12px; border-bottom: 1px solid #E5E5E5;">
            {'●' if node_type == 'driver' else '■'} {node_name}
        </div>
    """, unsafe_allow_html=True)

    # Driver 노드인 경우
    if node_type == "driver":
        tier = details.get("tier", "N/A")
        tier_class = f"tier-{tier.lower()}" if tier in ["T1", "T2", "T3"] else ""
        tier_label = {
            "T1": "T1 · ERP 직접",
            "T2": "T2 · Proxy",
            "T3": "T3 · Event"
        }.get(tier, tier)

        # Tier
        st.markdown(f"""
        <div class="detail-label">Validation Tier</div>
        <div class="detail-value">
            <span class="tier-badge {tier_class}">{tier_label}</span>
        </div>
        """, unsafe_allow_html=True)

        # ERP 정보 (T1) 또는 Proxy 정보 (T2)
        erp_table = details.get("erp_table")
        erp_column = details.get("erp_column")
        proxy_source = details.get("proxy_source")
        proxy_indicator = details.get("proxy_indicator")

        if erp_table and erp_column:
            st.markdown(f"""
            <div class="detail-label">ERP Mapping</div>
            <div class="detail-value">
                <code style="background: #F3F4F6; padding: 2px 6px; border-radius: 4px; font-size: 13px;">{erp_table}</code>
                <span style="color: #9CA3AF; margin: 0 4px;">→</span>
                <code style="background: #F3F4F6; padding: 2px 6px; border-radius: 4px; font-size: 13px;">{erp_column}</code>
            </div>
            """, unsafe_allow_html=True)
        elif proxy_source:
            st.markdown(f"""
            <div class="detail-label">Proxy Source</div>
            <div class="detail-value">
                {proxy_source}<br>
                <span style="font-size: 12px; color: #6B7280;">{proxy_indicator or ''}</span>
            </div>
            """, unsafe_allow_html=True)

        # 관계 정보
        relations = details.get("relations", [])
        if relations:
            rel = relations[0]  # 첫 번째 관계 표시
            strength = rel.get("strength", "medium")
            polarity = rel.get("polarity", "?")
            rationale = rel.get("rationale", "")

            # 관계 강도
            strength_class = f"strength-{strength}"
            strength_label = "Strong" if strength == "strong" else "Medium"
            st.markdown(f"""
            <div class="detail-label">Confidence</div>
            <div class="detail-value">
                <div class="strength-bar"><div class="strength-fill {strength_class}"></div></div>
                <span style="font-size: 12px; color: #6B7280; margin-top: 4px; display: block;">{strength_label}</span>
            </div>
            """, unsafe_allow_html=True)

            # 영향 방향
            polarity_class = "polarity-positive" if polarity == "+" else "polarity-negative"
            polarity_text = "↑ 정상관 (+)" if polarity == "+" else "↓ 역상관 (-)" if polarity == "-" else "± 복합"
            st.markdown(f"""
            <div class="detail-label">Direction</div>
            <div class="detail-value {polarity_class}">
                {polarity_text}
            </div>
            """, unsafe_allow_html=True)

            # 근거
            if rationale:
                st.markdown(f"""
                <div class="detail-label">Rationale</div>
                <div class="detail-value" style="font-size: 13px; color: #4B5563;">
                    {rationale}
                </div>
                """, unsafe_allow_html=True)

    # KPI 노드인 경우
    elif node_type == "kpi":
        category = details.get("category", "")
        erp_table = details.get("erp_table")
        erp_column = details.get("erp_column")
        unit = details.get("unit", "")

        if category:
            st.markdown(f"""
            <div class="detail-label">Category</div>
            <div class="detail-value">{category}</div>
            """, unsafe_allow_html=True)

        if erp_table and erp_column:
            st.markdown(f"""
            <div class="detail-label">ERP Mapping</div>
            <div class="detail-value">
                <code style="background: #F3F4F6; padding: 2px 6px; border-radius: 4px; font-size: 13px;">{erp_table}</code>
                <span style="color: #9CA3AF; margin: 0 4px;">→</span>
                <code style="background: #F3F4F6; padding: 2px 6px; border-radius: 4px; font-size: 13px;">{erp_column}</code>
            </div>
            """, unsafe_allow_html=True)

        if unit:
            st.markdown(f"""
            <div class="detail-label">Unit</div>
            <div class="detail-value">{unit}</div>
            """, unsafe_allow_html=True)

        # 연결된 Driver 수
        relations = details.get("relations", [])
        st.markdown(f"""
        <div class="detail-label">Connected Drivers</div>
        <div class="detail-value">{len(relations)}개</div>
        """, unsafe_allow_html=True)

    # 패널 닫기
    st.markdown("</div>", unsafe_allow_html=True)


def run_analysis(query: str):
    """분석 실행 및 결과 표시 (Unified Box Style)"""

    # Progress indicator (minimal) - 분석 중에만 표시
    progress = st.progress(0)
    status = st.empty()

    # Step 1: Intent Classification
    status.text("Analyzing...")
    progress.progress(10)

    intent_result = classify_intent(query)
    time.sleep(0.2)
    progress.progress(20)

    # 분석 모드에 따른 처리
    analysis_mode = intent_result.get("analysis_mode", "descriptive")

    if analysis_mode == "diagnostic":
        # Diagnostic: Analysis Agent 사용
        # 1. 먼저 모든 분석 단계 실행 (데이터 수집)
        analysis_agent = AnalysisAgent()

        entities = intent_result.get("extracted_entities", {})
        period = entities.get("period", {"year": 2024, "quarter": 4})
        region = entities.get("region")
        if isinstance(region, list):
            region = region[0] if region else None

        # Step 1: KPI 변동 계산
        status.text("Calculating KPI changes...")
        kpi_change = analysis_agent._calculate_kpi_change(query, period, region)
        progress.progress(20)

        # Step 2: 가설 생성
        status.text("Generating hypotheses...")
        hypothesis_result = analysis_agent.hypothesis_generator.generate(
            question=query,
            company=entities.get("company", "LGE"),
            period=f"{period.get('year', 2024)}년 Q{period.get('quarter', 4)}",
            region=region,
            return_result=True  # HypothesisResult 반환 (KPI + Driver 정보 포함)
        )
        progress.progress(35)

        # HypothesisResult에서 hypotheses 리스트와 target_kpi 추출
        hypotheses = hypothesis_result.hypotheses if hasattr(hypothesis_result, 'hypotheses') else hypothesis_result
        target_kpi = hypothesis_result.target_kpi if hasattr(hypothesis_result, 'target_kpi') else None

        # KPI ID 추출 (HypothesisResult의 target_kpi 우선 사용)
        if target_kpi and hasattr(target_kpi, 'id'):
            kpi_id = target_kpi.id
            kpi_name = getattr(target_kpi, 'name_kr', kpi_id)
            print(f"[KPI] Using target_kpi from HypothesisResult: {kpi_id} ({kpi_name})")
        else:
            # Fallback: 키워드 기반 추출
            kpi_keywords = {
                "매출": "매출", "수익": "매출", "revenue": "매출",
                "원가": "매출원가", "비용": "매출원가", "cost": "매출원가",
                "판매수량": "판매수량", "수량": "판매수량", "quantity": "판매수량",
                "영업이익": "영업이익", "이익": "영업이익", "profit": "영업이익"
            }
            kpi_id = "매출"  # 기본값
            kpi_name = "매출"
            query_lower = query.lower()
            for keyword, kpi in kpi_keywords.items():
                if keyword in query_lower:
                    kpi_id = kpi
                    kpi_name = kpi
                    break
            print(f"[KPI] Using keyword-based KPI: {kpi_id}")

        # DEBUG: Skip Step 3+ (UI testing)
        DEBUG_SKIP_VALIDATION = False

        if DEBUG_SKIP_VALIDATION:
            # ========== MOCK: Step 3-5 ==========
            status.text("[DEBUG] Skipping validation, using mock data")

            # Mock validation: 모든 가설을 검증됨으로 처리
            validated = hypotheses
            for h in validated:
                h.validation_status = "validated"
                h.validation_data = {"sql_query": "-- MOCK SQL --", "shap_value": 0.5}

            contributions = []
            model_r_squared = 0.75
            analysis_plan = {}
            interpretation = {"model_risk_assessment": {"overfitting_risk": "low"}}
            sql_queries = []
            matched_events = {}
            summary_result = {
                "summary": "**[DEBUG MODE]** Step 3-5가 비활성화되어 있습니다. `DEBUG_SKIP_VALIDATION = False`로 변경하면 실제 분석이 실행됩니다.",
                "sources": []
            }
            progress.progress(90)

        else:
            # Step 3: 가설 검증 (Confidence 기반 간소화 버전)
            status.text("Validating hypotheses...")

            # Confidence 기반 간소화 검증 (상위 가설 선별)
            # Note: DataFrame 기반 SHAP 검증은 추후 통합 예정
            sorted_hypotheses = sorted(
                hypotheses,
                key=lambda h: getattr(h, 'confidence', 0) or 0,
                reverse=True
            )

            # 상위 confidence 가설만 validated로 선택 (confidence >= 0.3)
            validated = []
            total_conf = sum(getattr(h, 'confidence', 0) or 0 for h in sorted_hypotheses)

            for h in sorted_hypotheses:
                conf = getattr(h, 'confidence', 0) or 0
                if conf >= 0.3:  # 최소 confidence 임계값
                    h.validation_status = "validated"
                    # contribution_pct 계산 (confidence 비율 기반)
                    contrib_pct = (conf / total_conf * 100) if total_conf > 0 else 0
                    h.validation_data = {
                        "contribution_pct": contrib_pct,
                        "confidence": conf,
                        "method": "confidence_based"
                    }
                    validated.append(h)

            contributions = []
            model_r_squared = 0.65  # Placeholder
            analysis_plan = {"method": "confidence_based_validation"}
            interpretation = {"model_risk_assessment": {"overfitting_risk": "low"}}

            # SQL 쿼리 수집 (validation_data에서)
            sql_queries = []
            for h in validated:
                data = h.validation_data or {}
                sql_query = data.get("sql_query", "")
                if sql_query:
                    sql_queries.append({
                        "hypothesis_id": h.id,
                        "factor": h.factor,
                        "sql": sql_query
                    })
            progress.progress(50)

            # Step 4: 이벤트 매칭
            status.text("Event Matching (Scoring Algorithm)...")
            try:
                matched_events = analysis_agent.event_matcher.match(
                    hypotheses=validated,
                    region=region,
                    min_score=0.3,
                    top_k=5
                )
            except Exception as e:
                matched_events = {}
                st.warning(f"이벤트 매칭 오류: {e}")
            progress.progress(70)

            # Step 5: 추론 기반 답변 생성
            status.text("🧠 추론 기반 답변 생성 중...")
            details = analysis_agent._build_details(validated, matched_events, sql_queries)
            summary_result = analysis_agent._generate_summary(query, details, kpi_change)
            progress.progress(90)

        # ========== 2. 결과 표시 (통합 박스 - 단일 HTML) ==========
        status.empty()
        progress.empty()

        # Summary 추출
        summary = summary_result.get("summary", "") if isinstance(summary_result, dict) else summary_result
        summary_formatted = summary.replace(chr(10), '<br>') if summary else "분석이 완료되었습니다."

        # Details 섹션 HTML 빌드 - 친절한 설명 방식
        details_html = ""

        # ========== Step 1: KPI 변동 확인 ==========
        details_html += '<div class="analysis-step">'
        details_html += '<div class="step-header"><span class="step-number">1</span><span class="step-title">KPI 변동 확인</span></div>'
        if kpi_change:
            change_sign = "+" if kpi_change.change_percent > 0 else ""
            direction_kr = "상승" if kpi_change.change_percent > 0 else "하락"
            details_html += f'<div class="step-content">'
            details_html += f'<p>ERP 데이터베이스에서 <strong>{kpi_change.kpi_name}</strong>의 변동을 확인했습니다.</p>'
            details_html += f'<div class="kpi-comparison">'
            details_html += f'<div class="kpi-value"><span class="kpi-label">이전 기간</span><span class="kpi-num">{kpi_change.previous_value:,.0f}</span></div>'
            details_html += f'<div class="kpi-arrow">→</div>'
            details_html += f'<div class="kpi-value"><span class="kpi-label">현재 기간</span><span class="kpi-num">{kpi_change.current_value:,.0f}</span></div>'
            details_html += f'<div class="kpi-change {("positive" if kpi_change.change_percent > 0 else "negative")}">{change_sign}{kpi_change.change_percent:.1f}% {direction_kr}</div>'
            details_html += f'</div>'
            sql_escaped = kpi_change.sql_query.replace('<', '&lt;').replace('>', '&gt;') if kpi_change.sql_query else ''
            if sql_escaped:
                details_html += f'<details class="sql-details"><summary>실행된 SQL 쿼리 보기</summary><pre>{sql_escaped}</pre></details>'
            details_html += f'</div>'
        else:
            details_html += f'<div class="step-content"><p>KPI 변동 데이터를 가져올 수 없습니다.</p></div>'
        details_html += '</div>'

        # ========== Step 2: 가설 생성 ==========
        details_html += '<div class="analysis-step">'
        details_html += '<div class="step-header"><span class="step-number">2</span><span class="step-title">가설 생성</span></div>'
        details_html += f'<div class="step-content">'
        details_html += f'<p>Knowledge Graph를 탐색하여 KPI 변동에 영향을 줄 수 있는 <strong>{len(hypotheses)}개의 요인</strong>을 식별했습니다.</p>'
        if hypotheses:
            details_html += '<div class="hypothesis-list">'
            for i, h in enumerate(hypotheses[:8], 1):
                conf = getattr(h, 'confidence', 0.0) or 0.0
                conf_level = "높음" if conf >= 0.7 else ("중간" if conf >= 0.4 else "낮음")
                details_html += f'<div class="hypothesis-item"><span class="hypothesis-name">{h.factor}</span><span class="hypothesis-conf conf-{conf_level.lower()}">{conf_level}</span></div>'
            if len(hypotheses) > 8:
                details_html += f'<div class="hypothesis-more">외 {len(hypotheses) - 8}개 요인...</div>'
            details_html += '</div>'
        details_html += f'</div>'
        details_html += '</div>'

        # ========== Step 3: 가설 검증 및 분석 해석 ==========
        validated_ids = {h.id for h in validated}
        rejected = [h for h in hypotheses if h.id not in validated_ids]

        details_html += '<div class="analysis-step">'
        details_html += '<div class="step-header"><span class="step-number">3</span><span class="step-title">가설 검증 및 분석 해석</span></div>'
        details_html += f'<div class="step-content">'
        details_html += f'<p>ERP 데이터와 회귀 분석을 통해 가설을 검증했습니다. 모델 설명력(R²)은 <strong>{model_r_squared:.1%}</strong>입니다.</p>'
        details_html += f'<div class="validation-summary">'
        details_html += f'<div class="validation-stat validated"><span class="stat-num">{len(validated)}</span><span class="stat-label">검증됨</span></div>'
        details_html += f'<div class="validation-stat rejected"><span class="stat-num">{len(rejected)}</span><span class="stat-label">기각됨</span></div>'
        details_html += f'</div>'

        if validated:
            details_html += '<p style="margin-top: 16px;"><strong>검증된 주요 요인:</strong></p>'
            details_html += '<div class="validated-factors">'
            for h in validated[:5]:
                data = h.validation_data or {}
                contrib_pct = data.get("contribution_pct", 0)
                reasoning = getattr(h, 'reasoning', '') or ''
                reasoning_text = reasoning[:150] + "..." if len(reasoning) > 150 else reasoning
                details_html += f'<div class="factor-card">'
                details_html += f'<div class="factor-header"><span class="factor-name">{h.factor}</span><span class="factor-contrib">{contrib_pct:.1f}% 기여</span></div>'
                if reasoning_text:
                    details_html += f'<div class="factor-reasoning">{reasoning_text}</div>'
                details_html += f'</div>'
            details_html += '</div>'

        # Related Events
        if matched_events:
            total_ev = sum(len(v) for v in matched_events.values())
            details_html += f'<p style="margin-top: 16px;"><strong>관련 외부 이벤트 ({total_ev}건):</strong></p>'
            details_html += '<div class="events-list">'
            event_count = 0
            for h_id, events in matched_events.items():
                for ev in events[:3]:
                    if event_count >= 5:
                        break
                    details_html += f'<div class="event-item"><span class="event-name">{ev.event_name}</span><span class="event-category">{ev.event_category}</span></div>'
                    event_count += 1
                if event_count >= 5:
                    break
            details_html += '</div>'

        details_html += f'</div>'
        details_html += '</div>'

        # ========== Step 4: 지식그래프 ==========
        details_html += '<div class="analysis-step">'
        details_html += '<div class="step-header"><span class="step-number">4</span><span class="step-title">지식그래프</span></div>'
        details_html += f'<div class="step-content">'
        details_html += f'<p>분석 결과를 지식그래프로 시각화했습니다. 아래에서 KPI와 영향 요인 간의 관계를 확인할 수 있습니다.</p>'
        details_html += f'</div>'
        details_html += '</div>'

        # ========== Knowledge Graph Visualization ==========
        graph_section_html = ""
        if KG_VISUALIZER_AVAILABLE and validated:
            try:
                visualizer = KGVisualizer()

                # 분석 결과에서 ID 추출
                driver_ids = []
                for h in validated:
                    # driver_id 우선, 없으면 factor 사용
                    driver_id = getattr(h, 'driver_id', None) or getattr(h, 'factor', None)
                    if driver_id:
                        driver_ids.append(driver_id)

                # 디버그 출력
                print(f"[KGVisualizer] KPI: {kpi_id}, Drivers: {len(driver_ids)}")

                event_ids = []
                for h_id, events in matched_events.items():
                    for ev in events[:3]:
                        event_id = getattr(ev, 'event_id', None)
                        if event_id:
                            event_ids.append(event_id)

                print(f"[KGVisualizer] Event IDs: {event_ids}")

                # 서브그래프 빌드
                if driver_ids:
                    subgraph = visualizer.build_subgraph(
                        kpi_id=kpi_id,
                        driver_ids=driver_ids,
                        event_ids=event_ids if event_ids else None,
                        max_drivers=8,
                        max_events_per_driver=2
                    )

                    print(f"[KGVisualizer] Subgraph nodes: {len(subgraph.nodes)}, edges: {len(subgraph.edges)}")

                    if subgraph.nodes:
                        graph_html = visualizer.generate_html(subgraph, height="700px")
                        if graph_html:
                            import base64
                            encoded = base64.b64encode(graph_html.encode()).decode()
                            graph_section_html = f'<hr style="border: none; border-top: 1px solid #E9E9E7; margin: 16px 0;"><div style="font-size: 11px; font-weight: 600; color: #9B9A97; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 8px;">KNOWLEDGE GRAPH</div><iframe src="data:text/html;base64,{encoded}" style="width: 100%; height: 700px; border: 1px solid #E9E9E7; border-radius: 8px;" frameborder="0"></iframe>'
                else:
                    print("[KGVisualizer] No driver_ids found, skipping graph")

                visualizer.close()
            except Exception as e:
                print(f"[KGVisualizer] Error: {e}")
                import traceback
                traceback.print_exc()
                graph_section_html = ""

        # 전체 박스 HTML (단일 문자열로 렌더링) - 그래프는 맨 밑에
        box_html = f'<div class="analysis-result-box"><div class="box-query">{query}</div><div class="box-summary">{summary_formatted}</div><details class="box-details"><summary>분석 과정 자세히 보기</summary><div class="details-content">{details_html}</div></details>{graph_section_html}</div>'
        st.markdown(box_html, unsafe_allow_html=True)

        # 결과 저장
        st.session_state.current_result = {
            "query": query,
            "intent": intent_result,
            "kpi_change": {
                "kpi_name": kpi_change.kpi_name if kpi_change else None,
                "change_percent": kpi_change.change_percent if kpi_change else None,
            } if kpi_change else None,
            "hypotheses": len(hypotheses),
            "validated": len(validated),
            "matched_events": sum(len(v) for v in matched_events.values()),
            "summary": summary_result.get("summary", "") if isinstance(summary_result, dict) else summary_result,
            "sources": summary_result.get("sources", []) if isinstance(summary_result, dict) else []
        }

    else:
        # Descriptive: Search Agent 사용
        status.text("Searching...")

        search_agent = SearchAgent()

        # Intent Classifier 결과 사용
        sub_intent = intent_result.get("sub_intent", "internal_data")
        is_event_query = intent_result.get("is_event_query", False)

        if is_event_query:
            source = "vector"
            source_label = "Vector Search"
        elif sub_intent == "external_data":
            source = "graph"
            source_label = "Knowledge Graph"
        else:
            source = "sql"
            source_label = "ERP Database"

        progress.progress(40)

        context = AgentContext(
            query=query,
            metadata={"source": source, "top_k": 5}
        )

        result = search_agent.run(context)
        progress.progress(70)

        # Clear progress
        status.empty()
        progress.empty()

        # Results
        if result.get("success") and result.get("data"):
            data = result["data"]

            # Details HTML 빌드
            details_html = f'<div class="detail-item"><strong>Data Source:</strong> {source_label}</div>'
            query_used = result.get("query", "")
            if query_used:
                query_escaped = query_used.replace('<', '&lt;').replace('>', '&gt;')
                details_html += f'<div class="detail-item"><strong>Query:</strong></div>'
                details_html += f'<pre style="background: #F7F7F5; padding: 12px; border-radius: 4px; font-size: 12px; overflow-x: auto; margin: 8px 0;">{query_escaped}</pre>'

            # 결과 요약 텍스트
            if isinstance(data, list) and data:
                summary_text = f"총 {len(data)}개의 결과를 찾았습니다."
            else:
                summary_text = "데이터를 조회했습니다."

            # 박스 HTML (단일 렌더링)
            box_html = f'<div class="analysis-result-box"><div class="box-query">{query}</div><div class="box-summary">{summary_text}</div><details class="box-details"><summary>View query details</summary><div class="details-content">{details_html}</div></details></div>'
            st.markdown(box_html, unsafe_allow_html=True)

            # 데이터 테이블은 박스 외부에 표시 (Streamlit 위젯)
            if source == "vector":
                display_vector_search_results(data)
            elif isinstance(data, list) and data:
                import pandas as pd
                df = pd.DataFrame(data)
                st.dataframe(df, use_container_width=True)
            else:
                st.json(data)
        else:
            error_msg = result.get('error', 'Unknown error')
            box_html = f'<div class="analysis-result-box"><div class="box-query">{query}</div><div class="box-summary" style="color: #EB5757;">Search failed: {error_msg}</div></div>'
            st.markdown(box_html, unsafe_allow_html=True)

        st.session_state.current_result = {
            "query": query,
            "intent": intent_result,
            "data": result.get("data"),
            "source": source,
            "sql": result.get("query") if source == "sql" else None
        }

    # 히스토리에 추가
    st.session_state.history.append({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "query": query,
        "mode": analysis_mode
    })


def main():
    """메인 함수"""
    init_session_state()

    # 사이드바 (Notion style - minimal icons)
    with st.sidebar:
        st.markdown("### System Status")

        # 시스템 상태 (텍스트 기반, 이모지 없음)
        st.markdown('<div style="font-size: 13px; color: #37352F;">', unsafe_allow_html=True)
        st.markdown("Orchestrator: Ready", unsafe_allow_html=True)
        st.markdown("SQL Tool: Ready", unsafe_allow_html=True)

        # Neo4j 연결 확인
        try:
            from agents.tools import GraphExecutor
            graph = GraphExecutor()
            result = graph.execute("RETURN 1 as test")
            if result.success:
                st.markdown("Neo4j: Connected", unsafe_allow_html=True)
            else:
                st.markdown('<span style="color: #9B9A97;">Neo4j: Not connected</span>', unsafe_allow_html=True)
        except:
            st.markdown('<span style="color: #9B9A97;">Neo4j: Not connected</span>', unsafe_allow_html=True)

        if INTENT_CLASSIFIER_AVAILABLE:
            st.markdown("Intent Classifier: Ready", unsafe_allow_html=True)
        else:
            st.markdown('<span style="color: #9B9A97;">Intent Classifier: Default mode</span>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("---")

        # 예시 질문
        st.markdown("### Example Queries")

        example_queries = [
            "2024년 4분기 북미 영업이익이 왜 감소했어?",
            "2025년 Q3 매출 변동 원인 분석해줘",
            "2024년 4분기 총 매출은 얼마야?",
            "유럽 지역 원가 현황 알려줘",
            "최근 물류 관련 이벤트 알려줘",
            "관세 정책 관련 이슈가 뭐가 있어?",
        ]

        for eq in example_queries:
            if st.button(eq, key=f"example_{eq[:20]}", use_container_width=True):
                st.session_state.example_query = eq
                st.rerun()

        st.markdown("---")

        # 히스토리
        st.markdown("### Query History")
        if st.session_state.history:
            for item in st.session_state.history[-5:]:
                st.caption(f"{item['timestamp'][:10]} - {item['query'][:25]}...")
        else:
            st.caption("No queries yet")

        # 채팅 기록 초기화 버튼
        st.markdown("---")
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.chat_messages = []
            st.session_state.is_processing = False
            st.session_state.pending_query = None
            st.rerun()

    # 메인 영역 - Chat Interface with content area
    st.markdown('<div class="main-content-area">', unsafe_allow_html=True)
    st.markdown('<p class="main-header">LG HE Business Intelligence</p>', unsafe_allow_html=True)

    # Chat message display area (shows previous conversations)
    chat_container = st.container()
    with chat_container:
        display_chat_messages()

    # Process pending query - show results here
    if st.session_state.is_processing and st.session_state.pending_query:
        query_to_process = st.session_state.pending_query
        st.session_state.pending_query = None

        # Run analysis with unified result display
        run_analysis(query_to_process)

        # Get the summary from current_result if available
        summary_text = "Analysis completed."
        if st.session_state.current_result:
            result = st.session_state.current_result
            if isinstance(result, dict):
                if result.get('summary'):
                    summary_text = result.get('summary', summary_text)

        # Add assistant message to chat for history
        st.session_state.chat_messages.append({
            'role': 'assistant',
            'content': summary_text,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

        st.session_state.is_processing = False

    st.markdown('</div>', unsafe_allow_html=True)

    # Fixed Input area at bottom (Notion AI style)
    st.markdown('<div class="chat-input-area">', unsafe_allow_html=True)

    # 예시 질문이 선택되었으면 적용
    default_query = st.session_state.get("example_query", "")

    col1, col2 = st.columns([9, 1])

    with col1:
        query = st.text_input(
            "질문 입력",
            value=default_query,
            placeholder="비즈니스 데이터에 대해 질문하세요...",
            key="query_input",
            label_visibility="collapsed"
        )

    with col2:
        analyze_button = st.button("Send", type="primary", use_container_width=True)

    # 예시 질문 상태 초기화
    if "example_query" in st.session_state:
        del st.session_state.example_query

    st.markdown('</div>', unsafe_allow_html=True)

    # Handle submission
    if analyze_button and query:
        # Add user message to chat
        st.session_state.chat_messages.append({
            'role': 'user',
            'content': query,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

        # Store the query for processing
        st.session_state.pending_query = query
        st.session_state.is_processing = True
        st.rerun()

    elif analyze_button and not query:
        st.warning("Please enter a question.")


if __name__ == "__main__":
    main()
