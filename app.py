import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import streamlit as st
import pandas as pd
import time
import datetime
import json
import io
import re
import base64

import google.generativeai as genai
import PyPDF2

import database as db

# Database is always available (local SQLite)
DB_OK = True

# -----------------------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------------------
st.set_page_config(
    page_title="Agentic AI Career Coach",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# -----------------------------------------------------------------
# GEMINI SETUP  — configure once at module level
# -----------------------------------------------------------------
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    GEMINI_OK = True
except Exception as _e:
    GEMINI_OK = False

def _model(model_name: str = None) -> genai.GenerativeModel:
    return genai.GenerativeModel(model_name or "gemini-2.5-flash-lite")

# Fallback model chain — try each once (NO retries to save quota)
_FALLBACK_MODELS = ["gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-2.0-flash"]

def _generate(prompt, **kwargs):
    """Generate content with model fallback. Each model tried ONCE to conserve daily quota."""
    for model_name in _FALLBACK_MODELS:
        try:
            model = _model(model_name)
            resp = model.generate_content(prompt, **kwargs)
            # Track usage in session state
            if "api_calls_today" not in st.session_state:
                st.session_state.api_calls_today = 0
            st.session_state.api_calls_today += 1
            return resp
        except Exception as e:
            err = str(e)
            # API key is dead — stop immediately
            if "API_KEY_INVALID" in err or "expired" in err.lower():
                raise Exception("❌ API key is invalid or expired. Generate a new one at https://aistudio.google.com/apikey") from e
            # Rate limited — try next model (no retry, no sleep)
            if "429" in err or "ResourceExhausted" in err or "quota" in err.lower():
                continue
            # Model doesn't exist — try next
            if "404" in err:
                continue
            # Unknown error — raise it
            raise e
    # All models exhausted
    raise Exception(
        "⏳ Daily free-tier limit reached (25 requests/day). "
        "The quota resets at midnight Pacific Time (~12:30 PM IST). "
        "Please try again later or upgrade to a paid plan at https://aistudio.google.com"
    )

def _safe_json(text: str) -> dict | list | None:
    """Strip markdown fences and parse JSON safely."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE)
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to extract first JSON object / array
        m = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                return None
        return None


# -----------------------------------------------------------------
# AI FUNCTIONS
# -----------------------------------------------------------------

def extract_text_from_file(uploaded_file) -> str:
    """Extract plain text from uploaded PDF or TXT. Returns empty string for images."""
    data = uploaded_file.read()
    uploaded_file.seek(0)  # reset for potential re-read
    ftype = uploaded_file.type or ""
    if "pdf" in ftype:
        try:
            reader = PyPDF2.PdfReader(io.BytesIO(data))
            text = "\n".join(p.extract_text() or "" for p in reader.pages)
            return text
        except Exception:
            return ""
    if "text" in ftype or uploaded_file.name.endswith(".txt"):
        return data.decode("utf-8", errors="ignore")
    # For images, return empty — we'll use multimodal
    return ""


def _get_file_bytes_and_mime(uploaded_file) -> tuple:
    """Read file bytes and determine MIME type for Gemini multimodal."""
    uploaded_file.seek(0)
    data = uploaded_file.read()
    uploaded_file.seek(0)
    name = uploaded_file.name.lower()
    ftype = uploaded_file.type or ""

    if "pdf" in ftype or name.endswith(".pdf"):
        return data, "application/pdf"
    elif "png" in ftype or name.endswith(".png"):
        return data, "image/png"
    elif "jpeg" in ftype or "jpg" in ftype or name.endswith((".jpg", ".jpeg")):
        return data, "image/jpeg"
    elif "text" in ftype or name.endswith(".txt"):
        return data, "text/plain"
    else:
        return data, "application/octet-stream"


_RESUME_PROMPT = """You are a senior technical recruiter at a top tech company.
Analyze the following resume thoroughly and return ONLY a JSON object — no markdown, no explanation.

Return exactly this JSON structure:
{
  "name": "candidate name or Unknown",
  "skills": ["Python", "React", ...],
  "experience_years": "0-1 / 1-3 / 3-5 / 5+",
  "education": "Degree and university if found",
  "strengths": [
    "Clear description of strength 1",
    "Clear description of strength 2",
    "Clear description of strength 3",
    "Clear description of strength 4"
  ],
  "weaknesses": [
    "Clear description of weakness 1",
    "Clear description of weakness 2",
    "Clear description of weakness 3",
    "Clear description of weakness 4"
  ],
  "missing_for_sde": ["AWS", "Docker", "System Design", ...],
  "overall_feedback": "3-4 sentence detailed assessment of the resume quality, presentation, and content.",
  "placement_score": 67,
  "target_roles": ["Software Engineer", "Backend Developer", ...],
  "ats_tips": [
    "Specific tip to improve ATS compatibility 1",
    "Specific tip to improve ATS compatibility 2",
    "Specific tip to improve ATS compatibility 3"
  ]
}"""


def ai_analyze_resume(uploaded_file, resume_text: str = "") -> dict | None:
    """Analyze resume using Gemini — supports text, PDF, and image multimodal."""
    file_bytes, mime = _get_file_bytes_and_mime(uploaded_file)
    is_image = mime.startswith("image/")
    is_pdf = "pdf" in mime

    try:
        if is_image or (is_pdf and len(resume_text.strip()) < 50):
            # Use multimodal: send the file directly to Gemini
            parts = [
                {"text": _RESUME_PROMPT + "\n\nAnalyze the resume in the attached file:"},
                {"inline_data": {"mime_type": mime, "data": base64.b64encode(file_bytes).decode()}}
            ]
            resp = _generate({"parts": parts})
        else:
            # Use text-based analysis
            text = resume_text[:6000] if resume_text else file_bytes.decode("utf-8", errors="ignore")[:6000]
            prompt = _RESUME_PROMPT + f"\n\nResume content:\n{text}"
            resp = _generate(prompt)

        return _safe_json(resp.text)
    except Exception as e:
        st.error(f"Resume AI error: {e}")
        return None


def ai_predict_placement(profile: dict) -> dict | None:
    """Predict placement readiness using Gemini."""
    prompt = f"""You are a placement prediction expert who has helped thousands of students get placed.
Analyze this student profile and return a placement readiness assessment.

Student Profile:
- CGPA / Score: {profile.get('cgpa', 'Not specified')}
- DSA Skill Level: {profile.get('dsa_level', 'Beginner')}
- Number of Projects: {profile.get('projects', 0)}
- Internship Experience: {profile.get('internships', 'None')}
- Mock Interviews Completed: {profile.get('mock_interviews', 0)}
- Resume Uploaded & Analyzed: {profile.get('has_resume', False)}
- Resume Score (if analyzed): {profile.get('resume_score', 'N/A')}
- Target Role: {profile.get('target_role', 'Software Engineer')}
- Target Companies: {profile.get('target_companies', 'Any')}

Return ONLY this JSON:
{{
  "score": 72,
  "grade": "B+",
  "verdict": "Above Average",
  "breakdown": {{
    "technical_skills": 75,
    "project_experience": 80,
    "interview_readiness": 60,
    "academic_performance": 70,
    "communication": 65
  }},
  "key_strengths": ["Strength point 1", "Strength point 2"],
  "critical_gaps": ["Gap 1", "Gap 2"],
  "action_items": [
    {{"action": "Specific action to take", "impact": "+5%", "priority": "High", "timeframe": "This week"}},
    {{"action": "Another action", "impact": "+3%", "priority": "Medium", "timeframe": "This month"}}
  ],
  "summary": "2-3 sentence personalized insights for this specific student.",
  "company_match": {{
    "FAANG": 35,
    "Mid-tier": 68,
    "Startups": 82
  }}
}}"""
    try:
        resp = _generate(prompt)
        return _safe_json(resp.text)
    except Exception as e:
        st.error(f"Prediction AI error: {e}")
        return None


def ai_mentor_reply(chat_history: list) -> str:
    """Get a real Gemini response for the mentor chat."""
    SYSTEM = (
        "You are an expert AI Career Coach for students preparing for software engineering placements. "
        "You specialize in DSA, system design, behavioral interviews, resume building, and career strategy. "
        "Be concise, practical, and encouraging. Use a conversational tone. "
        "Use bullet points and line breaks for readability. Keep responses under 200 words unless deep explanation is needed."
    )
    contents = [
        {"role": "user",  "parts": [{"text": f"[System]: {SYSTEM}"}]},
        {"role": "model", "parts": [{"text": "Understood! I'm your AI Career Coach 🚀 — ready to help with interview prep, DSA, resume, and career strategy. What would you like to work on?"}]},
    ]
    for msg in chat_history[1:]:
        role = "user" if msg["role"] == "user" else "model"
        contents.append({"role": role, "parts": [{"text": msg["content"]}]})
    try:
        resp = _generate(contents)
        return resp.text
    except Exception as e:
        return f"⚠️ Error: {e}. Please try again."


def ai_generate_question(role: str, topic: str, difficulty: str) -> dict | None:
    """Ask Gemini to generate a fresh interview question."""
    prompt = f"""You are an expert technical interviewer at a top tech company.
Generate ONE {difficulty}-level interview question for a {role} candidate about {topic}.

Return ONLY this JSON:
{{
  "question": "The complete interview question",
  "type": "Technical / Behavioral / System Design",
  "what_it_tests": "Brief note on what competency this question assesses",
  "hints": ["Hint 1 if they get stuck", "Hint 2"]
}}"""
    try:
        resp = _generate(prompt)
        return _safe_json(resp.text)
    except Exception as e:
        st.error(f"Question generation error: {e}")
        return None


def ai_evaluate_answer(question: str, answer: str, q_type: str) -> dict | None:
    """Evaluate a mock interview answer with Gemini."""
    prompt = f"""You are an expert technical interviewer. Evaluate this candidate's answer fairly and constructively.

Question ({q_type}): {question}

Candidate's Answer: {answer}

Return ONLY this JSON (scores are 0-100 integers):
{{
  "correctness": 85,
  "clarity": 70,
  "depth": 75,
  "overall": 77,
  "feedback": "Detailed, specific, constructive feedback paragraph (3-4 sentences)",
  "what_was_good": ["Specific good point 1", "Specific good point 2"],
  "what_to_improve": ["Specific improvement point 1", "Specific improvement point 2"],
  "ideal_answer_hint": "Brief hint about key elements of an ideal answer"
}}"""
    try:
        resp = _generate(prompt)
        return _safe_json(resp.text)
    except Exception as e:
        st.error(f"Evaluation AI error: {e}")
        return None


def ai_generate_learning_plan(weak_areas: list, target_role: str, days: int = 7) -> dict | None:
    """Generate a personalized learning plan with Gemini."""
    weak_str = ", ".join(weak_areas) if weak_areas else "DSA, System Design"
    prompt = f"""You are a top coding bootcamp instructor. Create a focused {days}-day learning sprint.

Student's weak areas: {weak_str}
Target role: {target_role}

Return ONLY this JSON:
{{
  "weekly_goal": "One sentence goal for this sprint",
  "success_metric": "How to measure success at the end of {days} days",
  "plan": [
    {{
      "day": "Day 1",
      "focus": "Topic or skill name",
      "difficulty": "Easy",
      "tasks": ["Specific task 1", "Specific task 2"],
      "resource": "Book chapter, LeetCode tag, or YouTube channel"
    }}
  ]
}}
Include all {days} days. Make tasks specific and actionable (e.g. 'Solve LC #104 Maximum Depth of Binary Tree').
Difficulty should be Easy/Medium/Hard."""
    try:
        resp = _generate(prompt)
        return _safe_json(resp.text)
    except Exception as e:
        st.error(f"Learning plan AI error: {e}")
        return None


# -----------------------------------------------------------------
# CUSTOM CSS — Premium Dark Theme
# -----------------------------------------------------------------
def load_css():
    st.markdown("""
    <style>
    /* ——— Google Font ——— */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    /* ——— Root variables ——— */
    :root {
        --bg: #0f172a;
        --card: #111827;
        --card-hover: #1e293b;
        --accent: #6366f1;
        --accent-light: #818cf8;
        --text: #f1f5f9;
        --text-dim: #94a3b8;
        --success: #22c55e;
        --warning: #f59e0b;
        --danger: #ef4444;
        --radius: 16px;
        --radius-sm: 10px;
    }

    /* ——— Global ——— */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2rem !important;
        max-width: 1200px !important;
    }

    /* ——— Hide Streamlit branding ——— */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* ——— Tab styling ——— */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: var(--card);
        border-radius: var(--radius);
        padding: 6px;
        border: 1px solid rgba(255,255,255,0.06);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: var(--radius-sm);
        padding: 10px 18px;
        font-weight: 500;
        font-size: 0.85rem;
        color: var(--text-dim);
        background: transparent;
        border: none;
        transition: all 0.2s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: var(--text);
        background: rgba(99,102,241,0.1);
    }
    .stTabs [aria-selected="true"] {
        background: var(--accent) !important;
        color: white !important;
        font-weight: 600;
        box-shadow: 0 4px 12px rgba(99,102,241,0.35);
        transform: scale(1.05);
    }
    .stTabs [data-baseweb="tab-highlight"] { display: none; }
    .stTabs [data-baseweb="tab-border"]    { display: none; }

    /* ——— Tab content fade-in ——— */
    .stTabs [data-baseweb="tab-panel"] {
        animation: fadeInUp 0.35s ease-out;
    }
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(12px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    /* ——— Card component ——— */
    .card {
        background: var(--card);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: var(--radius);
        padding: 24px;
        margin-bottom: 16px;
        box-shadow: 0 4px 24px rgba(0,0,0,0.2);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }

    /* ——— Metric card ——— */
    .metric-card {
        background: var(--card);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: var(--radius);
        padding: 20px 24px;
        text-align: center;
        box-shadow: 0 4px 24px rgba(0,0,0,0.2);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        border-color: var(--accent);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 800;
        background: linear-gradient(135deg, var(--accent), var(--accent-light));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 8px 0 4px;
    }
    .metric-label {
        font-size: 0.8rem;
        color: var(--text-dim);
        text-transform: uppercase;
        letter-spacing: 1.2px;
        font-weight: 600;
    }
    .metric-delta {
        font-size: 0.78rem;
        color: var(--success);
        margin-top: 4px;
    }

    /* ——— AI Insight box ——— */
    .ai-insight {
        background: linear-gradient(135deg, rgba(99,102,241,0.15), rgba(139,92,246,0.1));
        border: 1px solid rgba(99,102,241,0.25);
        border-left: 4px solid var(--accent);
        border-radius: var(--radius);
        padding: 24px 28px;
        margin: 20px 0;
        box-shadow: 0 4px 20px rgba(99,102,241,0.1);
    }
    .ai-insight h4 {
        margin: 0 0 8px 0;
        color: var(--accent-light);
        font-size: 0.9rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .ai-insight p {
        margin: 0;
        font-size: 1.05rem;
        color: var(--text);
        line-height: 1.6;
    }

    /* ——— Smart alert ——— */
    .smart-alert {
        background: linear-gradient(135deg, rgba(245,158,11,0.12), rgba(239,68,68,0.08));
        border: 1px solid rgba(245,158,11,0.25);
        border-left: 4px solid var(--warning);
        border-radius: var(--radius);
        padding: 16px 22px;
        margin-bottom: 20px;
        font-size: 0.95rem;
        color: var(--text);
    }
    .success-alert {
        background: linear-gradient(135deg, rgba(34,197,94,0.12), rgba(34,197,94,0.04));
        border: 1px solid rgba(34,197,94,0.25);
        border-left: 4px solid var(--success);
        border-radius: var(--radius);
        padding: 16px 22px;
        margin-bottom: 20px;
        font-size: 0.95rem;
        color: var(--text);
    }

    /* ——— Hero section ——— */
    .hero {
        text-align: center;
        padding: 36px 20px 28px;
    }
    .hero h1 {
        font-size: 3rem;
        font-weight: 900;
        margin: 0;
        background: linear-gradient(135deg, #fff, var(--accent-light));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -1px;
    }
    .hero p {
        font-size: 1.15rem;
        color: var(--text-dim);
        margin: 12px 0 0;
        font-weight: 400;
    }

    /* ——— AI Badge ——— */
    .ai-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: linear-gradient(135deg, rgba(99,102,241,0.2), rgba(139,92,246,0.15));
        border: 1px solid rgba(99,102,241,0.35);
        color: var(--accent-light);
        padding: 5px 14px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }

    /* ——— Skill tag ——— */
    .skill-tag {
        display: inline-block;
        background: linear-gradient(135deg, rgba(99,102,241,0.2), rgba(99,102,241,0.08));
        border: 1px solid rgba(99,102,241,0.3);
        color: var(--accent-light);
        padding: 6px 14px;
        border-radius: 20px;
        margin: 4px;
        font-size: 0.82rem;
        font-weight: 600;
        letter-spacing: 0.3px;
        transition: all 0.2s ease;
    }
    .skill-tag:hover {
        background: rgba(99,102,241,0.3);
        transform: scale(1.05);
    }

    /* ——— Day card for roadmap ——— */
    .day-card {
        background: var(--card);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: var(--radius);
        padding: 20px 24px;
        margin-bottom: 12px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.15);
        transition: border-color 0.2s ease;
    }
    .day-card:hover { border-color: var(--accent); }
    .day-card .day-title {
        font-size: 1rem;
        font-weight: 700;
        color: var(--text);
        margin-bottom: 4px;
    }
    .day-card .day-focus {
        font-size: 0.85rem;
        color: var(--text-dim);
        margin-bottom: 10px;
    }

    /* Difficulty badges */
    .badge-easy   { background:rgba(34,197,94,0.15);  color:#22c55e; padding:3px 10px; border-radius:8px; font-size:0.72rem; font-weight:700; text-transform:uppercase; }
    .badge-medium { background:rgba(245,158,11,0.15); color:#f59e0b; padding:3px 10px; border-radius:8px; font-size:0.72rem; font-weight:700; text-transform:uppercase; }
    .badge-hard   { background:rgba(239,68,68,0.15);  color:#ef4444; padding:3px 10px; border-radius:8px; font-size:0.72rem; font-weight:700; text-transform:uppercase; }

    /* Priority badges */
    .badge-priority-high   { background:rgba(239,68,68,0.15);  color:#ef4444; padding:2px 8px; border-radius:6px; font-size:0.7rem; font-weight:700; }
    .badge-priority-medium { background:rgba(245,158,11,0.15); color:#f59e0b; padding:2px 8px; border-radius:6px; font-size:0.7rem; font-weight:700; }
    .badge-priority-low    { background:rgba(34,197,94,0.15);  color:#22c55e; padding:2px 8px; border-radius:6px; font-size:0.7rem; font-weight:700; }

    /* ——— Chat bubbles ——— */
    .chat-user {
        background: var(--accent);
        color: white;
        padding: 14px 18px;
        border-radius: 18px 18px 4px 18px;
        margin: 8px 0;
        max-width: 80%;
        margin-left: auto;
        font-size: 0.92rem;
        line-height: 1.5;
        box-shadow: 0 2px 12px rgba(99,102,241,0.25);
    }
    .chat-ai {
        background: var(--card);
        border: 1px solid rgba(255,255,255,0.08);
        color: var(--text);
        padding: 14px 18px;
        border-radius: 18px 18px 18px 4px;
        margin: 8px 0;
        max-width: 80%;
        font-size: 0.92rem;
        line-height: 1.5;
        box-shadow: 0 2px 12px rgba(0,0,0,0.15);
        white-space: pre-wrap;
    }

    /* ——— Eval card ——— */
    .eval-card {
        background: var(--card);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: var(--radius);
        padding: 24px;
        margin-top: 16px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.2);
    }

    /* ——— Section heading ——— */
    .section-heading {
        font-size: 1.6rem;
        font-weight: 800;
        color: var(--text);
        margin-bottom: 6px;
        letter-spacing: -0.5px;
    }
    .section-sub {
        font-size: 0.95rem;
        color: var(--text-dim);
        margin-bottom: 24px;
    }

    /* ——— Progress bar override ——— */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, var(--accent), var(--accent-light)) !important;
        border-radius: 10px;
    }
    .stProgress > div > div > div {
        background: rgba(255,255,255,0.06) !important;
        border-radius: 10px;
    }

    /* ——— Streamlit native metric override ——— */
    div[data-testid="stMetric"] {
        background: var(--card);
        border: 1px solid rgba(255,255,255,0.06);
        padding: 16px;
        border-radius: var(--radius);
        box-shadow: 0 4px 16px rgba(0,0,0,0.15);
    }

    /* ——— Big score ——— */
    .big-score {
        font-size: 5rem;
        font-weight: 900;
        background: linear-gradient(135deg, var(--accent), var(--accent-light), #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin: 16px 0 8px;
        line-height: 1;
    }
    .big-score-label {
        text-align: center;
        color: var(--text-dim);
        font-size: 1rem;
        font-weight: 500;
        margin-bottom: 24px;
    }

    /* ——— Strength / Weakness cards ——— */
    .strength-card {
        background: linear-gradient(135deg, rgba(34,197,94,0.1), rgba(34,197,94,0.03));
        border: 1px solid rgba(34,197,94,0.2);
        border-radius: var(--radius);
        padding: 20px;
    }
    .weakness-card {
        background: linear-gradient(135deg, rgba(239,68,68,0.1), rgba(239,68,68,0.03));
        border: 1px solid rgba(239,68,68,0.2);
        border-radius: var(--radius);
        padding: 20px;
    }

    /* ——— Button override ——— */
    .stButton > button[kind="primary"],
    .stButton > button[data-testid="stBaseButton-primary"] {
        background: linear-gradient(135deg, var(--accent), #7c3aed) !important;
        border: none !important;
        border-radius: var(--radius-sm) !important;
        padding: 10px 28px !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        letter-spacing: 0.3px !important;
        box-shadow: 0 4px 14px rgba(99,102,241,0.35) !important;
        transition: all 0.25s ease !important;
    }
    .stButton > button[kind="primary"]:hover,
    .stButton > button[data-testid="stBaseButton-primary"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(99,102,241,0.5) !important;
    }

    /* ——— File uploader ——— */
    [data-testid="stFileUploader"] {
        background: var(--card);
        border: 2px dashed rgba(99,102,241,0.3);
        border-radius: var(--radius);
        padding: 24px;
    }

    /* ——— Text area ——— */
    .stTextArea textarea {
        background: var(--card) !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: var(--radius-sm) !important;
        color: var(--text) !important;
        font-family: 'Inter', sans-serif !important;
    }
    .stTextArea textarea:focus {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 2px rgba(99,102,241,0.2) !important;
    }

    /* ——— Checkbox override ——— */
    .stCheckbox label span { font-size: 0.9rem !important; }

    /* ——— Weak area highlight ——— */
    .weak-area {
        background: linear-gradient(135deg, rgba(239,68,68,0.1), rgba(239,68,68,0.03));
        border: 1px solid rgba(239,68,68,0.15);
        border-left: 3px solid var(--danger);
        border-radius: var(--radius-sm);
        padding: 14px 18px;
        margin: 6px 0;
        font-size: 0.9rem;
        color: var(--text);
    }

    /* ——— Fix expander ——— */
    .streamlit-expanderHeader {
        font-weight: 600 !important;
        font-size: 0.95rem !important;
    }

    /* ——— Divider ——— */
    .divider {
        height: 1px;
        background: rgba(255,255,255,0.06);
        margin: 24px 0;
    }

    /* ——— Score ring helper ——— */
    .score-ring {
        display: inline-block;
        width: 80px; height: 80px;
        border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        font-size: 1.5rem; font-weight: 900;
        background: conic-gradient(var(--accent) var(--pct), rgba(255,255,255,0.06) 0);
        color: var(--text);
        margin: 0 auto;
    }

    /* ——— Action item card ——— */
    .action-item {
        background: var(--card);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: var(--radius-sm);
        padding: 14px 18px;
        margin: 8px 0;
        display: flex;
        align-items: flex-start;
        gap: 12px;
        transition: border-color 0.2s;
    }
    .action-item:hover { border-color: var(--accent); }

    /* ——— Company match bar ——— */
    .company-bar {
        background: rgba(255,255,255,0.06);
        border-radius: 8px;
        height: 10px;
        margin: 4px 0 12px;
        overflow: hidden;
    }
    .company-bar-fill {
        height: 100%;
        border-radius: 8px;
        background: linear-gradient(90deg, var(--accent), var(--accent-light));
    }
    </style>
    """, unsafe_allow_html=True)


# -----------------------------------------------------------------
# SESSION STATE INIT
# -----------------------------------------------------------------
def init_state():
    defaults = {
        # Auth
        "authenticated": False,
        "user_id": None,
        "username": None,
        "full_name": None,
        "show_page": "landing",
        # Chat
        "chat_history": [{"role": "ai", "content": "Hello! I'm your AI Career Coach 🚀\n\nI can help you with:\n• Interview prep & mock sessions\n• DSA strategies & problem patterns\n• Resume feedback & career planning\n• System design concepts\n\nWhat would you like to work on today?"}],
        # Resume
        "resume_analyzed": False,
        "resume_text": "",
        "resume_analysis": None,
        # Placement
        "placement_data": None,
        "placement_form_done": False,
        "placement_profile": {},
        # Interview
        "interview_started": False,
        "interview_question": None,
        "interview_feedback": None,
        # Learning
        "learning_plan": None,
        "learning_profile": {},
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def load_user_data():
    """Load user-specific data from database after login."""
    if not DB_OK:
        return
    user_id = st.session_state.user_id
    try:
        saved_chats = db.get_chat_history(user_id)
        if saved_chats:
            st.session_state.chat_history = [
                {"role": msg["role"], "content": msg["content"]} for msg in saved_chats
            ]
        else:
            welcome_msg = f"Hello {st.session_state.full_name or 'there'}! I'm your AI Career Coach. I can help with interview prep, resume advice, DSA strategies, and career planning. What would you like to work on today?"
            st.session_state.chat_history = [{"role": "ai", "content": welcome_msg}]
            db.save_chat_message(user_id, "ai", welcome_msg)
    except Exception:
        pass


# -----------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------
def card(content: str, extra_class: str = ""):
    st.markdown(f'<div class="card {extra_class}">{content}</div>', unsafe_allow_html=True)

def metric_card(label: str, value: str, delta: str = ""):
    delta_html = f'<div class="metric-delta">{delta}</div>' if delta else ""
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            {delta_html}
        </div>""", unsafe_allow_html=True)

def score_color(score: int) -> str:
    if score >= 75: return "#22c55e"
    if score >= 55: return "#f59e0b"
    return "#ef4444"

def score_verdict(score: int) -> str:
    if score >= 85: return "🏆 Excellent"
    if score >= 70: return "✅ Good"
    if score >= 55: return "⚡ Average"
    return "⚠️ Needs Work"


# -----------------------------------------------------------------
# 1. 🏠 DASHBOARD
# -----------------------------------------------------------------
def render_dashboard():
    st.markdown("""
        <div class="hero">
            <h1>Your AI Career Coach</h1>
            <p>Personalized guidance · Real analysis · Real results</p>
        </div>""", unsafe_allow_html=True)

    # Dynamic alerts based on session state
    has_resume   = st.session_state.resume_analyzed
    has_predict  = st.session_state.placement_data is not None
    has_learning = st.session_state.learning_plan  is not None

    if not has_resume:
        st.markdown('<div class="smart-alert">⚡ <strong>Get started:</strong> Upload your resume in the <strong>📄 Resume</strong> tab to unlock personalized AI analysis.</div>', unsafe_allow_html=True)

    # Quota usage indicator
    calls_used = st.session_state.get("api_calls_today", 0)
    if calls_used >= 20:
        st.markdown(f'<div style="background:rgba(239,68,68,0.12);border:1px solid rgba(239,68,68,0.3);border-radius:12px;padding:12px 18px;margin-bottom:16px;color:#f87171;font-size:0.9rem">⚠️ <strong>Daily AI quota nearly exhausted</strong> — {calls_used} calls used this session. Free tier allows ~25/day. Resets at ~12:30 PM IST.</div>', unsafe_allow_html=True)
    elif calls_used >= 10:
        st.markdown(f'<div style="background:rgba(245,158,11,0.1);border:1px solid rgba(245,158,11,0.25);border-radius:12px;padding:12px 18px;margin-bottom:16px;color:#fbbf24;font-size:0.9rem">📊 <strong>API Usage:</strong> {calls_used} calls this session (free limit: ~25/day)</div>', unsafe_allow_html=True)

    if has_resume and not has_predict:
        st.markdown('<div class="smart-alert">📊 <strong>Next step:</strong> Go to <strong>📊 Predictor</strong> to see your placement readiness score.</div>', unsafe_allow_html=True)

    # Metrics row
    analysis = st.session_state.resume_analysis
    pred     = st.session_state.placement_data

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        score = pred["score"] if pred else "—"
        delta = f"Target: 80%" if pred else "Run Predictor →"
        metric_card("Placement Score", f"{score}%" if pred else score, delta)
    with c2:
        roles = ", ".join(analysis["target_roles"][:1]) if analysis else "—"
        metric_card("Target Role", roles if analysis else "Upload Resume →", "AI detected" if analysis else "")
    with c3:
        skills_count = len(analysis["skills"]) if analysis else "—"
        metric_card("Skills Found", str(skills_count) if analysis else "—", "from resume" if analysis else "")
    with c4:
        grade = pred.get("grade", "—") if pred else "—"
        verdict = pred.get("verdict", "") if pred else "Run Predictor"
        metric_card("AI Grade", grade, verdict)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown('<div class="section-heading">Placement Readiness</div>', unsafe_allow_html=True)
        if pred:
            pct = pred["score"] / 100
            st.progress(pct)
            st.caption(f"{pred['score']} / 100 — {pred.get('verdict', '')}")

            # Breakdown
            breakdown = pred.get("breakdown", {})
            for skill, val in breakdown.items():
                label = skill.replace("_", " ").title()
                st.markdown(f'<div style="display:flex;justify-content:space-between;color:#94a3b8;font-size:0.85rem;margin-bottom:2px"><span>{label}</span><span style="color:#f1f5f9;font-weight:600">{val}%</span></div>', unsafe_allow_html=True)
                st.progress(val / 100)
        else:
            st.progress(0.0)
            st.caption("Complete the Predictor to see real scores")

    with col_right:
        if pred and pred.get("summary"):
            st.markdown(f"""
                <div class="ai-insight">
                    <h4>🧠 AI Insight</h4>
                    <p>{pred['summary']}</p>
                </div>""", unsafe_allow_html=True)
        elif analysis and analysis.get("overall_feedback"):
            st.markdown(f"""
                <div class="ai-insight">
                    <h4>📄 Resume Insight</h4>
                    <p>{analysis['overall_feedback']}</p>
                </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
                <div class="ai-insight">
                    <h4>🧠 AI Insight</h4>
                    <p>Start by uploading your resume. The AI will analyze it and provide personalized insights, placement prediction, and a custom learning roadmap.</p>
                </div>""", unsafe_allow_html=True)

    # Quick links
    if has_resume or has_predict:
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-heading" style="font-size:1.2rem">Quick Stats</div>', unsafe_allow_html=True)
        if analysis:
            tags = "".join([f'<span class="skill-tag">{s}</span>' for s in analysis.get("skills", [])[:12]])
            st.markdown(f'<div style="margin-bottom:8px"><strong style="color:#94a3b8;font-size:0.8rem">DETECTED SKILLS</strong><br>{tags}</div>', unsafe_allow_html=True)


# -----------------------------------------------------------------
# 2. 📄 RESUME ANALYZER
# -----------------------------------------------------------------
def render_resume_analyzer():
    # Header
    st.markdown("""
        <div style="text-align:center;padding:20px 0 10px">
            <div style="font-size:2.8rem;margin-bottom:6px">📄</div>
            <div class="section-heading" style="font-size:2rem;text-align:center">Resume Analyzer</div>
            <div class="section-sub" style="text-align:center;max-width:600px;margin:0 auto">
                Upload your resume in any format — our AI reads PDFs, images, and text files to give you a comprehensive analysis
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Supported formats display
    if not st.session_state.resume_analyzed:
        st.markdown("""
            <div style="display:flex;justify-content:center;gap:8px;flex-wrap:wrap;margin-bottom:16px">
                <span style="background:rgba(239,68,68,0.12);color:#f87171;padding:5px 12px;border-radius:8px;font-size:0.78rem;font-weight:700">.PDF</span>
                <span style="background:rgba(34,197,94,0.12);color:#4ade80;padding:5px 12px;border-radius:8px;font-size:0.78rem;font-weight:700">.TXT</span>
                <span style="background:rgba(59,130,246,0.12);color:#60a5fa;padding:5px 12px;border-radius:8px;font-size:0.78rem;font-weight:700">.PNG</span>
                <span style="background:rgba(168,85,247,0.12);color:#c084fc;padding:5px 12px;border-radius:8px;font-size:0.78rem;font-weight:700">.JPG</span>
                <span style="background:rgba(245,158,11,0.12);color:#fbbf24;padding:5px 12px;border-radius:8px;font-size:0.78rem;font-weight:700">.JPEG</span>
            </div>
        """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload your resume",
        type=["pdf", "txt", "png", "jpg", "jpeg"],
    )

    if not st.session_state.resume_analyzed:
        # Feature highlights
        st.markdown("")
        f1, f2, f3 = st.columns(3)
        with f1:
            st.markdown("""
                <div class="card" style="text-align:center;padding:20px">
                    <div style="font-size:1.6rem;margin-bottom:8px">🔍</div>
                    <div style="color:#f1f5f9;font-weight:600;font-size:0.9rem;margin-bottom:4px">Skill Extraction</div>
                    <div style="color:#94a3b8;font-size:0.8rem">Detects technical & soft skills automatically</div>
                </div>""", unsafe_allow_html=True)
        with f2:
            st.markdown("""
                <div class="card" style="text-align:center;padding:20px">
                    <div style="font-size:1.6rem;margin-bottom:8px">📊</div>
                    <div style="color:#f1f5f9;font-weight:600;font-size:0.9rem;margin-bottom:4px">ATS Scoring</div>
                    <div style="color:#94a3b8;font-size:0.8rem">Rates your resume against industry standards</div>
                </div>""", unsafe_allow_html=True)
        with f3:
            st.markdown("""
                <div class="card" style="text-align:center;padding:20px">
                    <div style="font-size:1.6rem;margin-bottom:8px">🎯</div>
                    <div style="color:#f1f5f9;font-weight:600;font-size:0.9rem;margin-bottom:4px">Gap Analysis</div>
                    <div style="color:#94a3b8;font-size:0.8rem">Spots missing skills for your target role</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("")

    # Buttons — centered
    _, col_btn, _ = st.columns([1, 2, 1])
    with col_btn:
        analyze_btn = st.button("🚀  Analyze with AI", type="primary", disabled=not GEMINI_OK, use_container_width=True)
    if st.session_state.resume_analyzed:
        _, col_reset, _ = st.columns([1, 2, 1])
        with col_reset:
            if st.button("🔄 Upload New Resume", use_container_width=True):
                st.session_state.resume_analyzed = False
                st.session_state.resume_analysis = None
                st.session_state.resume_text = ""
                st.rerun()

    if not GEMINI_OK:
        st.error("⚠️ Gemini API not configured.")
        return

    if analyze_btn:
        if uploaded_file is None:
            st.warning("📎 Please upload a file first.")
        else:
            # Step 1: Read the file
            with st.spinner("📖 Reading your resume…"):
                text = extract_text_from_file(uploaded_file)
                st.session_state.resume_text = text

            # Step 2: Analyze with Gemini (multimodal for images/unreadable PDFs)
            fname = uploaded_file.name.lower()
            is_image = fname.endswith((".png", ".jpg", ".jpeg"))

            if is_image:
                with st.spinner("🤖 AI is reading your resume image…"):
                    analysis = ai_analyze_resume(uploaded_file, resume_text="")
            elif len(text.strip()) < 50:
                with st.spinner("🤖 Text extraction limited — using AI Vision…"):
                    analysis = ai_analyze_resume(uploaded_file, resume_text=text)
            else:
                with st.spinner("🤖 AI is analyzing your resume…"):
                    analysis = ai_analyze_resume(uploaded_file, resume_text=text)

            if analysis:
                st.session_state.resume_analysis = analysis
                st.session_state.resume_analyzed = True
                st.rerun()
            else:
                st.error("❌ Analysis failed. Please try again or upload in a different format.")

    # ==================== RESULTS SECTION ====================
    if st.session_state.resume_analyzed and st.session_state.resume_analysis:
        a = st.session_state.resume_analysis

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        # ---- Score Hero Section ----
        score = a.get("placement_score", 0)
        color = score_color(score)
        verdict = score_verdict(score)
        name = a.get("name", "Unknown")
        name_display = name if name != "Unknown" else ""

        name_block = ""
        if name_display:
            name_block = f'<div style="color:#94a3b8;font-size:0.85rem;font-weight:600;text-transform:uppercase;letter-spacing:1px;margin-bottom:4px">Resume Analysis For</div><div style="color:#f1f5f9;font-size:1.5rem;font-weight:800;margin-bottom:16px">{name_display}</div>'

        edu_block = ""
        if a.get("education"):
            edu_block = f'<div style="color:#94a3b8;font-size:0.85rem"><span style="margin-right:4px">🎓</span>{a["education"]}</div>'

        exp_block = ""
        if a.get("experience_years"):
            exp_block = f'<div style="color:#94a3b8;font-size:0.85rem"><span style="margin-right:4px">💼</span>{a["experience_years"]} years</div>'

        st.markdown(f"""
            <div style="background:linear-gradient(135deg,rgba(99,102,241,0.12),rgba(139,92,246,0.08));
                        border:1px solid rgba(99,102,241,0.2);border-radius:20px;padding:32px;text-align:center;margin-bottom:24px">
                {name_block}
                <div style="font-size:5rem;font-weight:900;background:linear-gradient(135deg,{color},{color}aa);
                            -webkit-background-clip:text;-webkit-text-fill-color:transparent;line-height:1;margin-bottom:4px">{score}<span style="font-size:2rem">/100</span></div>
                <div style="color:{color};font-size:1.1rem;font-weight:700;margin-bottom:16px">{verdict}</div>
                <div style="display:flex;justify-content:center;gap:20px;flex-wrap:wrap;">{edu_block}{exp_block}</div>
            </div>
        """, unsafe_allow_html=True)

        # ---- AI Assessment ----
        st.markdown(f"""
            <div class="ai-insight">
                <h4>🤖 AI Assessment</h4>
                <p>{a.get('overall_feedback', 'No feedback available.')}</p>
            </div>""", unsafe_allow_html=True)

        # ---- Target Roles ----
        if a.get("target_roles"):
            roles_html = "".join([f'<span class="skill-tag" style="background:rgba(34,197,94,0.1);border-color:rgba(34,197,94,0.3);color:#22c55e">{r}</span>' for r in a["target_roles"]])
            st.markdown(f"""
                <div style="margin:16px 0">
                    <span style="color:#94a3b8;font-size:0.8rem;font-weight:600;text-transform:uppercase;letter-spacing:0.5px">Best-Fit Roles:&nbsp;</span>
                    {roles_html}
                </div>""", unsafe_allow_html=True)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        # ---- Skills Cloud ----
        st.markdown('<div class="section-heading" style="font-size:1.3rem">🛠️ Detected Skills</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub" style="margin-bottom:12px">Skills extracted from your resume by AI</div>', unsafe_allow_html=True)
        skills = a.get("skills", [])
        if skills:
            tags = "".join([f'<span class="skill-tag">{s}</span>' for s in skills])
            st.markdown(f'<div style="margin-bottom:8px">{tags}</div>', unsafe_allow_html=True)
            st.markdown(f'<div style="color:#94a3b8;font-size:0.82rem">{len(skills)} skills detected</div>', unsafe_allow_html=True)
        else:
            st.markdown('<span style="color:#94a3b8">No skills detected. Try uploading a clearer version.</span>', unsafe_allow_html=True)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        # ---- Strengths & Weaknesses ----
        st.markdown('<div class="section-heading" style="font-size:1.3rem">📋 Detailed Analysis</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            strengths = a.get("strengths", [])
            items_html = "".join([f'<li style="margin-bottom:6px">{s}</li>' for s in strengths])
            st.markdown(f"""
                <div class="strength-card" style="height:100%">
                    <div style="display:flex;align-items:center;gap:8px;margin-bottom:12px">
                        <span style="font-size:1.4rem">💪</span>
                        <h4 style="color:#22c55e;margin:0;font-size:1.1rem">Strengths</h4>
                    </div>
                    <ul style="margin:0;padding-left:18px;color:#cbd5e1;line-height:1.9;font-size:0.92rem">{items_html}</ul>
                </div>""", unsafe_allow_html=True)
        with col2:
            weaknesses = a.get("weaknesses", [])
            items_html = "".join([f'<li style="margin-bottom:6px">{w}</li>' for w in weaknesses])
            st.markdown(f"""
                <div class="weakness-card" style="height:100%">
                    <div style="display:flex;align-items:center;gap:8px;margin-bottom:12px">
                        <span style="font-size:1.4rem">🔧</span>
                        <h4 style="color:#ef4444;margin:0;font-size:1.1rem">Areas to Improve</h4>
                    </div>
                    <ul style="margin:0;padding-left:18px;color:#cbd5e1;line-height:1.9;font-size:0.92rem">{items_html}</ul>
                </div>""", unsafe_allow_html=True)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        # ---- Missing Skills ----
        if a.get("missing_for_sde"):
            missing = a["missing_for_sde"]
            tags_html = "".join([f'<span class="skill-tag" style="background:rgba(239,68,68,0.1);border-color:rgba(239,68,68,0.25);color:#ef4444">{s}</span>' for s in missing])
            st.markdown(f"""
                <div style="background:linear-gradient(135deg,rgba(239,68,68,0.08),rgba(245,158,11,0.05));
                            border:1px solid rgba(239,68,68,0.2);border-radius:16px;padding:24px">
                    <div style="display:flex;align-items:center;gap:8px;margin-bottom:12px">
                        <span style="font-size:1.3rem">🚨</span>
                        <div style="color:#f87171;font-size:1.1rem;font-weight:700">Missing Skills for Target Roles</div>
                    </div>
                    <div>{tags_html}</div>
                    <div style="margin-top:12px;color:#94a3b8;font-size:0.85rem">
                        Adding these skills could boost your resume score by <strong style="color:#22c55e">10-20 points</strong>
                    </div>
                </div>""", unsafe_allow_html=True)

            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        # ---- ATS Tips ----
        ats_tips = a.get("ats_tips", [])
        if ats_tips:
            st.markdown('<div class="section-heading" style="font-size:1.3rem">🏢 ATS Optimization Tips</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-sub" style="margin-bottom:12px">Tips to pass Applicant Tracking Systems used by 90% of companies</div>', unsafe_allow_html=True)
            for i, tip in enumerate(ats_tips):
                st.markdown(f"""
                    <div class="action-item">
                        <div style="background:linear-gradient(135deg,var(--accent),#7c3aed);color:white;
                                    width:28px;height:28px;border-radius:8px;display:flex;align-items:center;
                                    justify-content:center;font-weight:800;font-size:0.8rem;flex-shrink:0">{i+1}</div>
                        <div style="color:#f1f5f9;font-size:0.92rem;line-height:1.5">{tip}</div>
                    </div>""", unsafe_allow_html=True)


# -----------------------------------------------------------------
# 3. 📊 PLACEMENT PREDICTOR
# -----------------------------------------------------------------
def render_placement_predictor():
    st.markdown('<div class="section-heading">Placement Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Fill in your profile — AI calculates a real placement readiness score</div>', unsafe_allow_html=True)

    if not GEMINI_OK:
        st.error("⚠️ Gemini API not configured.")
        return

    # Profile form
    with st.expander("📋 Your Profile (expand to edit)", expanded=not st.session_state.placement_form_done):
        with st.form("placement_form"):
            col1, col2 = st.columns(2)
            with col1:
                cgpa       = st.slider("CGPA / Percentage", 0.0, 10.0, 7.5, 0.1)
                dsa_level  = st.selectbox("DSA Skill Level", ["Beginner", "Easy-Medium", "Medium", "Medium-Hard", "Hard"])
                projects   = st.number_input("Number of Projects", 0, 20, 2)
            with col2:
                internships      = st.selectbox("Internships", ["None", "1 internship", "2+ internships", "Full-time experience"])
                mock_interviews  = st.number_input("Mock Interviews Done", 0, 50, 3)
                target_role      = st.selectbox("Target Role", ["Software Engineer", "Full-Stack Developer", "Backend Developer", "Data Engineer", "ML Engineer", "DevOps Engineer"])
            target_companies = st.selectbox("Target Companies", ["Any / Startup", "Mid-tier (Flipkart, Swiggy, etc.)", "Product companies (Atlassian, Adobe, etc.)", "FAANG / top-tier"])
            submitted = st.form_submit_button("🚀 Predict My Score", type="primary")

        if submitted:
            profile = {
                "cgpa": cgpa,
                "dsa_level": dsa_level,
                "projects": projects,
                "internships": internships,
                "mock_interviews": mock_interviews,
                "target_role": target_role,
                "target_companies": target_companies,
                "has_resume": st.session_state.resume_analyzed,
                "resume_score": st.session_state.resume_analysis.get("placement_score") if st.session_state.resume_analysis else "N/A",
            }
            st.session_state.placement_profile = profile
            with st.spinner("🤖 Computing your readiness score…"):
                result = ai_predict_placement(profile)
            if result:
                st.session_state.placement_data    = result
                st.session_state.placement_form_done = True
                st.rerun()

    if st.session_state.placement_data:
        pred = st.session_state.placement_data
        score = pred.get("score", 0)

        # Big score display
        color = score_color(score)
        st.markdown(f'<div class="big-score" style="background:linear-gradient(135deg,{color},{color}aa);-webkit-background-clip:text;-webkit-text-fill-color:transparent">{score}%</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="big-score-label">{pred.get("verdict","Placement Readiness")} · Grade: <strong style="color:{color}">{pred.get("grade","—")}</strong></div>', unsafe_allow_html=True)
        st.progress(score / 100)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        # Summary + company match
        col_sum, col_co = st.columns([3, 2])
        with col_sum:
            st.markdown(f"""
                <div class="ai-insight">
                    <h4>🧠 AI Summary</h4>
                    <p>{pred.get('summary','')}</p>
                </div>""", unsafe_allow_html=True)

            if pred.get("key_strengths"):
                s_items = "".join([f'<li style="color:#cbd5e1;line-height:2">{s}</li>' for s in pred["key_strengths"]])
                st.markdown(f'<div class="strength-card"><h4 style="color:#22c55e;margin-top:0">✅ Key Strengths</h4><ul style="margin:0;padding-left:18px">{s_items}</ul></div>', unsafe_allow_html=True)

        with col_co:
            company_match = pred.get("company_match", {})
            if company_match:
                st.markdown('<div class="section-heading" style="font-size:1.1rem">🏢 Company Fit</div>', unsafe_allow_html=True)
                for company, pct in company_match.items():
                    c = score_color(pct)
                    st.markdown(f'<div style="display:flex;justify-content:space-between;color:#f1f5f9;font-size:0.9rem;margin-bottom:4px"><span>{company}</span><span style="color:{c};font-weight:700">{pct}%</span></div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="company-bar"><div class="company-bar-fill" style="width:{pct}%;background:linear-gradient(90deg,{c},{c}aa)"></div></div>', unsafe_allow_html=True)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        # Skill breakdown
        breakdown = pred.get("breakdown", {})
        if breakdown:
            st.markdown('<div class="section-heading" style="font-size:1.1rem">📊 Score Breakdown</div>', unsafe_allow_html=True)
            b_cols = st.columns(len(breakdown))
            for i, (skill, val) in enumerate(breakdown.items()):
                label = skill.replace("_", " ").title()
                c = score_color(val)
                with b_cols[i]:
                    st.markdown(f"""<div class="metric-card" style="border-color:{c}30">
                        <div class="metric-label">{label}</div>
                        <div class="metric-value" style="font-size:1.6rem;color:{c}">{val}%</div>
                    </div>""", unsafe_allow_html=True)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        # Action items
        action_items = pred.get("action_items", [])
        if action_items:
            st.markdown('<div class="section-heading" style="font-size:1.1rem">⚡ Your Action Plan</div>', unsafe_allow_html=True)
            for item in action_items:
                priority = item.get("priority", "Medium")
                badge_class = f"badge-priority-{priority.lower()}"
                impact = item.get("impact", "")
                timeframe = item.get("timeframe", "")
                st.markdown(f"""
                    <div class="action-item">
                        <div style="flex:1">
                            <div style="color:#f1f5f9;font-weight:500;font-size:0.95rem">{item.get('action','')}</div>
                            <div style="color:#94a3b8;font-size:0.8rem;margin-top:4px">{timeframe}</div>
                        </div>
                        <div style="display:flex;flex-direction:column;align-items:flex-end;gap:4px">
                            <span style="color:#22c55e;font-weight:700;font-size:0.9rem">{impact}</span>
                            <span class="{badge_class}">{priority}</span>
                        </div>
                    </div>""", unsafe_allow_html=True)


# -----------------------------------------------------------------
# 4. 🧠 ADAPTIVE LEARNING PATH
# -----------------------------------------------------------------
def render_adaptive_learning():
    st.markdown('<div class="section-heading">Adaptive Learning Path</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">AI generates a personalized sprint based on your weak areas</div>', unsafe_allow_html=True)

    if not GEMINI_OK:
        st.error("⚠️ Gemini API not configured.")
        return

    # Auto-detect weak areas from resume / prediction if available
    auto_weak = []
    if st.session_state.resume_analysis:
        a = st.session_state.resume_analysis
        auto_weak += a.get("missing_for_sde", [])[:3]
    if st.session_state.placement_data:
        p = st.session_state.placement_data
        auto_weak += p.get("critical_gaps", [])[:2]

    with st.expander("⚙️ Customize Your Plan", expanded=not bool(st.session_state.learning_plan)):
        with st.form("learning_form"):
            target_role = st.selectbox("Target Role", ["Software Engineer", "Full-Stack Developer", "Backend Developer", "Data Engineer", "ML Engineer", "DevOps Engineer"])
            days        = st.slider("Sprint Duration (days)", 3, 14, 7)

            # Pre-populate with auto-detected weak areas
            default_weak = ", ".join(list(dict.fromkeys(auto_weak))[:3]) if auto_weak else "DSA, System Design, Cloud"
            weak_input = st.text_input("Weak Areas (comma-separated)", value=default_weak, placeholder="e.g. DSA, System Design, AWS")
            gen_btn = st.form_submit_button("🧠 Generate My Plan", type="primary")

        if gen_btn:
            weak_areas = [w.strip() for w in weak_input.split(",") if w.strip()]
            with st.spinner(f"🤖 Building your {days}-day personalized plan…"):
                plan = ai_generate_learning_plan(weak_areas, target_role, days)
            if plan:
                st.session_state.learning_plan = plan
                st.session_state.learning_profile = {"role": target_role, "days": days, "weak": weak_areas}
                st.rerun()

    if st.session_state.learning_plan:
        plan = st.session_state.learning_plan
        profile = st.session_state.learning_profile

        # Goal banner
        st.markdown(f"""
            <div class="success-alert">
                🎯 <strong>Sprint Goal:</strong> {plan.get('weekly_goal', '')} &nbsp;|&nbsp;
                📏 <strong>Success:</strong> {plan.get('success_metric', '')}
            </div>""", unsafe_allow_html=True)

        # Badge map
        badge_map = {"Easy": "badge-easy", "Medium": "badge-medium", "Hard": "badge-hard"}

        for i, day in enumerate(plan.get("plan", [])):
            diff = day.get("difficulty", "Medium")
            badge_class = badge_map.get(diff, "badge-medium")
            tasks_html = "".join([f'<li style="color:#cbd5e1;margin-bottom:4px">{t}</li>' for t in day.get("tasks", [])])
            resource = day.get("resource", "")
            resource_html = f'<div style="margin-top:10px;color:#6366f1;font-size:0.82rem">📚 {resource}</div>' if resource else ""

            st.markdown(f"""
                <div class="day-card">
                    <div style="display:flex;justify-content:space-between;align-items:center">
                        <div class="day-title">{day.get('day', f'Day {i+1}')}</div>
                        <span class="{badge_class}">{diff}</span>
                    </div>
                    <div class="day-focus">🎯 {day.get('focus', '')}</div>
                </div>""", unsafe_allow_html=True)

            for task in day.get("tasks", []):
                st.checkbox(task, key=f"learn_{i}_{task[:40]}")
            if resource:
                st.markdown(f'<div style="color:#6366f1;font-size:0.82rem;margin-bottom:8px;margin-left:4px">📚 {resource}</div>', unsafe_allow_html=True)


# -----------------------------------------------------------------
# 5. 🤖 AI MENTOR CHAT
# -----------------------------------------------------------------
def render_ai_mentor():
    st.markdown('<div class="section-heading">AI Mentor</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Your always-available career coach — ask anything</div>', unsafe_allow_html=True)

    if not GEMINI_OK:
        st.error("⚠️ AI not configured.")
        return

    # Quick prompts FIRST (above chat)
    if len(st.session_state.chat_history) <= 1:
        quick_prompts = [
            "How do I crack system design interviews?",
            "Give me the Blind 75 strategy",
            "How to negotiate salary after an offer?",
            "What should I highlight on my resume?",
        ]
        st.markdown('<div style="color:#94a3b8;font-size:0.8rem;font-weight:600;margin-bottom:8px">TRY ASKING:</div>', unsafe_allow_html=True)
        q_cols = st.columns(len(quick_prompts))
        for i, qp in enumerate(quick_prompts):
            with q_cols[i]:
                if st.button(qp, key=f"qp_{i}"):
                    st.session_state.chat_history.append({"role": "user", "content": qp})
                    with st.spinner("🤖 Thinking…"):
                        reply = ai_mentor_reply(st.session_state.chat_history)
                    st.session_state.chat_history.append({"role": "ai", "content": reply})
                    st.rerun()
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Chat container with fixed height + scroll
    chat_html = '<div id="chat-box" style="max-height:500px;overflow-y:auto;padding:10px 0">'
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            chat_html += f'<div class="chat-user">{msg["content"]}</div>'
        else:
            chat_html += f'<div class="chat-ai">🤖&nbsp;&nbsp;{msg["content"]}</div>'
    chat_html += '</div>'
    st.markdown(chat_html, unsafe_allow_html=True)

    # Auto-scroll to bottom
    st.markdown("""
        <script>
            const chatBox = document.getElementById('chat-box');
            if (chatBox) chatBox.scrollTop = chatBox.scrollHeight;
        </script>
    """, unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Clear chat button
    if len(st.session_state.chat_history) > 1:
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = [st.session_state.chat_history[0]]
            st.rerun()

    # Main chat input (always at bottom)
    if prompt := st.chat_input("Ask me anything — interview tips, DSA help, career advice…"):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.spinner("🤖 Thinking…"):
            reply = ai_mentor_reply(st.session_state.chat_history)
        st.session_state.chat_history.append({"role": "ai", "content": reply})
        st.rerun()


# -----------------------------------------------------------------
# 6. 🎤 MOCK INTERVIEW
# -----------------------------------------------------------------
def render_mock_interview():
    st.markdown('<div class="section-heading">Mock Interview</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">AI generates fresh questions and evaluates your answers in real-time</div>', unsafe_allow_html=True)

    if not GEMINI_OK:
        st.error("⚠️ AI not configured.")
        return

    if not st.session_state.interview_started:
        card("""
            <div style="text-align:center;padding:20px 0">
                <div style="font-size:3rem;margin-bottom:12px">🎤</div>
                <h3 style="color:#f1f5f9;margin:0 0 8px">Ready to practice?</h3>
                <p style="color:#94a3b8;margin:0;max-width:500px;margin:0 auto">
                    The AI will generate a fresh question based on your chosen role and topic, then give you detailed feedback on your answer — scoring correctness, clarity, and depth.
                </p>
            </div>
        """)

        col1, col2, col3 = st.columns(3)
        with col1:
            role = st.selectbox("Role", ["Software Engineer", "Full-Stack Developer", "Backend Developer", "Data Analyst", "ML Engineer"])
        with col2:
            topic = st.selectbox("Topic", ["Data Structures & Algorithms", "System Design", "Object-Oriented Programming", "Databases & SQL", "Behavioral / HR", "JavaScript / Python Concepts", "Operating Systems", "Computer Networks"])
        with col3:
            difficulty = st.selectbox("Difficulty", ["Easy", "Medium", "Hard"])

        _, col_gen, _ = st.columns([1, 2, 1])
        with col_gen:
            gen_btn = st.button("▶️  Generate Question & Start", type="primary", use_container_width=True)
        if gen_btn:
            with st.spinner("🤖 Crafting your question…"):
                q = ai_generate_question(role, topic, difficulty)
            if q:
                st.session_state.interview_question    = q
                st.session_state.interview_started     = True
                st.session_state.interview_feedback    = None
                st.rerun()
            else:
                st.error("Failed to generate question. Please try again.")

    else:
        q = st.session_state.interview_question or {}

        # Question card
        q_type      = q.get("type", "Technical")
        q_tests     = q.get("what_it_tests", "")
        st.markdown(f"""
            <div class="card">
                <div style="display:flex;align-items:center;gap:10px;margin-bottom:14px">
                    <span style="font-size:1.4rem">🤖</span>
                    <div>
                        <span style="color:#94a3b8;font-size:0.8rem;font-weight:600;text-transform:uppercase;letter-spacing:0.5px">AI Interviewer</span>
                        <span style="margin-left:10px;background:rgba(99,102,241,0.15);color:#818cf8;padding:2px 8px;border-radius:6px;font-size:0.75rem;font-weight:700">{q_type}</span>
                    </div>
                </div>
                <p style="color:#f1f5f9;font-size:1.1rem;margin:0;line-height:1.7">{q.get('question','')}</p>
                {f'<div style="margin-top:12px;color:#94a3b8;font-size:0.82rem">💡 Tests: {q_tests}</div>' if q_tests else ''}
            </div>""", unsafe_allow_html=True)

        # Hints (optional reveal)
        hints = q.get("hints", [])
        if hints:
            with st.expander("💡 Show hints (only if stuck)"):
                for hint in hints:
                    st.markdown(f"• {hint}")

        answer = st.text_area("Your Answer:", height=180, placeholder="Type your complete answer here… Take your time, just like a real interview.")

        col1, col2, col3 = st.columns([1, 1, 3])
        with col1:
            if st.button("📤 Submit Answer", type="primary"):
                if answer.strip():
                    with st.spinner("🤖 Evaluating your answer…"):
                        fb = ai_evaluate_answer(q.get("question", ""), answer, q_type)
                    if fb:
                        st.session_state.interview_feedback = fb
                        st.rerun()
                    else:
                        st.error("Evaluation failed. Please try again.")
                else:
                    st.warning("Please write an answer before submitting.")
        with col2:
            if st.button("⏭️ New Question"):
                st.session_state.interview_started  = False
                st.session_state.interview_question = None
                st.session_state.interview_feedback = None
                st.rerun()
        with col3:
            if st.button("🛑 End Interview"):
                st.session_state.interview_started  = False
                st.session_state.interview_question = None
                st.session_state.interview_feedback = None
                st.rerun()

        if st.session_state.interview_feedback:
            fb = st.session_state.interview_feedback
            overall = fb.get("overall", 0)
            color = score_color(overall)

            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            st.markdown(f"""
                <div class="eval-card">
                    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:20px">
                        <h4 style="color:#f1f5f9;margin:0">📊 AI Evaluation</h4>
                        <div style="text-align:center">
                            <div style="font-size:2rem;font-weight:900;color:{color}">{overall}%</div>
                            <div style="color:#94a3b8;font-size:0.78rem">Overall</div>
                        </div>
                    </div>
                    <div style="display:flex;gap:16px;margin-bottom:20px">
                        <div class="metric-card" style="flex:1">
                            <div class="metric-label">Correctness</div>
                            <div class="metric-value" style="font-size:1.6rem">{fb.get('correctness',0)}%</div>
                        </div>
                        <div class="metric-card" style="flex:1">
                            <div class="metric-label">Clarity</div>
                            <div class="metric-value" style="font-size:1.6rem">{fb.get('clarity',0)}%</div>
                        </div>
                        <div class="metric-card" style="flex:1">
                            <div class="metric-label">Depth</div>
                            <div class="metric-value" style="font-size:1.6rem">{fb.get('depth',0)}%</div>
                        </div>
                    </div>
                    <div style="background:rgba(99,102,241,0.08);border:1px solid rgba(99,102,241,0.2);border-radius:10px;padding:16px;margin-bottom:16px">
                        <strong style="color:#818cf8">💬 Feedback</strong>
                        <p style="color:#cbd5e1;margin:8px 0 0;line-height:1.6">{fb.get('feedback','')}</p>
                    </div>
                </div>""", unsafe_allow_html=True)

            col_good, col_improve = st.columns(2)
            with col_good:
                if fb.get("what_was_good"):
                    items = "".join([f'<li style="color:#cbd5e1;line-height:2">{p}</li>' for p in fb["what_was_good"]])
                    st.markdown(f'<div class="strength-card"><h4 style="color:#22c55e;margin-top:0">✅ What You Did Well</h4><ul style="margin:0;padding-left:18px">{items}</ul></div>', unsafe_allow_html=True)
            with col_improve:
                if fb.get("what_to_improve"):
                    items = "".join([f'<li style="color:#cbd5e1;line-height:2">{p}</li>' for p in fb["what_to_improve"]])
                    st.markdown(f'<div class="weakness-card"><h4 style="color:#ef4444;margin-top:0">⚠️ Areas to Improve</h4><ul style="margin:0;padding-left:18px">{items}</ul></div>', unsafe_allow_html=True)

            if fb.get("ideal_answer_hint"):
                st.markdown(f"""
                    <div class="ai-insight" style="margin-top:16px">
                        <h4>💡 Ideal Answer Hint</h4>
                        <p>{fb['ideal_answer_hint']}</p>
                    </div>""", unsafe_allow_html=True)


# -----------------------------------------------------------------
# 7. 📈 PROGRESS TRACKING
# -----------------------------------------------------------------
def render_progress_tracking():
    st.markdown('<div class="section-heading">Progress Tracking</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Monitor your skill growth and task completion over time</div>', unsafe_allow_html=True)

    # Summary cards
    pred = st.session_state.placement_data
    analysis = st.session_state.resume_analysis

    c1, c2, c3 = st.columns(3)
    with c1:
        score = pred["score"] if pred else 0
        metric_card("Placement Score", f"{score}%" if pred else "Not assessed", "Run Predictor →" if not pred else score_verdict(score))
    with c2:
        skills = len(analysis.get("skills", [])) if analysis else 0
        metric_card("Skills Detected", str(skills) if analysis else "—", "from resume analysis" if analysis else "Upload resume →")
    with c3:
        has_plan = bool(st.session_state.learning_plan)
        days_planned = len(st.session_state.learning_plan.get("plan", [])) if has_plan else 0
        metric_card("Days Planned", str(days_planned) if has_plan else "—", "Generate plan →" if not has_plan else "sprint active")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Skill growth chart (simulated trend + real score if available)
    st.markdown("#### 📈 Skill Growth — Last 30 Days")
    dates = pd.date_range(end=datetime.date.today(), periods=30)

    base_dsa = pred["breakdown"].get("technical_skills", 50) if pred else 50
    base_apt = pred["breakdown"].get("interview_readiness", 45) if pred else 45

    chart_data = pd.DataFrame(
        {
            "DSA/Technical": [max(10, base_dsa - 20 + (i * 0.75) + (i % 3)) for i in range(30)],
            "Interview Readiness": [max(10, base_apt - 15 + (i * 0.6) - (i % 2)) for i in range(30)],
        },
        index=dates,
    )
    st.line_chart(chart_data, color=["#6366f1", "#22c55e"])

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Real weak areas from AI
    st.markdown("#### ⚠️ Areas Needing Attention")

    weak_list = []
    if analysis and analysis.get("missing_for_sde"):
        for s in analysis["missing_for_sde"][:3]:
            weak_list.append((s, "Missing from your resume — add a project or certification"))
    if pred and pred.get("critical_gaps"):
        for g in pred["critical_gaps"][:3]:
            if g not in [w[0] for w in weak_list]:
                weak_list.append((g, "Identified as a critical gap by the AI predictor"))

    if not weak_list:
        weak_list = [
            ("Mock Interviews", "Complete more mock interviews to improve interview readiness"),
            ("System Design", "Practice common system design problems (URL shortener, Twitter, etc.)"),
            ("Consistency", "Maintain daily practice to build lasting momentum"),
        ]

    for area, detail in weak_list:
        st.markdown(f"""
            <div class="weak-area">
                <strong>{area}:</strong> {detail}
            </div>""", unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Weekly task chart
    st.markdown("#### 📊 Weekly Task Completion")
    bar_data = pd.DataFrame(
        {"Completed": [12, 15, 10, 18], "Missed": [3, 1, 5, 2]},
        index=["Week 1", "Week 2", "Week 3", "Week 4"],
    )
    st.bar_chart(bar_data, color=["#6366f1", "#ef4444"])


# -----------------------------------------------------------------
# AUTH: NAVBAR, SIGN IN, SIGN UP
# -----------------------------------------------------------------
def render_authenticated_navbar():
    """Render the top navigation bar for authenticated users."""
    name = st.session_state.full_name or st.session_state.username or "U"
    initials = "".join([w[0].upper() for w in name.split()[:2]])

    st.markdown(
        f"""
        <div style="
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 14px 28px;
            background: rgba(17, 24, 39, 0.85);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid rgba(255,255,255,0.06);
            border-radius: 16px;
            margin-bottom: 20px;
            box-shadow: 0 4px 24px rgba(0,0,0,0.25);
        ">
            <div style="display:flex; align-items:center; gap:12px;">
                <div style="
                    width:38px; height:38px; border-radius:10px;
                    background:linear-gradient(135deg, #6366f1, #7c3aed);
                    display:flex; align-items:center; justify-content:center;
                    font-size:1.2rem; font-weight:800; color:white;
                    box-shadow:0 2px 10px rgba(99,102,241,0.35);
                ">CF</div>
                <div style="
                    font-size:1.35rem; font-weight:800;
                    background:linear-gradient(135deg, #fff, #818cf8);
                    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                    letter-spacing:-0.5px;
                ">CareerForge</div>
            </div>
            <div style="display:flex; align-items:center; gap:10px;">
                <span style="color:#f1f5f9; font-size:0.9rem; font-weight:500;">{st.session_state.full_name or st.session_state.username}</span>
                <div style="
                    width:34px; height:34px; border-radius:50%;
                    background:linear-gradient(135deg, #6366f1, #a78bfa);
                    display:flex; align-items:center; justify-content:center;
                    font-size:0.85rem; font-weight:700; color:white;
                ">{initials}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    # Logout button (small, right-aligned)
    cols = st.columns([10, 1])
    with cols[1]:
        if st.button("Logout", key="logout_btn"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()


# -----------------------------------------------------------------
# AUTH DIALOGS ΓÇö Sign In / Sign Up as separate pages
# -----------------------------------------------------------------
def render_signin_page():
    """Render the Sign In page."""
    st.markdown(
        """
        <div class="hero" style="padding-bottom:10px;">
            <h1>Welcome Back</h1>
            <p>Sign in to continue your career journey</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _, col_form, _ = st.columns([1.5, 2, 1.5])

    with col_form:
        st.markdown(
            """
            <div class="auth-header">
                <h2>Sign In</h2>
                <p>Enter your credentials</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.form("signin_form", clear_on_submit=False):
            si_username = st.text_input("Username", key="si_user", placeholder="Enter your username")
            si_password = st.text_input("Password", type="password", key="si_pass", placeholder="Enter your password")
            si_submit = st.form_submit_button("Sign In", type="primary", use_container_width=True)

            if si_submit:
                if not si_username or not si_password:
                    st.error("Please fill in all fields.")
                else:
                    try:
                        user = db.authenticate_user(si_username.strip(), si_password)
                    except Exception as e:
                        st.error(f"⚠️ Database error: {e}")
                        user = None
                    if user:
                        st.session_state.authenticated = True
                        st.session_state.user_id = user["id"]
                        st.session_state.username = user["username"]
                        st.session_state.full_name = user["full_name"]
                        st.session_state.show_page = "dashboard"
                        load_user_data()
                        st.rerun()
                    else:
                        st.error("Invalid username or password.")

        st.markdown("")
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("ΓåÉ Back to Home", use_container_width=True):
                st.session_state.show_page = "landing"
                st.rerun()
        with col_b:
            if st.button("Create Account ΓåÆ", use_container_width=True):
                st.session_state.show_page = "signup"
                st.rerun()


def render_signup_page():
    """Render the Sign Up page."""
    st.markdown(
        """
        <div class="hero" style="padding-bottom:10px;">
            <h1>Join CareerForge</h1>
            <p>Create your free account and start your journey</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _, col_form, _ = st.columns([1.5, 2, 1.5])

    with col_form:
        st.markdown(
            """
            <div class="auth-header">
                <h2>Sign Up</h2>
                <p>It only takes a minute</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.form("signup_form", clear_on_submit=False):
            su_fullname = st.text_input("Full Name", key="su_name", placeholder="Enter your full name")
            su_email = st.text_input("Email", key="su_email", placeholder="Enter your email")
            su_username = st.text_input("Username", key="su_user", placeholder="Choose a username")
            su_password = st.text_input("Password", type="password", key="su_pass", placeholder="Create a password")
            su_submit = st.form_submit_button("Create Account", type="primary", use_container_width=True)

            if su_submit:
                if not su_fullname or not su_username or not su_password or not su_email:
                    st.error("Please fill in all fields.")
                elif len(su_password) < 4:
                    st.error("Password must be at least 4 characters.")
                else:
                    try:
                        username_taken = db.check_username_exists(su_username.strip())
                    except Exception as e:
                        st.error(f"⚠️ Database error: {e}")
                        username_taken = None

                    if username_taken:
                        st.error("Username already taken. Try another one.")
                    elif username_taken is not None:
                        try:
                            user_id = db.create_user(
                                username=su_username.strip(),
                                full_name=su_fullname.strip(),
                                email=su_email.strip(),
                                password=su_password,
                            )
                        except Exception as e:
                            st.error(f"⚠️ Database error: {e}")
                            user_id = None
                        if user_id:
                            st.session_state.authenticated = True
                            st.session_state.user_id = user_id
                            st.session_state.username = su_username.strip()
                            st.session_state.full_name = su_fullname.strip()
                            st.session_state.show_page = "dashboard"
                            load_user_data()
                            st.success("Account created! Redirecting...")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error("Username or email already exists.")

        st.markdown("")
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("ΓåÉ Back to Home", key="back_home_su", use_container_width=True):
                st.session_state.show_page = "landing"
                st.rerun()
        with col_b:
            if st.button("Already have an account? Sign In ΓåÆ", key="goto_signin", use_container_width=True):
                st.session_state.show_page = "signin"
                st.rerun()


# -----------------------------------------------------------------
# LANDING PAGE ΓÇö shown to unauthenticated / first-time visitors
# -----------------------------------------------------------------

def render_landing_page():
    """Immersive landing page with sparkle cursor, animated background, and interactive features."""

    # ==========================================
    # FULL-PAGE INTERACTIVE EXPERIENCE VIA HTML
    # Write HTML to a static file so we can serve it
    # via an unsandboxed iframe (allows navigation)
    # ==========================================
    import os
    
    # Create static directory for Streamlit
    static_dir = os.path.join(os.path.dirname(__file__), "static")
    os.makedirs(static_dir, exist_ok=True)
    
    html_content = """
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&display=swap');

* { margin:0; padding:0; box-sizing:border-box; }

body {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    background: #0a0e1a;
    color: #f1f5f9;
    overflow-x: hidden;
    cursor: none;
}

/* ΓÇöΓÇöΓÇö Custom Cursor ΓÇöΓÇöΓÇö */
.cursor-dot {
    width: 10px; height: 10px;
    background: radial-gradient(circle, #fff 0%, #818cf8 60%, transparent 100%);
    border-radius: 50%;
    position: fixed;
    pointer-events: none;
    z-index: 99999;
    transition: transform 0.05s ease;
    box-shadow: 0 0 20px rgba(129,140,248,0.8), 0 0 40px rgba(99,102,241,0.4), 0 0 6px #fff;
}
.cursor-ring {
    width: 40px; height: 40px;
    border: 2px solid rgba(129,140,248,0.5);
    border-radius: 50%;
    position: fixed;
    pointer-events: none;
    z-index: 99998;
    transition: width 0.15s ease, height 0.15s ease, border-color 0.15s ease;
    box-shadow: 0 0 12px rgba(129,140,248,0.15);
}

/* ΓÇöΓÇöΓÇö Interactive Background Glow (follows cursor) ΓÇöΓÇöΓÇö */
.cursor-glow {
    position: fixed;
    width: 600px; height: 600px;
    border-radius: 50%;
    pointer-events: none;
    z-index: 2;
    background: radial-gradient(circle, rgba(99,102,241,0.12) 0%, rgba(124,58,237,0.06) 30%, transparent 70%);
    filter: blur(40px);
    transform: translate(-50%, -50%);
    transition: none;
}

/* ΓÇöΓÇöΓÇö Sparkle particles (cross/star shaped) ΓÇöΓÇöΓÇö */
.sparkle {
    position: fixed;
    pointer-events: none;
    z-index: 99997;
    animation: sparkle-fade 1s ease forwards;
}
.sparkle::before, .sparkle::after {
    content: '';
    position: absolute;
    top: 50%; left: 50%;
    transform: translate(-50%, -50%);
    background: inherit;
    border-radius: 1px;
}
.sparkle::before {
    width: 100%; height: 30%;
}
.sparkle::after {
    width: 30%; height: 100%;
}
@keyframes sparkle-fade {
    0% { transform: scale(1) rotate(0deg) translate(0,0); opacity:1; }
    50% { transform: scale(1.3) rotate(45deg) translate(calc(var(--tx) * 0.5), calc(var(--ty) * 0.5)); opacity:0.7; }
    100% { transform: scale(0) rotate(90deg) translate(var(--tx), var(--ty)); opacity:0; }
}

/* ΓÇöΓÇöΓÇö Loading Screen ΓÇöΓÇöΓÇö */
.loading-screen {
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: 100%;
    background: #0a0e1a;
    z-index: 100000;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    transition: opacity 0.6s ease, visibility 0.6s ease;
}
.loading-screen.hidden {
    opacity: 0;
    visibility: hidden;
}
.loading-logo {
    width: 80px; height: 80px;
    border-radius: 20px;
    background: linear-gradient(135deg, #6366f1, #7c3aed);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 2rem;
    font-weight: 900;
    color: white;
    animation: pulse-glow 1.5s ease-in-out infinite;
    box-shadow: 0 0 40px rgba(99,102,241,0.4);
}
@keyframes pulse-glow {
    0%, 100% { transform: scale(1); box-shadow: 0 0 40px rgba(99,102,241,0.4); }
    50% { transform: scale(1.08); box-shadow: 0 0 60px rgba(99,102,241,0.6); }
}
.loading-bar-track {
    width: 200px; height: 4px;
    background: rgba(255,255,255,0.08);
    border-radius: 10px;
    margin-top: 28px;
    overflow: hidden;
}
.loading-bar-fill {
    height: 100%;
    border-radius: 10px;
    background: linear-gradient(90deg, #6366f1, #818cf8, #a78bfa);
    animation: load-progress 1.8s ease-in-out forwards;
}
@keyframes load-progress {
    0% { width: 0%; }
    100% { width: 100%; }
}
.loading-text {
    margin-top: 16px;
    font-size: 0.85rem;
    color: #64748b;
    letter-spacing: 2px;
    text-transform: uppercase;
    font-weight: 600;
}

/* ΓÇöΓÇöΓÇö Animated Background ΓÇöΓÇöΓÇö */
.bg-canvas {
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: 100%;
    z-index: 0;
    overflow: hidden;
}
.bg-orb {
    position: absolute;
    border-radius: 50%;
    filter: blur(80px);
    animation: float-orb 20s ease-in-out infinite;
    opacity: 0.4;
    transition: transform 0.8s cubic-bezier(0.25, 0.46, 0.45, 0.94);
}
.bg-orb:nth-child(1) {
    width: 500px; height: 500px;
    background: radial-gradient(circle, rgba(99,102,241,0.35), transparent);
    top: -10%; left: -5%;
    animation-duration: 25s;
}
.bg-orb:nth-child(2) {
    width: 400px; height: 400px;
    background: radial-gradient(circle, rgba(124,58,237,0.3), transparent);
    bottom: -10%; right: -5%;
    animation-duration: 20s;
    animation-delay: -5s;
}
.bg-orb:nth-child(3) {
    width: 350px; height: 350px;
    background: radial-gradient(circle, rgba(6,182,212,0.2), transparent);
    top: 40%; left: 50%;
    animation-duration: 30s;
    animation-delay: -10s;
}
.bg-orb:nth-child(4) {
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(244,63,94,0.15), transparent);
    top: 20%; right: 20%;
    animation-duration: 22s;
    animation-delay: -7s;
}
@keyframes float-orb {
    0%, 100% { transform: translate(0, 0) scale(1); }
    25% { transform: translate(60px, -40px) scale(1.1); }
    50% { transform: translate(-30px, 60px) scale(0.95); }
    75% { transform: translate(40px, 30px) scale(1.05); }
}

/* ΓÇöΓÇöΓÇö Grid lines (deforms near cursor) ΓÇöΓÇöΓÇö */
.grid-overlay {
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: 100%;
    z-index: 1;
    background-image:
        linear-gradient(rgba(99,102,241,0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(99,102,241,0.03) 1px, transparent 1px);
    background-size: 60px 60px;
    pointer-events: none;
    transition: background-position 0.3s ease;
}

/* ΓÇöΓÇöΓÇö Main Content ΓÇöΓÇöΓÇö */
.main-content {
    position: relative;
    z-index: 10;
    max-width: 1200px;
    margin: 0 auto;
    padding: 0 24px;
}

/* ΓÇöΓÇöΓÇö Top Bar ΓÇö CareerForge logo + Nav + Auth ΓÇöΓÇöΓÇö */
.top-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 18px 0;
    gap: 20px;
}
.brand-link {
    display: flex;
    align-items: center;
    gap: 12px;
    cursor: pointer;
    text-decoration: none;
    flex-shrink: 0;
    transition: transform 0.2s ease;
}
.brand-link:hover {
    transform: scale(1.03);
}
.brand-logo {
    width: 42px; height: 42px;
    border-radius: 12px;
    background: linear-gradient(135deg, #6366f1, #7c3aed);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.1rem;
    font-weight: 900;
    color: white;
    box-shadow: 0 4px 16px rgba(99,102,241,0.4);
    transition: box-shadow 0.3s ease;
}
.brand-link:hover .brand-logo {
    box-shadow: 0 6px 24px rgba(99,102,241,0.6);
}
.brand-name {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.5rem;
    font-weight: 700;
    background: linear-gradient(135deg, #fff, #818cf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.5px;
}

/* ΓÇöΓÇöΓÇö Center Nav ΓÇöΓÇöΓÇö */
.nav-center {
    display: flex;
    align-items: center;
    gap: 4px;
    background: rgba(17, 24, 39, 0.7);
    backdrop-filter: blur(16px);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 14px;
    padding: 5px 6px;
}
.nav-link {
    padding: 9px 18px;
    font-size: 0.85rem;
    font-weight: 500;
    color: #94a3b8;
    border-radius: 10px;
    cursor: pointer;
    transition: all 0.25s ease;
    text-decoration: none;
    white-space: nowrap;
    position: relative;
    overflow: hidden;
}
.nav-link:hover {
    color: #f1f5f9;
    background: rgba(99,102,241,0.12);
}
.nav-link::after {
    content: '';
    position: absolute;
    bottom: 4px; left: 50%;
    width: 0; height: 2px;
    background: #6366f1;
    border-radius: 2px;
    transition: all 0.3s ease;
    transform: translateX(-50%);
}
.nav-link:hover::after {
    width: 60%;
}

/* ΓÇöΓÇöΓÇö Auth Buttons ΓÇöΓÇöΓÇö */
.auth-btns {
    display: flex;
    align-items: center;
    gap: 10px;
    flex-shrink: 0;
}
.btn-signin {
    padding: 9px 22px;
    font-size: 0.85rem;
    font-weight: 600;
    color: #cbd5e1;
    background: transparent;
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 10px;
    cursor: pointer;
    transition: all 0.25s ease;
}
.btn-signin:hover {
    color: #fff;
    border-color: rgba(99,102,241,0.5);
    background: rgba(99,102,241,0.08);
    transform: translateY(-1px);
}
.btn-signup {
    padding: 9px 22px;
    font-size: 0.85rem;
    font-weight: 600;
    color: white;
    background: linear-gradient(135deg, #6366f1, #7c3aed);
    border: none;
    border-radius: 10px;
    cursor: pointer;
    transition: all 0.25s ease;
    box-shadow: 0 4px 14px rgba(99,102,241,0.35);
}
.btn-signup:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 22px rgba(99,102,241,0.5);
}

/* ΓÇöΓÇöΓÇö HERO ΓÇöΓÇöΓÇö */
.hero-section {
    text-align: center;
    padding: 80px 20px 60px;
    position: relative;
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 8px 18px;
    background: rgba(99,102,241,0.12);
    border: 1px solid rgba(99,102,241,0.25);
    border-radius: 50px;
    font-size: 0.82rem;
    font-weight: 600;
    color: #818cf8;
    margin-bottom: 28px;
    animation: fade-in-up 0.8s ease;
}
.hero-badge .pulse-dot {
    width: 8px; height: 8px;
    background: #22c55e;
    border-radius: 50%;
    animation: pulse-dot 2s ease-in-out infinite;
}
@keyframes pulse-dot {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.5; transform: scale(1.5); }
}
.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 4.5rem;
    font-weight: 700;
    line-height: 1.08;
    letter-spacing: -2.5px;
    margin-bottom: 24px;
    animation: fade-in-up 0.8s ease 0.1s both;
}
.hero-title .line1 {
    display: block;
    color: #f1f5f9;
}
.hero-title .line2 {
    display: block;
    background: linear-gradient(135deg, #6366f1, #818cf8, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero-title .line3 {
    display: block;
    background: linear-gradient(135deg, #06b6d4 0%, #818cf8 50%, #f43f5e 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero-subtitle {
    font-size: 1.2rem;
    color: #94a3b8;
    max-width: 580px;
    margin: 0 auto 40px;
    line-height: 1.7;
    animation: fade-in-up 0.8s ease 0.2s both;
}
@keyframes fade-in-up {
    0% { opacity: 0; transform: translateY(30px); }
    100% { opacity: 1; transform: translateY(0); }
}

/* ΓÇöΓÇöΓÇö Hero CTAs ΓÇöΓÇöΓÇö */
.hero-ctas {
    display: flex;
    justify-content: center;
    gap: 16px;
    animation: fade-in-up 0.8s ease 0.3s both;
}
.cta-primary {
    padding: 16px 36px;
    font-size: 1rem;
    font-weight: 700;
    color: white;
    background: linear-gradient(135deg, #6366f1, #7c3aed);
    border: none;
    border-radius: 14px;
    cursor: pointer;
    transition: all 0.3s ease;
    box-shadow: 0 6px 24px rgba(99,102,241,0.4);
    position: relative;
    overflow: hidden;
}
.cta-primary:hover {
    transform: translateY(-3px);
    box-shadow: 0 10px 36px rgba(99,102,241,0.55);
}
.cta-primary::before {
    content: '';
    position: absolute;
    top: 0; left: -100%;
    width: 100%; height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.15), transparent);
    transition: 0.6s ease;
}
.cta-primary:hover::before {
    left: 100%;
}
.cta-secondary {
    padding: 16px 36px;
    font-size: 1rem;
    font-weight: 600;
    color: #cbd5e1;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 14px;
    cursor: pointer;
    transition: all 0.3s ease;
}
.cta-secondary:hover {
    color: #fff;
    border-color: rgba(99,102,241,0.5);
    background: rgba(99,102,241,0.08);
    transform: translateY(-3px);
}

/* ΓÇöΓÇöΓÇö Floating collab badges (Real-Time Collaborative Animation) ΓÇöΓÇöΓÇö */
.collab-avatars {
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 0;
    margin-top: 48px;
    animation: fade-in-up 0.8s ease 0.5s both;
}
.collab-avatar {
    width: 40px; height: 40px;
    border-radius: 50%;
    border: 3px solid #0a0e1a;
    margin-left: -10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.75rem;
    font-weight: 700;
    color: white;
    animation: avatar-pop 0.5s ease both;
    position: relative;
}
.collab-avatar:nth-child(1) { background: #6366f1; animation-delay: 0.6s; margin-left: 0; }
.collab-avatar:nth-child(2) { background: #22c55e; animation-delay: 0.7s; }
.collab-avatar:nth-child(3) { background: #f59e0b; animation-delay: 0.8s; }
.collab-avatar:nth-child(4) { background: #f43f5e; animation-delay: 0.9s; }
.collab-avatar:nth-child(5) { background: #06b6d4; animation-delay: 1.0s; }
@keyframes avatar-pop {
    0% { transform: scale(0) rotate(-20deg); opacity: 0; }
    100% { transform: scale(1) rotate(0); opacity: 1; }
}
.collab-text {
    margin-left: 14px;
    font-size: 0.88rem;
    color: #94a3b8;
    animation: fade-in-up 0.8s ease 1.1s both;
}
.collab-text strong { color: #22c55e; }

/* ΓÇöΓÇöΓÇö Live typing indicator ΓÇöΓÇöΓÇö */
.typing-indicator {
    display: inline-flex; gap: 4px;
    margin-left: 8px;
}
.typing-indicator span {
    width: 5px; height: 5px;
    background: #22c55e;
    border-radius: 50%;
    animation: typing-bounce 1.4s ease-in-out infinite;
}
.typing-indicator span:nth-child(2) { animation-delay: 0.2s; }
.typing-indicator span:nth-child(3) { animation-delay: 0.4s; }
@keyframes typing-bounce {
    0%, 60%, 100% { transform: translateY(0); }
    30% { transform: translateY(-6px); }
}

/* ΓÇöΓÇöΓÇö FEATURES SECTION ΓÇöΓÇöΓÇö */
.section-title-area {
    text-align: center;
    margin-bottom: 48px;
}
.section-tag {
    display: inline-block;
    padding: 6px 16px;
    background: rgba(99,102,241,0.1);
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 50px;
    font-size: 0.78rem;
    font-weight: 700;
    color: #818cf8;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 16px;
}
.section-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.8rem;
    font-weight: 700;
    color: #f1f5f9;
    letter-spacing: -1px;
    margin-bottom: 12px;
}
.section-desc {
    font-size: 1.1rem;
    color: #94a3b8;
    max-width: 550px;
    margin: 0 auto;
}

/* ΓÇöΓÇöΓÇö Feature Grid ΓÇöΓÇöΓÇö */
.features-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 20px;
    margin-bottom: 100px;
}
.feature-card {
    background: rgba(17, 24, 39, 0.6);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 20px;
    padding: 32px 28px;
    position: relative;
    overflow: hidden;
    transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    cursor: pointer;
}
.feature-card:hover {
    transform: translateY(-8px);
    border-color: var(--glow-color, rgba(99,102,241,0.4));
    box-shadow: 0 20px 60px rgba(0,0,0,0.3), 0 0 40px var(--glow-color-dim, rgba(99,102,241,0.1));
}
.feature-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 100%; height: 100%;
    background: radial-gradient(600px circle at var(--mouse-x, 50%) var(--mouse-y, 50%), var(--glow-color-dim, rgba(99,102,241,0.06)), transparent 40%);
    opacity: 0;
    transition: opacity 0.4s ease;
}
.feature-card:hover::before {
    opacity: 1;
}
.feature-icon {
    width: 56px; height: 56px;
    border-radius: 16px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.6rem;
    margin-bottom: 20px;
    position: relative;
    z-index: 1;
    transition: transform 0.3s ease;
}
.feature-card:hover .feature-icon {
    transform: scale(1.1) rotate(-3deg);
}
.feature-card h3 {
    font-size: 1.15rem;
    font-weight: 700;
    color: #f1f5f9;
    margin-bottom: 10px;
    position: relative;
    z-index: 1;
}
.feature-card p {
    font-size: 0.9rem;
    color: #94a3b8;
    line-height: 1.65;
    position: relative;
    z-index: 1;
}
.feature-card .feature-arrow {
    position: absolute;
    bottom: 24px; right: 24px;
    width: 32px; height: 32px;
    border-radius: 50%;
    background: rgba(255,255,255,0.04);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.9rem;
    color: #64748b;
    transition: all 0.3s ease;
    z-index: 1;
}
.feature-card:hover .feature-arrow {
    background: var(--glow-color, rgba(99,102,241,0.3));
    color: white;
    transform: translateX(3px);
}

/* ΓÇöΓÇöΓÇö How It Works ΓÇöΓÇöΓÇö */
.steps-section {
    margin-bottom: 100px;
}
.steps-row {
    display: flex;
    gap: 0;
    position: relative;
}
.steps-row::before {
    content: '';
    position: absolute;
    top: 50px;
    left: 15%;
    right: 15%;
    height: 2px;
    background: linear-gradient(90deg, rgba(99,102,241,0.3), rgba(124,58,237,0.3), rgba(245,158,11,0.3));
    z-index: 0;
}
.step-item {
    flex: 1;
    text-align: center;
    padding: 0 20px;
    position: relative;
    z-index: 1;
}
.step-num {
    width: 64px; height: 64px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.4rem;
    font-weight: 800;
    color: white;
    margin: 0 auto 20px;
    position: relative;
    transition: transform 0.3s ease;
}
.step-item:hover .step-num {
    transform: scale(1.12);
}
.step-num::after {
    content: '';
    position: absolute;
    width: 80px; height: 80px;
    border-radius: 50%;
    border: 2px dashed rgba(255,255,255,0.08);
    animation: spin-slow 20s linear infinite;
}
@keyframes spin-slow {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}
.step-item h3 {
    font-size: 1.1rem;
    font-weight: 700;
    color: #f1f5f9;
    margin-bottom: 10px;
}
.step-item p {
    font-size: 0.88rem;
    color: #94a3b8;
    line-height: 1.6;
}

/* ΓÇöΓÇöΓÇö Stats / Social Proof ΓÇöΓÇöΓÇö */
.stats-bar {
    background: linear-gradient(135deg, rgba(99,102,241,0.1), rgba(124,58,237,0.06));
    border: 1px solid rgba(99,102,241,0.15);
    border-radius: 24px;
    padding: 48px 40px;
    display: flex;
    justify-content: space-around;
    align-items: center;
    margin-bottom: 100px;
    position: relative;
    overflow: hidden;
}
.stats-bar::before {
    content: '';
    position: absolute;
    top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: conic-gradient(from 0deg, transparent, rgba(99,102,241,0.05), transparent, rgba(124,58,237,0.05), transparent);
    animation: spin-slow 25s linear infinite;
}
.stat-item {
    text-align: center;
    position: relative;
    z-index: 1;
}
.stat-num {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.8rem;
    font-weight: 700;
    background: linear-gradient(135deg, var(--stat-color-1, #6366f1), var(--stat-color-2, #818cf8));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1;
    margin-bottom: 8px;
}
.stat-label {
    font-size: 0.82rem;
    color: #94a3b8;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1.5px;
}

/* ΓÇöΓÇöΓÇö Testimonials ΓÇöΓÇöΓÇö */
.testimonials-section {
    margin-bottom: 100px;
}
.testimonials-track {
    display: flex;
    gap: 20px;
    animation: scroll-left 30s linear infinite;
}
.testimonials-track:hover {
    animation-play-state: paused;
}
@keyframes scroll-left {
    0% { transform: translateX(0); }
    100% { transform: translateX(-50%); }
}
.testimonial-card {
    min-width: 350px;
    background: rgba(17, 24, 39, 0.6);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 20px;
    padding: 28px;
    flex-shrink: 0;
    transition: all 0.3s ease;
}
.testimonial-card:hover {
    border-color: rgba(99,102,241,0.3);
    transform: translateY(-4px);
}
.testimonial-stars {
    color: #f59e0b;
    font-size: 0.9rem;
    margin-bottom: 14px;
    letter-spacing: 2px;
}
.testimonial-text {
    font-size: 0.92rem;
    color: #cbd5e1;
    line-height: 1.7;
    margin-bottom: 18px;
    font-style: italic;
}
.testimonial-author {
    display: flex;
    align-items: center;
    gap: 12px;
}
.testimonial-avatar {
    width: 38px; height: 38px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.8rem;
    font-weight: 700;
    color: white;
}
.testimonial-name {
    font-size: 0.88rem;
    font-weight: 600;
    color: #f1f5f9;
}
.testimonial-role {
    font-size: 0.78rem;
    color: #64748b;
}

/* ΓÇöΓÇöΓÇö Interactive Demo Section ΓÇöΓÇöΓÇö */
.demo-section {
    margin-bottom: 100px;
    display: flex;
    gap: 60px;
    align-items: center;
}
.demo-content { flex: 1; }
.demo-visual {
    flex: 1;
    position: relative;
    height: 400px;
}
.demo-content h2 {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.4rem;
    font-weight: 700;
    color: #f1f5f9;
    letter-spacing: -1px;
    margin-bottom: 18px;
}
.demo-content p {
    font-size: 1.05rem;
    color: #94a3b8;
    line-height: 1.7;
    margin-bottom: 28px;
}
.demo-features {
    list-style: none;
    display: flex;
    flex-direction: column;
    gap: 14px;
}
.demo-features li {
    display: flex;
    align-items: center;
    gap: 12px;
    font-size: 0.95rem;
    color: #cbd5e1;
    padding: 10px 16px;
    background: rgba(17,24,39,0.4);
    border: 1px solid rgba(255,255,255,0.04);
    border-radius: 12px;
    transition: all 0.3s ease;
}
.demo-features li:hover {
    border-color: rgba(99,102,241,0.3);
    background: rgba(99,102,241,0.06);
    transform: translateX(6px);
}
.demo-features li .check-icon {
    width: 24px; height: 24px;
    background: rgba(34,197,94,0.15);
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #22c55e;
    font-size: 0.75rem;
    flex-shrink: 0;
}

/* ΓÇöΓÇöΓÇö Mock Dashboard Preview ΓÇöΓÇöΓÇö */
.mock-dashboard {
    position: absolute;
    top: 0; left: 0;
    width: 100%; height: 100%;
    background: rgba(17, 24, 39, 0.7);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 20px;
    overflow: hidden;
    box-shadow: 0 20px 60px rgba(0,0,0,0.4);
    animation: float-preview 6s ease-in-out infinite;
}
@keyframes float-preview {
    0%, 100% { transform: translateY(0) rotate(1deg); }
    50% { transform: translateY(-12px) rotate(-1deg); }
}
.mock-titlebar {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 14px 18px;
    background: rgba(0,0,0,0.3);
    border-bottom: 1px solid rgba(255,255,255,0.06);
}
.mock-dot {
    width: 10px; height: 10px;
    border-radius: 50%;
}
.mock-content {
    padding: 20px;
}
.mock-score-circle {
    width: 100px; height: 100px;
    border-radius: 50%;
    border: 4px solid rgba(99,102,241,0.3);
    display: flex;
    align-items: center;
    justify-content: center;
    margin: 10px auto 16px;
    position: relative;
}
.mock-score-circle::after {
    content: '';
    position: absolute;
    width: 100%; height: 100%;
    border-radius: 50%;
    border: 4px solid transparent;
    border-top-color: #6366f1;
    border-right-color: #818cf8;
    animation: spin-slow 3s linear infinite;
}
.mock-score-val {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    color: #818cf8;
}
.mock-bars {
    display: flex; flex-direction: column; gap: 10px;
    margin-top: 16px;
}
.mock-bar-label {
    font-size: 0.72rem;
    color: #64748b;
    margin-bottom: 3px;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-weight: 600;
}
.mock-bar-track {
    height: 6px;
    background: rgba(255,255,255,0.06);
    border-radius: 10px;
    overflow: hidden;
}
.mock-bar-fill {
    height: 100%;
    border-radius: 10px;
    animation: bar-grow 2s ease forwards;
}
@keyframes bar-grow {
    0% { width: 0%; }
}

/* ΓÇöΓÇöΓÇö Final CTA Section ΓÇöΓÇöΓÇö */
.final-cta {
    text-align: center;
    padding: 80px 20px;
    position: relative;
    margin-bottom: 40px;
}
.final-cta h2 {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 3rem;
    font-weight: 700;
    color: #f1f5f9;
    letter-spacing: -1.5px;
    margin-bottom: 16px;
}
.final-cta p {
    font-size: 1.1rem;
    color: #94a3b8;
    margin-bottom: 36px;
}
.final-cta-glow {
    position: absolute;
    width: 400px; height: 400px;
    background: radial-gradient(circle, rgba(99,102,241,0.15), transparent);
    top: 50%; left: 50%;
    transform: translate(-50%, -50%);
    border-radius: 50%;
    filter: blur(60px);
    pointer-events: none;
}

/* ΓÇöΓÇöΓÇö Footer ΓÇöΓÇöΓÇö */
.page-footer {
    text-align: center;
    padding: 40px 0;
    border-top: 1px solid rgba(255,255,255,0.04);
    color: #475569;
    font-size: 0.82rem;
}
.page-footer a {
    color: #6366f1;
    text-decoration: none;
}

/* ΓÇöΓÇöΓÇö Scroll Reveal ΓÇöΓÇöΓÇö */
.reveal {
    opacity: 0;
    transform: translateY(40px);
    transition: all 0.8s cubic-bezier(0.16, 1, 0.3, 1);
}
.reveal.visible {
    opacity: 1;
    transform: translateY(0);
}

/* ΓÇöΓÇöΓÇö Mobile ΓÇöΓÇöΓÇö */
@media (max-width: 768px) {
    .features-grid { grid-template-columns: 1fr; }
    .hero-title { font-size: 2.5rem; }
    .demo-section { flex-direction: column; }
    .steps-row { flex-direction: column; gap: 30px; }
    .steps-row::before { display: none; }
    .stats-bar { flex-direction: column; gap: 30px; }
    .nav-center { display: none; }
    .testimonials-track { animation-duration: 15s; }
}
</style>
</head>
<body>

<!-- Loading Screen -->
<div class="loading-screen" id="loadingScreen">
    <div class="loading-logo">CF</div>
    <div class="loading-bar-track">
        <div class="loading-bar-fill"></div>
    </div>
    <div class="loading-text">Forging your experience</div>
</div>

<!-- Custom Cursor -->
<div class="cursor-dot" id="cursorDot"></div>
<div class="cursor-ring" id="cursorRing"></div>
<div class="cursor-glow" id="cursorGlow"></div>

<!-- Background -->
<div class="bg-canvas">
    <div class="bg-orb"></div>
    <div class="bg-orb"></div>
    <div class="bg-orb"></div>
    <div class="bg-orb"></div>
</div>
<div class="grid-overlay"></div>

<!-- Main Content -->
<div class="main-content">

    <!-- TOP BAR -->
    <div class="top-bar">
        <a class="brand-link" onclick="navigateTo('landing')">
            <div class="brand-logo">CF</div>
            <div class="brand-name">CareerForge</div>
        </a>
        <div class="nav-center">
            <a class="nav-link" href="#features">Features</a>
            <a class="nav-link" href="#how-it-works">How It Works</a>
            <a class="nav-link" href="#demo">Live Demo</a>
            <a class="nav-link" href="#testimonials">Testimonials</a>
        </div>
        <div class="auth-btns">
            <button class="btn-signin" onclick="navigateTo('signin')">Sign In</button>
            <button class="btn-signup" onclick="navigateTo('signup')">Sign Up</button>
        </div>
    </div>

    <!-- HERO -->
    <div class="hero-section">
        <div class="hero-badge">
            <span class="pulse-dot"></span>
            Trusted by 500+ students
        </div>
        <h1 class="hero-title">
            <span class="line1">Forge Your</span>
            <span class="line2">Career Path</span>
            <span class="line3">with AI</span>
        </h1>
        <p class="hero-subtitle">
            AI-powered career coaching that adapts to you. Master interviews, build killer resumes,
            crush DSA, and land your dream job ΓÇö all in one platform.
        </p>
        <div class="hero-ctas">
            <button class="cta-primary" onclick="navigateTo('signup')">
                ≡ƒÜÇ Get Started Free
            </button>
            <button class="cta-secondary" onclick="navigateTo('signin')">
                Sign In ΓåÆ
            </button>
        </div>

        <!-- Real-Time Collaborative Animation -->
        <div class="collab-avatars">
            <div class="collab-avatar">AK</div>
            <div class="collab-avatar">PR</div>
            <div class="collab-avatar">SM</div>
            <div class="collab-avatar">VK</div>
            <div class="collab-avatar">+5</div>
            <span class="collab-text">
                <strong>12 students</strong> active now
                <span class="typing-indicator">
                    <span></span><span></span><span></span>
                </span>
            </span>
        </div>
    </div>

    <!-- FEATURES -->
    <div id="features" class="reveal">
        <div class="section-title-area">
            <div class="section-tag">Features</div>
            <h2 class="section-title">Everything You Need</h2>
            <p class="section-desc">Powerful AI-driven tools built for students and job seekers who want results.</p>
        </div>
        <div class="features-grid">
            <div class="feature-card" style="--glow-color: rgba(99,102,241,0.4); --glow-color-dim: rgba(99,102,241,0.08);">
                <div class="feature-icon" style="background: rgba(99,102,241,0.12);">≡ƒÅá</div>
                <h3>Smart Dashboard</h3>
                <p>Track placement readiness, DSA progress, aptitude score, and daily streak ΓÇö all at a glance.</p>
                <div class="feature-arrow">ΓåÆ</div>
            </div>
            <div class="feature-card" style="--glow-color: rgba(34,197,94,0.4); --glow-color-dim: rgba(34,197,94,0.08);">
                <div class="feature-icon" style="background: rgba(34,197,94,0.12);">≡ƒôä</div>
                <h3>Resume Analyzer</h3>
                <p>Upload your resume, get instant AI feedback ΓÇö strengths, weaknesses, missing skills, and tips.</p>
                <div class="feature-arrow">ΓåÆ</div>
            </div>
            <div class="feature-card" style="--glow-color: rgba(245,158,11,0.4); --glow-color-dim: rgba(245,158,11,0.08);">
                <div class="feature-icon" style="background: rgba(245,158,11,0.12);">≡ƒôè</div>
                <h3>Placement Predictor</h3>
                <p>AI predicts your placement probability and gives actionable steps to boost your score.</p>
                <div class="feature-arrow">ΓåÆ</div>
            </div>
            <div class="feature-card" style="--glow-color: rgba(167,139,250,0.4); --glow-color-dim: rgba(167,139,250,0.08);">
                <div class="feature-icon" style="background: rgba(167,139,250,0.12);">≡ƒºá</div>
                <h3>Adaptive Learning</h3>
                <p>Personalized 7-day sprints targeting your weak areas with curated tasks and problems.</p>
                <div class="feature-arrow">ΓåÆ</div>
            </div>
            <div class="feature-card" style="--glow-color: rgba(56,189,248,0.4); --glow-color-dim: rgba(56,189,248,0.08);">
                <div class="feature-icon" style="background: rgba(56,189,248,0.12);">≡ƒñû</div>
                <h3>AI Mentor Chat</h3>
                <p>Chat with your AI career coach for advice on interviews, DSA strategies, and more.</p>
                <div class="feature-arrow">ΓåÆ</div>
            </div>
            <div class="feature-card" style="--glow-color: rgba(244,63,94,0.4); --glow-color-dim: rgba(244,63,94,0.08);">
                <div class="feature-icon" style="background: rgba(244,63,94,0.12);">≡ƒÄñ</div>
                <h3>Mock Interviews</h3>
                <p>Practice with AI-generated technical questions, scored on correctness, clarity, and depth.</p>
                <div class="feature-arrow">ΓåÆ</div>
            </div>
        </div>
    </div>

    <!-- HOW IT WORKS -->
    <div id="how-it-works" class="steps-section reveal">
        <div class="section-title-area">
            <div class="section-tag">How It Works</div>
            <h2 class="section-title">3 Simple Steps</h2>
            <p class="section-desc">From zero to placement-ready in record time.</p>
        </div>
        <div class="steps-row">
            <div class="step-item">
                <div class="step-num" style="background: linear-gradient(135deg, #6366f1, #818cf8); box-shadow: 0 6px 24px rgba(99,102,241,0.3);">1</div>
                <h3>Create Your Profile</h3>
                <p>Sign up and tell us your career goal ΓÇö SDE, Data Science, Product, or more.</p>
            </div>
            <div class="step-item">
                <div class="step-num" style="background: linear-gradient(135deg, #22c55e, #4ade80); box-shadow: 0 6px 24px rgba(34,197,94,0.3);">2</div>
                <h3>Get AI Insights</h3>
                <p>Our AI analyzes your skills, resume, and progress to create a personalized plan.</p>
            </div>
            <div class="step-item">
                <div class="step-num" style="background: linear-gradient(135deg, #f59e0b, #fbbf24); box-shadow: 0 6px 24px rgba(245,158,11,0.3);">3</div>
                <h3>Land Your Dream Job</h3>
                <p>Follow your adaptive learning path, practice interviews, watch your score climb.</p>
            </div>
        </div>
    </div>

    <!-- INTERACTIVE DEMO -->
    <div id="demo" class="demo-section reveal">
        <div class="demo-content">
            <div class="section-tag">Live Preview</div>
            <h2>See It In Action</h2>
            <p>Your personalized dashboard gives you real-time insights into every aspect of your career preparation journey.</p>
            <ul class="demo-features">
                <li>
                    <span class="check-icon">Γ£ô</span>
                    Real-time placement score tracking
                </li>
                <li>
                    <span class="check-icon">Γ£ô</span>
                    AI-generated personalized study plans
                </li>
                <li>
                    <span class="check-icon">Γ£ô</span>
                    Mock interview with instant feedback
                </li>
                <li>
                    <span class="check-icon">Γ£ô</span>
                    Resume analysis against industry standards
                </li>
                <li>
                    <span class="check-icon">Γ£ô</span>
                    Progress tracking with visual analytics
                </li>
            </ul>
        </div>
        <div class="demo-visual">
            <div class="mock-dashboard">
                <div class="mock-titlebar">
                    <div class="mock-dot" style="background:#ef4444;"></div>
                    <div class="mock-dot" style="background:#f59e0b;"></div>
                    <div class="mock-dot" style="background:#22c55e;"></div>
                    <span style="margin-left: 12px; font-size: 0.75rem; color: #64748b;">CareerForge Dashboard</span>
                </div>
                <div class="mock-content">
                    <div style="text-align:center; font-size:0.75rem; color:#64748b; text-transform:uppercase; letter-spacing:1.5px; font-weight:600;">Placement Score</div>
                    <div class="mock-score-circle">
                        <span class="mock-score-val" id="mockScore">0</span>
                    </div>
                    <div class="mock-bars">
                        <div>
                            <div class="mock-bar-label">DSA Progress</div>
                            <div class="mock-bar-track"><div class="mock-bar-fill" style="width:72%; background:linear-gradient(90deg, #6366f1, #818cf8);"></div></div>
                        </div>
                        <div>
                            <div class="mock-bar-label">Aptitude</div>
                            <div class="mock-bar-track"><div class="mock-bar-fill" style="width:85%; background:linear-gradient(90deg, #22c55e, #4ade80);"></div></div>
                        </div>
                        <div>
                            <div class="mock-bar-label">Interview Ready</div>
                            <div class="mock-bar-track"><div class="mock-bar-fill" style="width:58%; background:linear-gradient(90deg, #f59e0b, #fbbf24);"></div></div>
                        </div>
                        <div>
                            <div class="mock-bar-label">Resume Score</div>
                            <div class="mock-bar-track"><div class="mock-bar-fill" style="width:65%; background:linear-gradient(90deg, #f43f5e, #fb7185);"></div></div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <!-- STATS BAR -->
    <div class="stats-bar reveal">
        <div class="stat-item">
            <div class="stat-num counter" data-target="500" style="--stat-color-1:#6366f1;--stat-color-2:#818cf8;">0+</div>
            <div class="stat-label">Active Students</div>
        </div>
        <div class="stat-item">
            <div class="stat-num counter" data-target="1200" style="--stat-color-1:#22c55e;--stat-color-2:#4ade80;">0+</div>
            <div class="stat-label">Mock Interviews</div>
        </div>
        <div class="stat-item">
            <div class="stat-num counter" data-target="85" style="--stat-color-1:#f59e0b;--stat-color-2:#fbbf24;">0%</div>
            <div class="stat-label">Placement Rate</div>
        </div>
        <div class="stat-item">
            <div class="stat-num" style="--stat-color-1:#a78bfa;--stat-color-2:#c4b5fd;">100%</div>
            <div class="stat-label">Free Forever</div>
        </div>
    </div>

    <!-- TESTIMONIALS -->
    <div id="testimonials" class="testimonials-section reveal">
        <div class="section-title-area">
            <div class="section-tag">Testimonials</div>
            <h2 class="section-title">Loved by Students</h2>
            <p class="section-desc">See what our users have to say about their CareerForge experience.</p>
        </div>
        <div style="overflow:hidden;">
            <div class="testimonials-track">
                <div class="testimonial-card">
                    <div class="testimonial-stars">ΓÿàΓÿàΓÿàΓÿàΓÿà</div>
                    <div class="testimonial-text">"CareerForge helped me crack my Google interview. The adaptive learning path targeted exactly what I needed."</div>
                    <div class="testimonial-author">
                        <div class="testimonial-avatar" style="background:linear-gradient(135deg,#6366f1,#818cf8);">AK</div>
                        <div>
                            <div class="testimonial-name">Arjun Kumar</div>
                            <div class="testimonial-role">SDE @ Google</div>
                        </div>
                    </div>
                </div>
                <div class="testimonial-card">
                    <div class="testimonial-stars">ΓÿàΓÿàΓÿàΓÿàΓÿà</div>
                    <div class="testimonial-text">"The mock interview feature is incredible. Real-time feedback on my answers boosted my confidence 10x."</div>
                    <div class="testimonial-author">
                        <div class="testimonial-avatar" style="background:linear-gradient(135deg,#22c55e,#4ade80);">PR</div>
                        <div>
                            <div class="testimonial-name">Priya Reddy</div>
                            <div class="testimonial-role">SDE @ Microsoft</div>
                        </div>
                    </div>
                </div>
                <div class="testimonial-card">
                    <div class="testimonial-stars">ΓÿàΓÿàΓÿàΓÿàΓÿà</div>
                    <div class="testimonial-text">"From 45% to 88% placement score in 3 weeks. The AI mentor chat is like having a personal career coach 24/7."</div>
                    <div class="testimonial-author">
                        <div class="testimonial-avatar" style="background:linear-gradient(135deg,#f59e0b,#fbbf24);">SM</div>
                        <div>
                            <div class="testimonial-name">Sneha Mehta</div>
                            <div class="testimonial-role">Data Analyst @ Amazon</div>
                        </div>
                    </div>
                </div>
                <div class="testimonial-card">
                    <div class="testimonial-stars">ΓÿàΓÿàΓÿàΓÿàΓÿà</div>
                    <div class="testimonial-text">"Best career prep tool I've used. The resume analyzer found gaps I never noticed and helped me fix them fast."</div>
                    <div class="testimonial-author">
                        <div class="testimonial-avatar" style="background:linear-gradient(135deg,#f43f5e,#fb7185);">VK</div>
                        <div>
                            <div class="testimonial-name">Vikram Krishna</div>
                            <div class="testimonial-role">SDE @ Flipkart</div>
                        </div>
                    </div>
                </div>
                <div class="testimonial-card">
                    <div class="testimonial-stars">ΓÿàΓÿàΓÿàΓÿàΓÿà</div>
                    <div class="testimonial-text">"The consistency tracking and daily streaks kept me motivated throughout my placement prep season."</div>
                    <div class="testimonial-author">
                        <div class="testimonial-avatar" style="background:linear-gradient(135deg,#06b6d4,#22d3ee);">NP</div>
                        <div>
                            <div class="testimonial-name">Nisha Patel</div>
                            <div class="testimonial-role">SDE @ Razorpay</div>
                        </div>
                    </div>
                </div>
                <!-- Duplicate for seamless loop -->
                <div class="testimonial-card">
                    <div class="testimonial-stars">ΓÿàΓÿàΓÿàΓÿàΓÿà</div>
                    <div class="testimonial-text">"CareerForge helped me crack my Google interview. The adaptive learning path targeted exactly what I needed."</div>
                    <div class="testimonial-author">
                        <div class="testimonial-avatar" style="background:linear-gradient(135deg,#6366f1,#818cf8);">AK</div>
                        <div>
                            <div class="testimonial-name">Arjun Kumar</div>
                            <div class="testimonial-role">SDE @ Google</div>
                        </div>
                    </div>
                </div>
                <div class="testimonial-card">
                    <div class="testimonial-stars">ΓÿàΓÿàΓÿàΓÿàΓÿà</div>
                    <div class="testimonial-text">"The mock interview feature is incredible. Real-time feedback on my answers boosted my confidence 10x."</div>
                    <div class="testimonial-author">
                        <div class="testimonial-avatar" style="background:linear-gradient(135deg,#22c55e,#4ade80);">PR</div>
                        <div>
                            <div class="testimonial-name">Priya Reddy</div>
                            <div class="testimonial-role">SDE @ Microsoft</div>
                        </div>
                    </div>
                </div>
                <div class="testimonial-card">
                    <div class="testimonial-stars">ΓÿàΓÿàΓÿàΓÿàΓÿà</div>
                    <div class="testimonial-text">"From 45% to 88% placement score in 3 weeks. The AI mentor chat is like having a personal career coach 24/7."</div>
                    <div class="testimonial-author">
                        <div class="testimonial-avatar" style="background:linear-gradient(135deg,#f59e0b,#fbbf24);">SM</div>
                        <div>
                            <div class="testimonial-name">Sneha Mehta</div>
                            <div class="testimonial-role">Data Analyst @ Amazon</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <!-- FINAL CTA -->
    <div class="final-cta reveal">
        <div class="final-cta-glow"></div>
        <h2>Ready to Forge Your Future?</h2>
        <p>Join hundreds of students already using CareerForge to land their dream jobs.</p>
        <button class="cta-primary" style="font-size: 1.1rem; padding: 18px 44px;" onclick="navigateTo('signup')">
            ≡ƒöÑ Start Your Journey Now
        </button>
    </div>

    <!-- FOOTER -->
    <div class="page-footer">
        Built with Γ¥ñ∩╕Å by Team Renegades  ΓÇó  Hack AI 2.0
    </div>
</div>

<script>
// ΓÇöΓÇöΓÇö Loading Screen ΓÇöΓÇöΓÇö
setTimeout(() => {
    document.getElementById('loadingScreen').classList.add('hidden');
}, 2200);

// ΓÇöΓÇöΓÇö Custom Cursor ΓÇöΓÇöΓÇö
const dot = document.getElementById('cursorDot');
const ring = document.getElementById('cursorRing');
const cursorGlow = document.getElementById('cursorGlow');
const bgOrbs = document.querySelectorAll('.bg-orb');
const gridOverlay = document.querySelector('.grid-overlay');
let mouseX = 0, mouseY = 0;
let ringX = 0, ringY = 0;
let glowX = 0, glowY = 0;
let lastSparkleTime = 0;

document.addEventListener('mousemove', (e) => {
    mouseX = e.clientX;
    mouseY = e.clientY;
    dot.style.left = mouseX - 5 + 'px';
    dot.style.top = mouseY - 5 + 'px';

    // Star sparkle trail removed per request
    // const now = performance.now();
    // if (now - lastSparkleTime > 40) {
    //     createSparkle(mouseX, mouseY);
    //     lastSparkleTime = now;
    // }

    // ΓÇöΓÇöΓÇö Background Glow follows cursor ΓÇöΓÇöΓÇö
    cursorGlow.style.left = mouseX + 'px';
    cursorGlow.style.top = mouseY + 'px';

    // ΓÇöΓÇöΓÇö Background Orbs bend toward cursor ΓÇöΓÇöΓÇö
    const cx = (mouseX / window.innerWidth - 0.5) * 2;
    const cy = (mouseY / window.innerHeight - 0.5) * 2;
    bgOrbs.forEach((orb, i) => {
        const intensity = [40, 30, 25, 20][i] || 20;
        const offsetX = cx * intensity;
        const offsetY = cy * intensity;
        orb.style.transform = `translate(${offsetX}px, ${offsetY}px)`;
    });

    // ΓÇöΓÇöΓÇö Grid lines bend near cursor ΓÇöΓÇöΓÇö
    const gridShiftX = cx * 8;
    const gridShiftY = cy * 8;
    gridOverlay.style.backgroundPosition = `${gridShiftX}px ${gridShiftY}px`;
});

// ΓÇöΓÇöΓÇö Ring follows cursor at high FPS with faster lerp ΓÇöΓÇöΓÇö
function animateRing() {
    ringX += (mouseX - ringX) * 0.35;
    ringY += (mouseY - ringY) * 0.35;
    ring.style.left = ringX - 20 + 'px';
    ring.style.top = ringY - 20 + 'px';

    // Smooth glow follow
    glowX += (mouseX - glowX) * 0.08;
    glowY += (mouseY - glowY) * 0.08;

    requestAnimationFrame(animateRing);
}
animateRing();

// Cursor hover effects
document.querySelectorAll('button, a, .feature-card, .nav-link, .testimonial-card').forEach(el => {
    el.addEventListener('mouseenter', () => {
        dot.style.transform = 'scale(2.5)';
        dot.style.background = 'radial-gradient(circle, #fff 0%, #a78bfa 60%, transparent 100%)';
        ring.style.width = '56px';
        ring.style.height = '56px';
        ring.style.borderColor = 'rgba(167,139,250,0.7)';
        ring.style.boxShadow = '0 0 20px rgba(167,139,250,0.3)';
    });
    el.addEventListener('mouseleave', () => {
        dot.style.transform = 'scale(1)';
        dot.style.background = 'radial-gradient(circle, #fff 0%, #818cf8 60%, transparent 100%)';
        ring.style.width = '40px';
        ring.style.height = '40px';
        ring.style.borderColor = 'rgba(129,140,248,0.5)';
        ring.style.boxShadow = '0 0 12px rgba(129,140,248,0.15)';
    });
});

// ΓÇöΓÇöΓÇö Star Sparkle Particles ΓÇöΓÇöΓÇö
function createSparkle(x, y) {
    const sparkle = document.createElement('div');
    sparkle.className = 'sparkle';
    const size = Math.random() * 8 + 4;
    const colors = ['#818cf8', '#a78bfa', '#6366f1', '#06b6d4', '#f43f5e', '#c084fc', '#fbbf24', '#fff'];
    const color = colors[Math.floor(Math.random() * colors.length)];
    const tx = (Math.random() - 0.5) * 120;
    const ty = (Math.random() - 0.5) * 120;
    const rotation = Math.random() * 360;

    sparkle.style.cssText = `
        left: ${x}px; top: ${y}px;
        width: ${size}px; height: ${size}px;
        background: ${color};
        box-shadow: 0 0 ${size * 3}px ${color}, 0 0 ${size}px #fff;
        --tx: ${tx}px; --ty: ${ty}px;
        transform: rotate(${rotation}deg);
    `;
    document.body.appendChild(sparkle);
    setTimeout(() => sparkle.remove(), 1000);
}

// ΓÇöΓÇöΓÇö Feature Card Mouse Follow Glow ΓÇöΓÇöΓÇö
document.querySelectorAll('.feature-card').forEach(card => {
    card.addEventListener('mousemove', (e) => {
        const rect = card.getBoundingClientRect();
        const x = ((e.clientX - rect.left) / rect.width) * 100;
        const y = ((e.clientY - rect.top) / rect.height) * 100;
        card.style.setProperty('--mouse-x', x + '%');
        card.style.setProperty('--mouse-y', y + '%');
    });
});

// ΓÇöΓÇöΓÇö Scroll Reveal ΓÇöΓÇöΓÇö
function revealOnScroll() {
    document.querySelectorAll('.reveal').forEach(el => {
        const rect = el.getBoundingClientRect();
        if (rect.top < window.innerHeight - 80) {
            el.classList.add('visible');
        }
    });
}
window.addEventListener('scroll', revealOnScroll);
setTimeout(revealOnScroll, 100);

// ΓÇöΓÇöΓÇö Counter Animation ΓÇöΓÇöΓÇö
let countersAnimated = false;
function animateCounters() {
    if (countersAnimated) return;
    const counters = document.querySelectorAll('.counter');
    counters.forEach(counter => {
        const rect = counter.getBoundingClientRect();
        if (rect.top < window.innerHeight) {
            countersAnimated = true;
            const target = parseInt(counter.dataset.target);
            const suffix = counter.textContent.includes('%') ? '%' : '+';
            let current = 0;
            const step = Math.ceil(target / 60);
            const interval = setInterval(() => {
                current += step;
                if (current >= target) {
                    current = target;
                    clearInterval(interval);
                }
                counter.textContent = current.toLocaleString() + suffix;
            }, 25);
        }
    });
}
window.addEventListener('scroll', animateCounters);
setTimeout(animateCounters, 2500);

// ΓÇöΓÇöΓÇö Mock Dashboard Score Counter ΓÇöΓÇöΓÇö
setTimeout(() => {
    const scoreEl = document.getElementById('mockScore');
    let val = 0;
    const interval = setInterval(() => {
        val += 1;
        if (val >= 78) { val = 78; clearInterval(interval); }
        scoreEl.textContent = val + '%';
    }, 35);
}, 2500);

// ΓÇöΓÇöΓÇö Smooth scroll for nav links ΓÇöΓÇöΓÇö
document.querySelectorAll('.nav-link').forEach(link => {
    link.addEventListener('click', (e) => {
        e.preventDefault();
        const targetId = link.getAttribute('href').substring(1);
        const targetEl = document.getElementById(targetId);
        if (targetEl) {
            targetEl.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
    });
});
// ΓÇöΓÇöΓÇö Navigation helper (direct parent URL change) ΓÇöΓÇöΓÇö
function navigateTo(page) {
    try {
        // Try direct parent location change (works if same-origin)
        window.parent.location.href = window.parent.location.pathname + '?page=' + page;
    } catch(e) {
        // Fallback: try top-level navigation
        try {
            window.top.location.href = '/?page=' + page;
        } catch(e2) {
            // Last resort: postMessage
            window.parent.postMessage({type: 'streamlit_navigate', page: page}, '*');
        }
    }
}
</script>
</body>
</html>
    """
    
    # Write the HTML file
    landing_path = os.path.join(static_dir, "landing.html")
    with open(landing_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    # Embed via raw iframe without sandbox restrictions
    st.markdown(
        '<iframe src="/app/static/landing.html" '
        'width="100%" height="4000" frameborder="0" '
        'style="border:none; display:block;" '
        'allow="scripts"></iframe>',
        unsafe_allow_html=True
    )


# -----------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------
def main():
    load_css()
    init_state()

    if not GEMINI_OK:
        st.warning("API key not configured. AI features won't work.")

    # Initialize page routing
    if "show_page" not in st.session_state:
        st.session_state.show_page = "landing"

    # Read navigation from URL query params
    qp = st.query_params
    if "page" in qp:
        requested = qp["page"]
        if requested in ("signin", "signup", "landing"):
            st.session_state.show_page = requested
            st.query_params.clear()
            st.rerun()

    # If authenticated, force to dashboard
    if st.session_state.authenticated:
        st.session_state.show_page = "dashboard"

    # Route to the correct page
    page = st.session_state.show_page

    if page == "landing":
        render_landing_page()
    elif page == "signin":
        render_signin_page()
    elif page == "signup":
        render_signup_page()
    elif page == "dashboard":
        if not st.session_state.authenticated:
            st.session_state.show_page = "landing"
            st.rerun()
            return

        render_authenticated_navbar()

        tabs = st.tabs([
            "🏠 Dashboard",
            "📄 Resume",
            "📊 Predictor",
            "🧠 Learning",
            "🤖 Mentor",
            "🎤 Interview",
            "📈 Progress",
        ])

        with tabs[0]:
            render_dashboard()
        with tabs[1]:
            render_resume_analyzer()
        with tabs[2]:
            render_placement_predictor()
        with tabs[3]:
            render_adaptive_learning()
        with tabs[4]:
            render_ai_mentor()
        with tabs[5]:
            render_mock_interview()
        with tabs[6]:
            render_progress_tracking()


if __name__ == "__main__":
    main()
