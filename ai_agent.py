"""
SalaryIQ Gemini AI Agent
------------------------
Connects the Streamlit chat UI to Gemini and to the locally trained
India / US salary prediction pipelines.

Required packages:
    python -m pip install -U google-genai python-dotenv pandas joblib

Required .env:
    GEMINI_API_KEY=your_key
    GEMINI_MODEL=gemini-2.5-flash

The AI agent never invents a model prediction: salary figures are produced
by the saved scikit-learn pipelines and then explained by Gemini.
"""

import json
import os
import re
from pathlib import Path

import joblib
import pandas as pd
from dotenv import load_dotenv
from google import genai

# Resolve project paths relative to this file, not the terminal directory.
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

INDIA_MODEL_V2_PATH = MODEL_DIR / "india_salary_model_v2.pkl"
INDIA_MODEL_V1_PATH = MODEL_DIR / "india_salary_model.pkl"
US_MODEL_PATH = MODEL_DIR / "best_salary_model.pkl"

load_dotenv(BASE_DIR / ".env")

API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.8-flash").strip()


# Keep one long-lived Gemini client for the lifetime of this Python process.
# Creating a temporary client inline can leave the SDK transport unavailable
# during Streamlit reruns or garbage collection.
_GEMINI_CLIENT = None


def _client():
    global _GEMINI_CLIENT

    if not API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY is missing. Add it to the .env file beside app.py."
        )

    if _GEMINI_CLIENT is None:
        _GEMINI_CLIENT = genai.Client(api_key=API_KEY)

    return _GEMINI_CLIENT


def _generate_content(**kwargs):
    """Send a request through the persistent client without closing it."""
    client = _client()
    return client.models.generate_content(**kwargs)


def _history_text(history):
    """Keep recent conversation concise and avoid sending excess chat history."""
    clean = []
    for item in (history or [])[-12:]:
        role = item.get("role", "")
        content = str(item.get("content", "")).strip()
        if role in ("user", "assistant") and content:
            clean.append({"role": role, "content": content[:2500]})
    return json.dumps(clean, ensure_ascii=False)


def _load_json_response(response):
    text = (getattr(response, "text", None) or "").strip()
    # Defensive extraction in case a model returns fenced JSON.
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Gemini returned an unreadable request. Please rephrase.") from exc
    if not isinstance(value, dict):
        raise RuntimeError("Gemini returned an invalid request format.")
    return value


def _extract_request(user_message, history):
    """
    Parse the latest message plus prior chat into structured fields.
    The model is instructed to preserve known details from the conversation.
    """
    prompt = f"""
You are the structured request parser for SalaryIQ.
Return ONLY a JSON object. Do not include Markdown.

The user wants either a salary prediction or general career/salary help.
Use the conversation history to carry forward details already supplied.
Do not invent missing values.

Return exactly these keys:
{{
  "intent": "prediction" or "general",
  "country": "India" or "US" or "unknown",
  "job_title": "",
  "role_family": "",
  "seniority": "",
  "experience_band": "",
  "city": "",
  "department": "",
  "experience_years": null,
  "education_level": "",
  "location": ""
}}

Rules:
- Set intent to prediction only when the user asks for an estimated salary,
  salary prediction, salary calculation, or compensation estimate.
- If the user asks a general question, set intent to general.
- Infer India when the location is an Indian city, and US for a US location.
- Do not infer a country from a job title alone.
- For India, role_family should be one of the closest matching dataset labels.
  Common examples include software-eng, data, marketing, finance, sales, other.
  Use an exact label from the dataset if the user explicitly supplies one.
- For India, seniority should be one of:
  intern, entry, mid, senior, manager, unknown.
- For India, map experience to one of:
  0-1 yrs, 2-3 yrs, 4-6 yrs, 7-10 yrs, 11-15 yrs, 16+ yrs.
- For US, capture department, job_title, experience_years, education_level,
  and location. Do not guess department or education.
- Preserve the user's actual city/location wording.
- If a value is not known, use "" or null.
- Never output explanatory text outside JSON.

Conversation history:
{_history_text(history)}

Latest user message:
{user_message}
"""
    response = _generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config={"response_mime_type": "application/json"},
    )
    return _load_json_response(response)


def _missing_fields(data):
    country = data.get("country")
    if country == "India":
        required = ["role_family", "seniority", "experience_band", "city"]
    elif country == "US":
        required = [
            "department",
            "job_title",
            "experience_years",
            "education_level",
            "location",
        ]
    else:
        return ["country"]

    missing = []
    for field in required:
        value = data.get(field)
        if value is None or (isinstance(value, str) and not value.strip()):
            missing.append(field)
    return missing


def _missing_question(fields):
    questions = {
        "country": "Are you estimating a salary in India or the United States?",
        "role_family": "What is the employee's role family? For example, software engineering, data, marketing or finance.",
        "seniority": "What is the seniority level: intern, entry, mid, senior or manager?",
        "experience_band": "How much experience does the employee have? For example, 4-6 years.",
        "city": "Which Indian city should I use?",
        "department": "Which department should I use? Please choose the closest department from your US dataset.",
        "job_title": "What is the exact job title?",
        "experience_years": "How many years of experience does the employee have?",
        "education_level": "What is the education level? Please use a category from your US dataset.",
        "location": "Which US location should I use?",
    }
    if len(fields) == 1:
        return questions[fields[0]]
    return "I need a few more details before predicting the salary:\n\n" + "\n".join(
        f"- {questions[field]}" for field in fields
    )


def _load_model(path):
    if not path.exists():
        raise FileNotFoundError(f"Required model file was not found: {path.name}")
    return joblib.load(path)


def _india_prediction(data):
    # Prefer v2: role_family, seniority, experience_band, city.
    if INDIA_MODEL_V2_PATH.exists():
        model = _load_model(INDIA_MODEL_V2_PATH)
        features = pd.DataFrame([{
            "role_family": str(data["role_family"]),
            "seniority": str(data["seniority"]),
            "experience_band": str(data["experience_band"]),
            "city": str(data["city"]),
        }])
    elif INDIA_MODEL_V1_PATH.exists():
        # v1 was trained with job_title as an additional feature.
        model = _load_model(INDIA_MODEL_V1_PATH)
        if not data.get("job_title"):
            raise ValueError(
                "The available India model is version 1 and requires a job title. "
                "Please provide the exact job title."
            )
        features = pd.DataFrame([{
            "job_title": str(data["job_title"]),
            "role_family": str(data["role_family"]),
            "seniority": str(data["seniority"]),
            "experience_band": str(data["experience_band"]),
            "city": str(data["city"]),
        }])
    else:
        raise FileNotFoundError(
            "No Indian model found. Expected models/india_salary_model_v2.pkl "
            "or models/india_salary_model.pkl."
        )

    prediction = float(model.predict(features)[0])
    return max(0.0, prediction)


def _us_prediction(data):
    model = _load_model(US_MODEL_PATH)
    features = pd.DataFrame([{
        "Department": str(data["department"]),
        "Job_Title": str(data["job_title"]),
        "Experience_Years": float(data["experience_years"]),
        "Education_Level": str(data["education_level"]),
        "Location": str(data["location"]),
    }])
    prediction = float(model.predict(features)[0])
    return max(0.0, prediction)


def _general_answer(user_message, history):
    prompt = f"""
You are SalaryIQ, a practical, professional salary and career coach.
Answer the latest message directly, using the recent conversation to retain context.

You can help with:
- explaining salary components, annual vs monthly pay, CTC, base pay, bonus and benefits;
- career planning, skill development, interview preparation and promotion discussions;
- salary negotiation, including respectful scripts and evidence the user can gather;
- explaining how SalaryIQ's local prediction models work and their limitations.

Accuracy and safety rules:
- Do not claim to have searched live salary market data or current job listings.
- Do not invent a personalized model prediction, salary range, confidence score, or market statistic.
- If asked for a personalized prediction, invite the user to say "predict my salary" and provide country, role, experience and location.
- Clearly distinguish general career guidance from a prediction produced by SalaryIQ's saved ML model.
- Avoid guarantees. Give practical, actionable steps and explain assumptions when needed.
- Use concise headings or bullets when they improve readability; avoid generic introductions.

Recent conversation:
{_history_text(history)}

Latest user message:
{user_message}
"""
    response = _generate_content(
        model=MODEL_NAME,
        contents=prompt,
    )
    return (getattr(response, "text", None) or "").strip() or (
        "I couldn't generate a response just now. Please try again."
    )


def _is_prediction_turn(user_message, history):
    """Avoid a JSON-parser API call for ordinary chat; retain prediction follow-ups."""
    text = str(user_message).lower()
    prediction_terms = (
        "predict", "prediction", "estimate my salary", "salary estimate",
        "calculate my salary", "how much salary", "expected salary",
        "salary for", "compensation estimate",
    )
    if any(term in text for term in prediction_terms):
        return True

    # A short reply such as "Hyderabad" may answer a question asked during
    # an unfinished prediction flow. Only parse it when the recent context
    # clearly contains that flow.
    recent = (history or [])[-6:]
    assistant_asked_for_details = any(
        item.get("role") == "assistant"
        and any(word in str(item.get("content", "")).lower()
                for word in ("which indian city", "how many years", "what is the employee", "need a few more details", "are you estimating a salary"))
        for item in recent
    )
    prior_prediction_request = any(
        item.get("role") == "user"
        and any(term in str(item.get("content", "")).lower()
                for term in ("predict", "prediction", "estimate my salary", "salary estimate", "expected salary"))
        for item in recent
    )
    return assistant_asked_for_details and prior_prediction_request


def generate_ai_response(user_message, history=None):
    """Main entry point used by the SalaryIQ Streamlit application."""
    if not str(user_message).strip():
        return "Please enter a message."

    # General chat now takes one Gemini call instead of first asking Gemini
    # to classify JSON and then making a second call for the actual answer.
    if not _is_prediction_turn(user_message, history):
        return _general_answer(user_message, history)

    data = _extract_request(user_message, history)

    if data.get("intent") != "prediction":
        return _general_answer(user_message, history)

    country = data.get("country")
    if country not in ("India", "US"):
        return _missing_question(["country"])

    missing = _missing_fields(data)
    if missing:
        return _missing_question(missing)

    if country == "India":
        prediction = _india_prediction(data)
        currency = "INR"
        formatted = f"₹{prediction:,.0f}"
        experience = data.get("experience_band", "Not supplied")
        location = data.get("city", "")
    else:
        prediction = _us_prediction(data)
        currency = "USD"
        formatted = f"${prediction:,.0f}"
        experience = f"{data.get('experience_years')} years"
        location = data.get("location", "")

    # The model's numeric output is authoritative. Gemini only explains it.
    explanation_prompt = f"""
You are SalaryIQ. Explain this completed machine-learning prediction.
The prediction value below came from the local saved model and MUST NOT
be changed, recalculated, rounded differently, or contradicted.

Country: {country}
Job title supplied by user: {data.get('job_title') or 'Not supplied'}
Role family: {data.get('role_family') or 'Not supplied'}
Seniority: {data.get('seniority') or 'Not supplied'}
Experience: {experience}
Location: {location}
Predicted annual salary: {formatted}
Currency: {currency}

Respond in a concise, friendly format:
- Show the predicted annual salary exactly as given.
- Summarize the profile used.
- Say this is a model estimate, not a guaranteed offer or verified market quote.
- Mention that actual compensation can vary by employer, skills, industry,
  benefits, and market conditions.
- Do not invent a confidence interval or salary range.
"""
    response = _generate_content(
        model=MODEL_NAME,
        contents=explanation_prompt,
    )
    explanation = (getattr(response, "text", None) or "").strip()

    if not explanation:
        explanation = (
            f"**Estimated annual salary: {formatted}**\n\n"
            f"Market: {country} · Location: {location} · Experience: {experience}\n\n"
            "This is a model estimate, not a guaranteed offer. Actual compensation "
            "varies by employer, skills, industry and market conditions."
        )
    return explanation


def chat_with_salaryiq(user_message, history=None):
    """Safe wrapper for Streamlit: return readable errors instead of crashing."""
    try:
        return generate_ai_response(user_message, history)
    except FileNotFoundError as exc:
        return f"**Model file issue:** {exc}"
    except ValueError as exc:
        return f"**Input issue:** {exc}"
    except Exception as exc:
        error_text = str(exc)
        if "429" in error_text or "RESOURCE_EXHAUSTED" in error_text:
            return (
                "Gemini's current free-tier request or token limit has been reached. "
                "Please wait and try again later, or check your AI Studio quota."
            )
        if "API_KEY_INVALID" in error_text or "401" in error_text:
            return (
                "Gemini API key was rejected. Check GEMINI_API_KEY in your .env file "
                "and create a new key in Google AI Studio if needed."
            )
        if "404" in error_text or "NOT_FOUND" in error_text:
            return (
                f"The configured Gemini model '{MODEL_NAME}' was not found or is not "
                "available to this API key. Set GEMINI_MODEL to a model enabled in "
                "your Google AI Studio project (for example, gemini-3.8-flash)."
            )
        return f"**SalaryIQ AI error:** {error_text}"
