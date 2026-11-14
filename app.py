import os
from pathlib import Path
from datetime import datetime

import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# ============================================================
# SalaryIQ — ChatGPT-inspired dark workspace
# Preserves the existing US workflows and adds optional India
# model/data support. Gemini chat uses ai_agent.py if installed.
# ============================================================
st.set_page_config(page_title="SalaryIQ | Salary Intelligence", page_icon="✦", layout="wide", initial_sidebar_state="expanded")
BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
MODELS = BASE / "models"
US_MODEL_PATH = MODELS / "best_salary_model.pkl"
US_DATA_PATH = DATA / "cleaned_salary_data.csv"
US_RESULTS_PATH = MODELS / "model_results.csv"
IN_MODEL_PATH = MODELS / "india_salary_model_v2.pkl"
IN_MODEL_V1_PATH = MODELS / "india_salary_model.pkl"
IN_DATA_PATH = DATA / "india_salary_cleaned.csv"
IN_RAW_PATH = DATA / "india_salary_data.csv"
IN_RESULTS_PATH = MODELS / "india_model_results_v2.csv"
IN_RESULTS_V1_PATH = MODELS / "india_model_results.csv"

st.markdown(r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Manrope:wght@500;600;700;800&display=swap');
:root{--bg:#101114;--panel:#191b22;--line:#2c2f39;--text:#f4f4f5;--muted:#a1a1aa;--purple:#a78bfa}
html,body,[class*="css"]{font-family:'DM Sans',sans-serif}
.stApp{background:radial-gradient(ellipse at 75% 0%,rgba(124,58,237,.13),transparent 36%),radial-gradient(ellipse at 5% 75%,rgba(59,130,246,.06),transparent 35%),#101114;color:var(--text)}
[data-testid="stHeader"]{background:rgba(16,17,20,.75)}
[data-testid="stSidebar"]{background:#15161b;border-right:1px solid #292b33}
[data-testid="stSidebar"] .block-container{padding:1.1rem .9rem}
.block-container{max-width:1450px;padding-top:2rem;padding-bottom:3rem}
h1,h2,h3{font-family:'Manrope',sans-serif!important;letter-spacing:-.035em}
.brand{display:flex;align-items:center;gap:10px;margin:5px 0 2px}.brand-icon{width:36px;height:36px;border-radius:12px;background:linear-gradient(135deg,#8b5cf6,#4f46e5);display:grid;place-items:center;color:white;font-size:20px}.brand-name{font:800 23px Manrope;color:#fafafa;letter-spacing:-1px}.brand-sub{margin:2px 0 24px 46px;color:#858895;font-size:9px;letter-spacing:2px}
.kicker{color:#a78bfa;font-size:10px;font-weight:800;letter-spacing:2px;text-transform:uppercase;margin-bottom:8px}.page-title{font:800 32px Manrope;color:#fafafa;letter-spacing:-1.2px;margin:0 0 5px}.subtitle{color:#a1a1aa;font-size:14px;margin-bottom:24px}
.welcome{text-align:center;padding:42px 10px 28px}.orb{width:68px;height:68px;margin:0 auto 20px;border-radius:22px;display:grid;place-items:center;font-size:32px;color:white;background:linear-gradient(140deg,#8b5cf6,#4f46e5 65%,#2563eb);box-shadow:0 0 55px rgba(139,92,246,.25)}.welcome h1{font-size:34px!important;color:#fafafa!important;margin-bottom:8px}.welcome p{max-width:650px;margin:auto;color:#a1a1aa;font-size:14px;line-height:1.7}
.metric{background:linear-gradient(145deg,#1d1f27,#17191f);border:1px solid #2b2e38;border-radius:16px;padding:18px;min-height:115px}.metric-label{color:#a1a1aa;font-size:12px}.metric-value{color:#fafafa;font-size:25px;font-weight:800;margin-top:12px}.metric-note{color:#777b88;font-size:10px;margin-top:4px}
.result{background:linear-gradient(135deg,rgba(91,33,182,.35),rgba(30,41,89,.35));border:1px solid rgba(167,139,250,.45);border-radius:20px;padding:26px;text-align:center;margin:18px 0}.result-label{color:#c4b5fd;font-size:10px;font-weight:800;letter-spacing:2px}.result-value{color:white;font:800 40px Manrope;margin:8px 0}
.chat-user{background:#292b35;border:1px solid #383b47;border-radius:18px 18px 5px 18px;padding:12px 16px;margin:8px 0 8px auto;max-width:82%;color:#f4f4f5}.chat-ai{background:rgba(28,30,38,.95);border:1px solid #30333e;border-radius:18px 18px 18px 5px;padding:14px 17px;margin:8px auto 14px 0;max-width:94%;color:#e4e4e7}.chat-role{color:#a78bfa;font-size:10px;font-weight:800;letter-spacing:1px;margin-bottom:6px}
.pill{display:inline-block;padding:5px 9px;border-radius:999px;background:rgba(139,92,246,.14);color:#c4b5fd;border:1px solid rgba(167,139,250,.24);font-size:10px;font-weight:700;margin:2px 3px 2px 0}
.side-label{color:#777b88;font-size:10px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;margin:18px 0 8px}
div.stButton>button,div.stDownloadButton>button{border-radius:11px!important;border:1px solid #59418d!important;background:linear-gradient(110deg,#7042bd,#4f46a5)!important;color:white!important;font-weight:700!important;min-height:40px}div.stButton>button:hover,div.stDownloadButton>button:hover{border-color:#a78bfa!important;filter:brightness(1.12)}
div[data-baseweb="select"]>div,.stTextInput input,.stNumberInput input,.stTextArea textarea{background:#1b1d25!important;color:#f4f4f5!important;border-color:#363945!important;border-radius:10px!important}
[data-testid="stDataFrame"]{border:1px solid #30333e;border-radius:12px;overflow:hidden}[data-testid="stTabs"] button{color:#a1a1aa}[data-testid="stTabs"] button[aria-selected="true"]{color:#c4b5fd}hr{border-color:#2b2e38!important}.footer{text-align:center;color:#696d78;font-size:10px;padding:26px 0 4px}
</style>
""", unsafe_allow_html=True)

# ---------- Utility functions ----------
def header(kicker, title, subtitle):
    st.markdown(f'<div class="kicker">{kicker}</div><div class="page-title">{title}</div><div class="subtitle">{subtitle}</div>', unsafe_allow_html=True)

def metric(label, value, note=""):
    st.markdown(f'<div class="metric"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>', unsafe_allow_html=True)

def chart(fig, height=360):
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#d4d4d8",family="DM Sans"), margin=dict(l=12,r=12,t=55,b=12),height=height,legend=dict(bgcolor="rgba(0,0,0,0)"))
    fig.update_xaxes(gridcolor="#292c35",zerolinecolor="#292c35"); fig.update_yaxes(gridcolor="#292c35",zerolinecolor="#292c35")
    return fig

@st.cache_resource
def load_pickle(path): return joblib.load(path)
@st.cache_data
def load_csv(path): return pd.read_csv(path)
def optional_csv(path):
    try: return load_csv(str(path)) if path.exists() else pd.DataFrame()
    except Exception: return pd.DataFrame()
def optional_model(path):
    try: return load_pickle(str(path)) if path.exists() else None
    except Exception as exc:
        st.sidebar.warning(f"Could not load {path.name}: {exc}"); return None
def add_history(item):
    st.session_state.prediction_history.insert(0,item); st.session_state.prediction_history=st.session_state.prediction_history[:30]

# ---------- Load existing datasets and models ----------
us_df=optional_csv(US_DATA_PATH); us_results=optional_csv(US_RESULTS_PATH)
india_df=optional_csv(IN_DATA_PATH)
if india_df.empty: india_df=optional_csv(IN_RAW_PATH)
india_results=optional_csv(IN_RESULTS_PATH)
if india_results.empty: india_results=optional_csv(IN_RESULTS_V1_PATH)
us_model=optional_model(US_MODEL_PATH)
india_model=optional_model(IN_MODEL_PATH)
india_model_v2=india_model is not None
if india_model is None:
    india_model=optional_model(IN_MODEL_V1_PATH)

# ---------- Session state ----------
for key,default in [("page","New Chat"),("prediction_history",[]),("chat_messages",[]),("pending_prompt","")]:
    if key not in st.session_state: st.session_state[key]=default

# ---------- Sidebar navigation ----------
with st.sidebar:
    st.markdown('<div class="brand"><div class="brand-icon">✦</div><div class="brand-name">SalaryIQ</div></div><div class="brand-sub">SALARY INTELLIGENCE</div>',unsafe_allow_html=True)
    if st.button("＋  New chat",use_container_width=True,key="new_chat_sidebar"):
        st.session_state.chat_messages=[]; st.session_state.pending_prompt=""; st.session_state.page="New Chat"; st.experimental_rerun()
    st.markdown('<div class="side-label">Workspace</div>',unsafe_allow_html=True)
    for icon,label in [("◫","Dashboard"),("⌁","Salary Predictor"),("▥","Analytics"),("⇧","Bulk Prediction"),("◈","Model Insights"),("✦","AI Assistant")]:
        if st.button(("▸ " if st.session_state.page==label else "")+icon+"  "+label,use_container_width=True,key="nav_"+label):
            st.session_state.page=label; st.experimental_rerun()
    st.markdown('<div class="side-label">Recent predictions</div>',unsafe_allow_html=True)
    if st.session_state.prediction_history:
        for item in st.session_state.prediction_history[:5]:
            symbol="₹" if item["currency"]=="INR" else "$"
            st.caption(f'{item["country"]} · {item["job_title"][:22]} · {symbol}{item["salary"]:,.0f}')
    else: st.caption("Your salary estimates will appear here.")
    st.markdown("---")
    st.markdown('<div class="side-label">System status</div>',unsafe_allow_html=True)
    st.markdown(f'<span class="pill">US model {"ready" if us_model is not None else "missing"}</span><span class="pill">India model {"ready" if india_model is not None else "missing"}</span>',unsafe_allow_html=True)
    st.caption(f"US records: {len(us_df):,}" if not us_df.empty else "US dataset not found")
    st.caption(f"India records: {len(india_df):,}" if not india_df.empty else "India dataset not found")
    st.markdown("---"); st.caption("SalaryIQ · ML + AI")

page=st.session_state.page

# ============================================================
# NEW CHAT — ChatGPT-style launch screen
# ============================================================
if page=="New Chat":
    st.markdown('<div class="welcome"><div class="orb">✦</div><h1>Where should we start?</h1><p>Explore compensation insights, estimate salaries, or ask your AI career assistant. Your existing SalaryIQ datasets and prediction models power this workspace.</p></div>',unsafe_allow_html=True)
    prompts=[("Estimate a salary","Predict a software engineer salary in Hyderabad with 4-6 years of experience."),("Explore analytics","Show me salary trends by department."),("Understand model performance","Explain the model evaluation metrics."),("Career guidance","What skills should I develop for a data science career?")]
    cols=st.columns(2)
    for i,(title,prompt) in enumerate(prompts):
        with cols[i%2]:
            with st.container():
                st.markdown(f"**{title}**"); st.caption(prompt)
                if st.button("Start →",key=f"home_{i}",use_container_width=True):
                    st.session_state.page="AI Assistant"; st.session_state.pending_prompt=prompt; st.experimental_rerun()

# ============================================================
# DASHBOARD — existing US metrics + India data overview
# ============================================================
elif page=="Dashboard":
    header("OVERVIEW","Your salary intelligence hub","A unified view of your available workforce data and prediction systems.")
    if not us_df.empty and "Salary" in us_df:
        c=st.columns(4)
        with c[0]: metric("US employee records",f"{len(us_df):,}","Cleaned US dataset")
        with c[1]: metric("Average US salary",f"${us_df.Salary.mean():,.0f}","Source dataset currency")
        with c[2]: metric("Average experience",f"{us_df.Experience_Years.mean():.1f} yrs" if "Experience_Years" in us_df else "—","US dataset")
        with c[3]: metric("Departments",str(us_df.Department.nunique()) if "Department" in us_df else "—","US dataset categories")
    else: st.info("US dashboard metrics are unavailable: cleaned_salary_data.csv was not found.")
    if not india_df.empty and "salary_mid_inr" in india_df:
        c=st.columns(3)
        with c[0]: metric("India job postings",f"{len(india_df):,}","Indian benchmark dataset")
        with c[1]: metric("Average India midpoint",f"₹{india_df.salary_mid_inr.mean():,.0f}","Annual INR midpoint")
        with c[2]: metric("Median India midpoint",f"₹{india_df.salary_mid_inr.median():,.0f}","Annual INR midpoint")
    if not us_df.empty:
        st.markdown("### Workforce overview · US dataset")
        a,b=st.columns(2)
        with a:
            if {"Department","Salary"}.issubset(us_df.columns):
                d=us_df.groupby("Department",as_index=False).Salary.mean().sort_values("Salary")
                st.plotly_chart(chart(px.bar(d,x="Salary",y="Department",orientation="h",color="Salary",color_continuous_scale="Purples",title="Average salary by department")),use_container_width=True)
        with b:
            if {"Location","Salary"}.issubset(us_df.columns):
                d=us_df.groupby("Location",as_index=False).Salary.mean().sort_values("Salary",ascending=False).head(12)
                st.plotly_chart(chart(px.bar(d,x="Location",y="Salary",color="Salary",color_continuous_scale="Teal",title="Average salary by location")),use_container_width=True)
        with st.expander("Preview US employee records"): st.dataframe(us_df.head(20),use_container_width=True)
    if not india_df.empty:
        with st.expander("Preview Indian salary records"): st.dataframe(india_df.head(20),use_container_width=True)

# ============================================================
# SALARY PREDICTOR — country-specific models and currencies
# ============================================================
elif page=="Salary Predictor":
    header("PREDICTION STUDIO","Estimate an annual salary","Select the market to use its own trained model. Indian and US predictions remain separate.")
    market=st.radio("Prediction market",["India","US"],horizontal=True,key="predict_market")
    if market=="India":
        if india_model is None: st.warning("India model not found. Train and save models/india_salary_model_v2.pkl first.")
        else:
            role_values=sorted(india_df.role_family.dropna().astype(str).unique()) if "role_family" in india_df else ["software-eng","data","marketing","finance","sales","other"]
            city_values=sorted(india_df.city.dropna().astype(str).unique()) if "city" in india_df else ["Hyderabad","Bengaluru","Mumbai","Pune","Chennai","Delhi"]
            title_values=sorted(india_df.job_title.dropna().astype(str).unique()) if "job_title" in india_df else ["Software Engineer"]
            with st.container():
                st.markdown("#### Indian salary estimate")
                with st.form("india_salary_form"):
                    a,b=st.columns(2)
                    with a:
                        job=st.selectbox("Job title (reference)",title_values)
                        role=st.selectbox("Role family",role_values)
                        seniority=st.selectbox("Seniority",["intern","entry","mid","senior","manager","unknown"],index=2)
                    with b:
                        band=st.selectbox("Experience band",["0-1 yrs","2-3 yrs","4-6 yrs","7-10 yrs","11-15 yrs","16+ yrs"],index=2)
                        city=st.selectbox("City",city_values,index=city_values.index("Hyderabad") if "Hyderabad" in city_values else 0)
                    go=st.form_submit_button("✦ Generate India estimate",use_container_width=True)
            if go:
                try:
                    if india_model_v2:
                        x=pd.DataFrame([{"role_family":role,"seniority":seniority,"experience_band":band,"city":city}])
                    else:
                        x=pd.DataFrame([{"job_title":job,"role_family":role,"seniority":seniority,"experience_band":band,"city":city}])
                    value=float(india_model.predict(x)[0])
                    st.markdown(f'<div class="result"><div class="result-label">ESTIMATED ANNUAL SALARY · INDIA</div><div class="result-value">₹{value:,.0f}</div><div class="muted">INR · Model estimate, not a guaranteed offer</div></div>',unsafe_allow_html=True)
                    item={"country":"India","job_title":job,"salary":value,"currency":"INR","date":datetime.now().isoformat(timespec="minutes")}; st.session_state.last_prediction=item; add_history(item)
                except Exception as e: st.error(f"India prediction failed: {e}")
    else:
        if us_model is None or us_df.empty: st.warning("US model or cleaned US dataset is missing. Check the models and data folders.")
        else:
            with st.container():
                st.markdown("#### US salary estimate")
                with st.form("us_salary_form"):
                    a,b=st.columns(2)
                    with a:
                        department=st.selectbox("Department",sorted(us_df.Department.dropna().unique()))
                        titles=sorted(us_df.loc[us_df.Department==department,"Job_Title"].dropna().unique())
                        title=st.selectbox("Job title",titles or sorted(us_df.Job_Title.dropna().unique()))
                        years=st.number_input("Years of experience",min_value=0.0,max_value=50.0,value=5.0,step=.5)
                    with b:
                        education=st.selectbox("Education level",sorted(us_df.Education_Level.dropna().unique()))
                        location=st.selectbox("Location",sorted(us_df.Location.dropna().unique()))
                    go=st.form_submit_button("✦ Generate US estimate",use_container_width=True)
            if go:
                try:
                    x=pd.DataFrame([{"Experience_Years":years,"Department":department,"Job_Title":title,"Education_Level":education,"Location":location}])
                    value=float(us_model.predict(x)[0])
                    st.markdown(f'<div class="result"><div class="result-label">ESTIMATED ANNUAL SALARY · UNITED STATES</div><div class="result-value">${value:,.0f}</div><div class="muted">USD · Model estimate, not a guaranteed offer</div></div>',unsafe_allow_html=True)
                    item={"country":"US","job_title":title,"salary":value,"currency":"USD","date":datetime.now().isoformat(timespec="minutes")}; st.session_state.last_prediction=item; add_history(item)
                except Exception as e: st.error(f"US prediction failed: {e}")

# ============================================================
# ANALYTICS — preserve existing US charts and add India charts
# ============================================================
elif page=="Analytics":
    header("DATA EXPLORER","Compensation analytics","Explore each source independently. Salary values are shown in their source currencies.")
    market=st.radio("Dataset",["US employee dataset","India salary postings"],horizontal=True)
    if market=="US employee dataset":
        if us_df.empty: st.warning("US dataset is unavailable.")
        else:
            f=us_df.copy()
            if "Department" in f:
                choices=sorted(f.Department.dropna().unique()); selected=st.multiselect("Departments",choices,default=choices); f=f[f.Department.isin(selected)]
            if "Location" in f:
                choices=sorted(f.Location.dropna().unique()); selected=st.multiselect("Locations",choices,default=choices); f=f[f.Location.isin(selected)]
            if f.empty: st.warning("No records match the filters.")
            else:
                c=st.columns(3)
                with c[0]: metric("Records",f"{len(f):,}")
                with c[1]: metric("Average salary",f"${f.Salary.mean():,.0f}")
                with c[2]: metric("Median salary",f"${f.Salary.median():,.0f}")
                t1,t2,t3=st.tabs(["Salary distribution","Experience analysis","Education analysis"])
                with t1: st.plotly_chart(chart(px.histogram(f,x="Salary",color="Department",nbins=35,title="Salary distribution")),use_container_width=True)
                with t2: st.plotly_chart(chart(px.scatter(f,x="Experience_Years",y="Salary",color="Department",hover_data=["Job_Title","Location"],title="Experience vs salary")),use_container_width=True)
                with t3:
                    d=f.groupby("Education_Level",as_index=False).Salary.mean(); st.plotly_chart(chart(px.bar(d,x="Education_Level",y="Salary",color="Education_Level",title="Average salary by education")),use_container_width=True)
    else:
        if india_df.empty or "salary_mid_inr" not in india_df: st.warning("India salary dataset is unavailable.")
        else:
            f=india_df.copy()
            if "city" in f:
                cities=sorted(f.city.dropna().astype(str).unique()); selected=st.multiselect("Cities",cities,default=cities[:min(12,len(cities))]); f=f[f.city.astype(str).isin(selected)]
            if f.empty: st.warning("No records match the filters.")
            else:
                c=st.columns(3)
                with c[0]: metric("Job postings",f"{len(f):,}")
                with c[1]: metric("Average midpoint",f"₹{f.salary_mid_inr.mean():,.0f}")
                with c[2]: metric("Median midpoint",f"₹{f.salary_mid_inr.median():,.0f}")
                t1,t2=st.tabs(["Salary by city","Salary by role family"])
                with t1:
                    d=f.groupby("city",as_index=False).salary_mid_inr.median().sort_values("salary_mid_inr",ascending=False).head(20); st.plotly_chart(chart(px.bar(d,x="city",y="salary_mid_inr",title="Median salary midpoint by city")),use_container_width=True)
                with t2:
                    if "role_family" in f:
                        d=f.groupby("role_family",as_index=False).salary_mid_inr.median().sort_values("salary_mid_inr",ascending=False); st.plotly_chart(chart(px.bar(d,x="role_family",y="salary_mid_inr",title="Median salary midpoint by role family")),use_container_width=True)

# ============================================================
# BULK PREDICTION — existing US CSV flow + India CSV flow
# ============================================================
elif page=="Bulk Prediction":
    header("BATCH STUDIO","Bulk salary prediction","Upload a CSV to generate estimates for multiple employee or job profiles.")
    market=st.radio("Prediction market",["US","India"],horizontal=True,key="bulk_market")
    if market=="US":
        if us_model is None: st.warning("US model is unavailable.")
        else:
            required=["Experience_Years","Department","Job_Title","Education_Level","Location"]
            st.info("Required columns: "+", ".join(required))
            file=st.file_uploader("Upload US employee CSV",type=["csv"],key="us_bulk")
            if file:
                try:
                    batch=pd.read_csv(file); st.dataframe(batch.head(10),use_container_width=True); missing=[x for x in required if x not in batch]
                    if missing: st.error("Missing columns: "+", ".join(missing))
                    elif st.button("Generate US bulk predictions",key="run_us_bulk"):
                        out=batch.copy(); out["Predicted_Salary_USD"]=us_model.predict(batch[required]); st.success(f"Generated {len(out):,} predictions."); st.dataframe(out,use_container_width=True); st.download_button("Download predictions CSV",out.to_csv(index=False).encode(),"salary_predictions_us.csv","text/csv")
                except Exception as e: st.error(f"CSV processing failed: {e}")
    else:
        if india_model is None: st.warning("India model is unavailable.")
        else:
            required=["role_family","seniority","experience_band","city"] if india_model_v2 else ["job_title","role_family","seniority","experience_band","city"]
            st.info("Required columns: "+", ".join(required))
            file=st.file_uploader("Upload India job profile CSV",type=["csv"],key="india_bulk")
            if file:
                try:
                    batch=pd.read_csv(file); st.dataframe(batch.head(10),use_container_width=True); missing=[x for x in required if x not in batch]
                    if missing: st.error("Missing columns: "+", ".join(missing))
                    elif st.button("Generate India bulk predictions",key="run_india_bulk"):
                        out=batch.copy(); out["Predicted_Salary_INR"]=india_model.predict(batch[required]); st.success(f"Generated {len(out):,} predictions."); st.dataframe(out,use_container_width=True); st.download_button("Download predictions CSV",out.to_csv(index=False).encode(),"salary_predictions_india.csv","text/csv")
                except Exception as e: st.error(f"CSV processing failed: {e}")

# ============================================================
# MODEL INSIGHTS — retain evaluation tables and charts
# ============================================================
elif page=="Model Insights":
    header("MODEL LAB","Model performance & insights","Review evaluation results for the independent US and India prediction models.")
    t1,t2=st.tabs(["US model","India model"])
    with t1:
        st.markdown("#### US model evaluation")
        if not us_results.empty:
            st.dataframe(us_results,use_container_width=True)
            col="R2 Score" if "R2 Score" in us_results else ("R2_Score" if "R2_Score" in us_results else None)
            if col and "Model" in us_results: st.plotly_chart(chart(px.bar(us_results,x="Model",y=col,color="Model",title="US model R² comparison")),use_container_width=True)
        else: st.info("US model_results.csv is not available.")
        st.caption("Features: Experience_Years, Department, Job_Title, Education_Level, Location. Target: Salary.")
    with t2:
        st.markdown("#### India model evaluation")
        if not india_results.empty:
            st.dataframe(india_results,use_container_width=True)
            col="R2_Score" if "R2_Score" in india_results else ("R2 Score" if "R2 Score" in india_results else None)
            if col and "Model" in india_results: st.plotly_chart(chart(px.bar(india_results,x="Model",y=col,color="Model",title="India model R² comparison")),use_container_width=True)
        else: st.info("India model evaluation CSV is not available.")
        st.caption("India v2 features: role_family, seniority, experience_band, city. Target: salary_mid_inr (INR).")
    st.warning("The India and US datasets differ in source, features and salary distributions. Their metrics should not be compared as if they were measured on the same data.")

# ============================================================
# AI ASSISTANT — optional Gemini integration via ai_agent.py
# ============================================================
elif page=="AI Assistant":
    header("SALARYIQ COPILOT","Ask SalaryIQ","Chat about salary estimates, workforce insights and career questions.")
    def ask_agent(prompt,history):
        try:
            from ai_agent import chat_with_salaryiq
            return chat_with_salaryiq(prompt,history)
        except ImportError:
            return "Gemini integration is not connected yet. Add ai_agent.py, install google-genai and python-dotenv, and configure GEMINI_API_KEY in your .env file."
        except Exception as e: return f"AI assistant error: {e}"
    if st.button("＋ New conversation",key="new_ai_conversation"):
        st.session_state.chat_messages=[]; st.session_state.pending_prompt=""; st.experimental_rerun()
    if not st.session_state.chat_messages:
        st.markdown('<div class="welcome"><div class="orb">✦</div><h1 style="font-size:29px!important">How can I help today?</h1><p>Ask a question in your own words. SalaryIQ can use its connected salary models when the required details are provided.</p></div>',unsafe_allow_html=True)
        prompts=["Estimate a software engineer salary in Hyderabad with 4-6 years of experience.","Predict a data analyst salary in New York with 3 years of experience.","Explain the difference between MAE, RMSE and R².","What skills are useful for a machine learning engineer?"]
        cols=st.columns(2)
        for i,prompt in enumerate(prompts):
            with cols[i%2]:
                if st.button(prompt,key=f"ai_prompt_{i}",use_container_width=True): st.session_state.pending_prompt=prompt; st.experimental_rerun()
    for msg in st.session_state.chat_messages:
        if msg["role"]=="user": st.markdown(f'<div class="chat-user"><div class="chat-role">YOU</div>{msg["content"]}</div>',unsafe_allow_html=True)
        else: st.markdown(f'<div class="chat-ai"><div class="chat-role">✦ SALARYIQ</div>{msg["content"]}</div>',unsafe_allow_html=True)
    pending=st.session_state.pending_prompt
    with st.form("salaryiq_chat_form",clear_on_submit=True):
        prompt=st.text_input("Message SalaryIQ",value=pending,placeholder="Ask about salary, skills, experience or market trends...")
        send=st.form_submit_button("Send message  ↑",use_container_width=True)
    if send and prompt.strip():
        st.session_state.pending_prompt=""; history=list(st.session_state.chat_messages); st.session_state.chat_messages.append({"role":"user","content":prompt})
        with st.spinner("SalaryIQ is thinking..."): answer=ask_agent(prompt,history)
        st.session_state.chat_messages.append({"role":"assistant","content":answer}); st.experimental_rerun()

st.markdown('<div class="footer">✦ SALARYIQ · SALARY INTELLIGENCE WORKSPACE<br><br>Machine Learning · Workforce Analytics · Gemini AI (optional)</div>',unsafe_allow_html=True)
