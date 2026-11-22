"""
نبرة (Nabra) - Streamlit prototype
الشاشات: Splash -> Upload -> Analyzing -> Result -> Explanation (XAI)

التشغيل:
    py -m pip install -r requirements.txt
    py -m streamlit run app.py

الموديل:
    nabra_explain.py (WavLM-Large + WPT + AASIST مع تفسير GATR) + أوزانه best.pt في نفس المجلد.
"""

import base64
import io
import os
import sys
import tempfile
import time
from datetime import datetime

import numpy as np
import streamlit as st
import streamlit.components.v1 as components

# ─────────────────────────────── الربط مع الموديل ───────────────────────────────
# الموديل الحقيقي (من عائشة). الاستيراد هنا خفيف، والأوزان تتحمّل في شاشة البداية.
try:
    from nabra_explain import SR, clean_wave, explain, load_model, load_wave, predict
    IMPORT_ERROR = None
except Exception as e:  # مثلًا nabra_explain.py مو موجود أو مكتبة ناقصة (torch / transformers)
    IMPORT_ERROR = f"تعذّر تحميل nabra_explain.py: {e}"


def check_result(res):
    """يتأكد إن ناتج الموديل بالشكل المتفق عليه: label و confidence و score."""
    if res.get("label") not in ("real", "fake"):
        raise ValueError(f"label غير متوقع من الموديل: {res.get('label')}")
    if not 0 <= float(res.get("confidence", -1)) <= 1:
        raise ValueError(f"confidence لازم يكون بين 0 و 1: {res.get('confidence')}")
    return res


def audio_meta(path):
    """المدة ومعدل العينة (sample rate) الحقيقيين من الملف."""
    import soundfile as sf
    info = sf.info(path)
    secs = int(round(info.duration))
    return {
        "duration": f"{secs // 60:02d}:{secs % 60:02d}",
        "format": os.path.splitext(path)[1].lstrip(".").upper(),
    }


# ─────────────────────────────── التصميم ───────────────────────────────
st.set_page_config(page_title="نبرة", page_icon="🎙️", layout="centered")

C = {
    "screen": "#11122b",
    "card": "#1b1c40",
    "border": "#2c2e5c",
    "violet": "#8b7cf6",
    "lavender": "#a99bff",
    "teal": "#3ec9b4",
    "text": "#ffffff",
    "muted": "#8c8db8",
    "fake": "#ff6b6b",
    "real": "#3ddc97",
}

# أيقونات SVG
ICON_WAVE = '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M2 12h3l3-7 4 14 3-9 2 2h5"/></svg>'
ICON_UPLOAD = '<svg width="30" height="30" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 15V4M7 9l5-5 5 5M4 15v4a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-4"/></svg>'
ICON_BARS = '<svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="#0e0f26" stroke-width="2" stroke-linecap="round"><path d="M4 10v4M8 7v10M12 4v16M16 7v10M20 10v4"/></svg>'
ICON_INFO = f'<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="{C["lavender"]}" stroke-width="2" stroke-linecap="round" style="flex:0 0 18px"><circle cx="12" cy="12" r="9"/><path d="M12 11v5M12 8h.01"/></svg>'
ICON_LOCK = '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><rect x="5" y="11" width="14" height="10" rx="2"/><path d="M8 11V7a4 4 0 0 1 8 0v4"/></svg>'
ICON_CHECK = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12l5 5 9-10"/></svg>'

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Arabic:wght@400;500;600;700&display=swap');

html, body, .stApp, button, input, p, div, span {{ font-family: 'IBM Plex Sans Arabic', sans-serif; }}
.stApp {{ background: #ececf4; direction: rtl; }}
header[data-testid="stHeader"], footer, #MainMenu, [data-testid="stToolbar"],
[data-testid="stDecoration"], [data-testid="stStatusWidget"] {{ display: none !important; }}
[data-stale="true"] {{ display: none !important; }}

/* ── إطار iPhone 17 Pro (على اللابتوب): الشاشة 402×874 + إطار 12px ── */
.block-container {{
    box-sizing: border-box !important; flex-shrink: 0 !important;
    width: 426px !important; max-width: 426px !important; height: 898px;
    margin: 28px auto !important; padding: 14px 22px 34px !important;
    position: relative; overflow-y: auto; overflow-x: hidden; scrollbar-width: none;
    border-radius: 56px; border: 12px solid #0a0a0f;
    outline: 3px solid #b9a6cc; outline-offset: 0;
    box-shadow: 0 30px 60px rgba(20,20,60,.25);
    background: radial-gradient(circle at 90% 4%, #2b2a5c 0%, {C['screen']} 40%);
    color: {C['text']};
}}
.block-container::-webkit-scrollbar {{ display: none; }}
.block-container::before {{   /* Dynamic Island */
    content: ""; position: absolute; top: 11px; left: 50%; transform: translateX(-50%);
    width: 126px; height: 37px; border-radius: 20px; background: #000; z-index: 10;
}}
/* على الجوال الحقيقي: بدون إطار */
@media (max-width: 500px) {{
    .stApp {{ background: {C['screen']}; }}
    .block-container {{ width: 100% !important; max-width: 100% !important; height: auto; min-height: 100vh;
        margin: 0 !important; border: 0; outline: 0; border-radius: 0; box-shadow: none; padding: 16px 18px !important; }}
    .block-container::before, .statusbar {{ display: none; }}
}}

.statusbar {{ display:flex; justify-content:space-between; align-items:center; direction: ltr;
    font-size: 15px; font-weight: 600; padding: 4px 16px 0; height: 40px; margin-bottom: 6px; }}
.statusbar .r {{ display:flex; gap:6px; align-items:center; font-size: 13px; }}
.battery {{ width: 24px; height: 12px; border: 1.5px solid #fff; border-radius: 4px; padding: 1.5px; }}
.battery span {{ display:block; height:100%; width:75%; background:#fff; border-radius: 1.5px; }}

/* ── الأزرار ── */
.stButton > button {{
    width: 100%; min-height: 56px; border-radius: 18px; font-size: 1.1rem; font-weight: 700;
    background: {C['card']}; color: {C['text']}; border: 1px solid {C['border']};
}}
.stButton > button p {{ font-size: 1.1rem; font-weight: 700; }}
.stButton > button:hover, .stButton > button:focus {{ border-color: {C['violet']}; color: {C['text']}; }}
.stButton > button[kind="primary"] {{
    background: linear-gradient(90deg, {C['lavender']}, #7a69f2); color: #15122e; border: none;
    box-shadow: 0 10px 30px rgba(139,124,246,.30);
}}
.stButton > button[kind="primary"]:hover {{ color: #15122e; }}
.stButton > button:disabled, .stButton > button[kind="primary"]:disabled {{
    background: #34325c; color: #1a1838; box-shadow: none; opacity: 1; border: none;
}}
[class*="st-key-back"] button {{
    width: 46px !important; min-height: 46px !important; height: 46px; padding: 0 !important;
    border-radius: 14px !important;
}}
[class*="st-key-back"] button p {{ font-size: 1.5rem !important; }}

/* الأعمدة ما تنكسر على شاشة صغيرة */
[data-testid="stHorizontalBlock"] {{ flex-wrap: nowrap !important; align-items: center; gap: .5rem; }}
[data-testid="stColumn"], [data-testid="column"] {{ min-width: 0 !important; }}

/* ── رافع الملفات: شفاف وفوق كرت الرفع، فالكرت كله يصير قابل للضغط ── */
[data-testid="stFileUploader"] {{ margin-top: -322px; position: relative; z-index: 5; }}
[data-testid="stFileUploader"] label {{ display: none; }}
[data-testid="stFileUploaderDropzone"] {{ height: 306px !important; opacity: 0; cursor: pointer; }}

/* ── مكونات ── */
.topbar-title {{ text-align:center; font-weight:700; font-size:1.15rem; }}
.brandbar {{ display:flex; justify-content:flex-start; align-items:center; gap:.6rem; font-size:1.45rem; font-weight:700;
    padding: 10px 22px 14px; border-bottom: 1px solid {C['border']}; margin: 0 -22px 20px; }}
.logo {{ width:40px; height:40px; border-radius:11px; display:flex; align-items:center; justify-content:center;
    background: linear-gradient(135deg, {C['lavender']}, {C['teal']}); color:#0e0f26; }}
.eyebrow {{ color:{C['lavender']}; font-size:.85rem; }}
.h1 {{ font-size:1.6rem; font-weight:700; margin:.1rem 0 1rem; line-height:1.4; }}
.muted {{ color:{C['muted']}; font-size:.88rem; }}
.center {{ text-align:center; }}
.card {{ background:{C['card']}; border:1px solid {C['border']}; border-radius:20px; padding:1.1rem; }}
.note {{ display:flex; gap:.55rem; align-items:center; font-size:.82rem; color:#c9c9e6;
    background:{C['card']}; border:1px solid {C['border']}; border-radius:14px; padding:.8rem 1rem; margin:.4rem 0 1rem; }}

/* Splash */
.splash {{ height: 770px; display:flex; flex-direction:column; align-items:center; justify-content:center; }}
.orb {{ width: 165px; height: 165px; border-radius: 50%; margin-bottom: 1.6rem;
    background:
        repeating-linear-gradient(165deg, transparent 0 15px, rgba(255,255,255,.45) 15px 17px),
        radial-gradient(circle at 70% 28%, #ffd36b 0%, #f7a6c9 26%, #9b8cff 52%, #45c4c9 82%);
    box-shadow: 0 0 70px rgba(139,124,246,.45), inset -14px -18px 30px rgba(20,10,60,.35); }}
.splash-title {{ font-size: 3rem; font-weight: 700; line-height: 1.2; }}
.loader {{ width: 70px; height: 4px; border-radius: 4px; background:{C['border']}; margin-top: 11rem; overflow:hidden; }}
.loader span {{ display:block; height:100%; width:55%; background:{C['violet']}; animation: load 1.2s ease-in-out infinite; }}
@keyframes load {{ 0% {{ transform: translateX(130%); }} 100% {{ transform: translateX(-130%); }} }}

/* Upload */
.upload-card {{ height: 306px; border: 1.5px dashed {C['border']}; border-radius: 22px;
    background: radial-gradient(circle at 50% 38%, #2a2a5e 0%, {C['card']} 55%);
    display:flex; flex-direction:column; align-items:center; justify-content:center; text-align:center; }}
.upload-card.has-file {{ border-color: {C['violet']}; border-style: solid; }}
.rings {{ width:118px; height:118px; border-radius:50%; border:2px solid rgba(255,255,255,.85);
    display:flex; align-items:center; justify-content:center; margin-bottom: 1rem; }}
.rings > div {{ width:98px; height:98px; border-radius:50%; border:1px solid rgba(255,255,255,.55);
    display:flex; align-items:center; justify-content:center; }}
.rings > div > div {{ width:82px; height:82px; border-radius:50%; display:flex; align-items:center; justify-content:center;
    background: linear-gradient(135deg, {C['lavender']}, #6d5ce6); }}
.upload-title {{ font-weight:700; font-size:1.2rem; }}
.file-name {{ margin-top:.55rem; color:{C['lavender']}; font-size:.85rem; direction:ltr; font-family: monospace; }}
.formats {{ text-align:center; color:{C['muted']}; font-size:.75rem; letter-spacing:.2em; margin: .9rem 0 1.1rem;
    font-family: monospace; direction: ltr; }}

/* Analyzing */
.pulse-wrap {{ width: 210px; height: 210px; margin: .4rem auto .6rem; border-radius:50%;
    border: 1px solid rgba(139,124,246,.18); display:flex; align-items:center; justify-content:center; }}
.pulse-wrap > div {{ width: 160px; height: 160px; border-radius:50%; border: 1px solid rgba(139,124,246,.25);
    display:flex; align-items:center; justify-content:center; }}
.pulse {{ width: 96px; height: 96px; border-radius:50%; display:flex; align-items:center; justify-content:center;
    background: linear-gradient(135deg, {C['lavender']}, {C['teal']}); animation: pulse 1.6s ease-in-out infinite; }}
@keyframes pulse {{ 50% {{ transform: scale(1.07); box-shadow: 0 0 40px rgba(139,124,246,.5); }} }}
.steps {{ padding: 1rem 1rem; }}
.step {{ display:flex; align-items:flex-start; gap:.8rem; position: relative; padding-bottom: 1.15rem; }}
.step:last-child {{ padding-bottom: 0; }}
.step .dot {{ width:36px; height:36px; flex: 0 0 36px; border-radius:50%; display:flex; align-items:center; justify-content:center;
    font-size:.85rem; border:1.5px solid {C['border']}; color:{C['muted']}; background:{C['screen']}; z-index:1; }}
.step:not(:last-child)::after {{ content:""; position:absolute; right:17px; top:36px; bottom:0; width:1.5px; background:{C['border']}; }}
.step.done .dot {{ border-color: rgba(62,201,180,.55); color:{C['teal']}; background: #16303f; }}
.step.active .dot {{ border-color:{C['violet']}; color:{C['text']}; background: #2a2860; }}
.step .name {{ font-weight:600; font-size:.98rem; }}
.step .state {{ font-size:.75rem; color:{C['muted']}; }}
.step.active .state {{ color:{C['lavender']}; }}
.step .spin {{ margin-right:auto; align-self:center; width:20px; height:20px; border-radius:50%;
    border:2px solid {C['border']}; border-top-color:{C['violet']}; animation: spin .9s linear infinite; }}
@keyframes spin {{ to {{ transform: rotate(360deg); }} }}
.bar {{ height:4px; border-radius:4px; background:{C['border']}; margin: 1.3rem .2rem 0; overflow:hidden; }}
.bar span {{ display:block; height:100%; background: linear-gradient(270deg, {C['violet']}, {C['teal']}); }}
.privacy {{ display:flex; justify-content:center; gap:.4rem; align-items:center; color:{C['muted']}; font-size:.82rem; margin-top: 2rem; }}

/* Result */
.result-card {{ height: 340px; border-radius: 24px; display:flex; flex-direction:column; align-items:center; justify-content:center;
    margin-top: .6rem; }}
.verdict-icon {{ width:80px; height:80px; border-radius:50%; display:flex; align-items:center; justify-content:center;
    color:#1a0f1f; margin-bottom: 1.2rem; }}
.verdict {{ font-size: 3rem; font-weight: 700; line-height: 1.25; }}
.conf {{ color:#d6d6ee; font-size: 1rem; }}
.conf b {{ font-size: 1.1rem; }}
.tiles {{ display:flex; gap:.6rem; margin: .8rem 0 .6rem; }}
.tile {{ flex:1; background:{C['card']}; border:1px solid {C['border']}; border-radius:16px; padding:.8rem 1rem; }}
.tile .k {{ font-size:.75rem; color:{C['muted']}; }}
.tile .v {{ font-weight:700; font-size:1.05rem; margin-top: .25rem; }}
.h1-sm {{ font-size:1.6rem; font-weight:700; line-height:1.4; margin-top:.4rem; }}
[data-testid="stExpander"] details {{ background:{C['card']}; border:1px solid {C['border']}; border-radius:14px; }}
[data-testid="stExpander"] summary, [data-testid="stExpander"] p {{ color:{C['text']}; }}
[data-testid="stIFrame"], iframe {{ border: none; }}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ─────────────────────────────── أدوات ───────────────────────────────
def html(s):
    st.markdown(s, unsafe_allow_html=True)


def init_state():
    defaults = {"screen": "splash", "audio_path": None, "file_name": None,
                "result": None, "meta": None, "error": None, "explanation": None, "play_wave": None}
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


def go(screen):
    st.session_state.screen = screen
    st.rerun()


def reset():
    path = st.session_state.audio_path
    if path and os.path.exists(path):
        os.remove(path)
    for k in ("audio_path", "file_name", "result", "meta", "error", "explanation", "play_wave"):
        st.session_state[k] = None
    go("upload")


def fail(kind, detail):
    """رسالة الخطأ التقنية تطلع في الـ terminal للمطورين، والمستخدم يشوف رسالة بسيطة."""
    print(f"[nabra] {kind}: {detail}", file=sys.stderr, flush=True)
    st.session_state.error = kind
    go("error")


def statusbar():
    now = datetime.now().strftime("%I:%M").lstrip("0")  # الوقت الحقيقي من جهازك
    html(f'<div class="statusbar"><span>{now}</span>'
         f'<span class="r">5G <span class="battery"><span></span></span></span></div>')


def topbar(title, key, back_to=None):
    """زر رجوع على اليمين وعنوان في النص (نفس تصميم فيقما)."""
    c_back, c_title, c_space = st.columns([1, 4, 1], vertical_alignment="center")
    with c_back:
        if st.button("‹", key=f"back_{key}"):
            if back_to:
                go(back_to)
            reset()
    with c_title:
        html(f'<div class="topbar-title">{title}</div>')


# ─────────────────────────────── الشاشات ───────────────────────────────
def splash_screen():
    statusbar()
    html('<div class="splash"><div class="orb"></div><div class="splash-title">نبرة</div>'
         '<div class="muted">تحقق من موثوقية التسجيلات الصوتية</div>'
         '<div class="loader"><span></span></div></div>')
    start = time.time()
    if IMPORT_ERROR:
        fail("service", IMPORT_ERROR)
    # تحميل الموديل مرة وحدة (WavLM-Large + best.pt). أول مرة ينزّل WavLM من Hugging Face.
    error = None
    try:
        load_model()
    except Exception as e:
        error = f"تعذّر تحميل الموديل: {e}"
    if error:
        fail("service", error)
    time.sleep(max(0.0, 2.5 - (time.time() - start)))
    go("upload")


def upload_screen():
    statusbar()
    html(f'<div class="brandbar"><span class="logo">{ICON_WAVE}</span>نبرة</div>')
    html('<div class="eyebrow">تحليل جديد</div><div class="h1">لنتحقق من الصوت</div>')

    card = st.empty()  # الكرت يترسم بعد ما نعرف إذا فيه ملف
    uploaded = st.file_uploader("ملف صوتي", type=["wav", "mp3"], label_visibility="collapsed")

    file_line = f'<div class="file-name">{uploaded.name}</div>' if uploaded else ""
    card.markdown(
        f'<div class="upload-card {"has-file" if uploaded else ""}">'
        f'<div class="rings"><div><div>{ICON_UPLOAD}</div></div></div>'
        f'<div class="upload-title">{"تم اختيار التسجيل" if uploaded else "اختر تسجيلًا صوتيًا"}</div>'
        f'<div class="muted">{"اضغط مرة أخرى لتغييره" if uploaded else "وسنحلل مدى موثوقيته"}</div>'
        f'{file_line}</div>',
        unsafe_allow_html=True,
    )
    html('<div class="formats">WAV &nbsp;•&nbsp; MP3</div>')

    if st.button("ابدأ التحليل   ›", type="primary", disabled=uploaded is None, width="stretch"):
        suffix = os.path.splitext(uploaded.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.getbuffer())
        st.session_state.audio_path = tmp.name
        st.session_state.file_name = uploaded.name
        go("analyzing")

    html(f'<div class="note">{ICON_INFO}<span>لفحص مكالمة، يجب تسجيلها أولًا</span></div>')


STEPS = ["تجهيز الملف", "استخراج خصائص الصوت", "فحص التسجيل"]


def render_steps(box, active):
    rows = ""
    for i, name in enumerate(STEPS):
        if i < active:
            cls, dot, state, spin = "done", ICON_CHECK, "اكتمل", ""
        elif i == active:
            cls, dot, state, spin = "active", str(i + 1), "جارٍ الآن", '<div class="spin"></div>'
        else:
            cls, dot, state, spin = "", str(i + 1), "بالانتظار", ""
        rows += (f'<div class="step {cls}"><div class="dot">{dot}</div>'
                 f'<div><div class="name">{name}</div><div class="state">{state}</div></div>{spin}</div>')
    pct = int(100 * active / len(STEPS))
    box.markdown(f'<div class="card steps">{rows}</div><div class="bar"><span style="width:{pct}%"></span></div>',
                 unsafe_allow_html=True)


def analyzing_screen():
    statusbar()
    topbar("جاري التحليل", "analyzing")
    html(f'<div class="pulse-wrap"><div><div class="pulse">{ICON_BARS}</div></div></div>'
         '<div class="h1 center" style="margin-bottom:0">نحلل التسجيل...</div>'
         '<div class="muted center" style="margin-bottom:1rem">قد يستغرق هذا بضع لحظات</div>')
    box = st.empty()
    html(f'<div class="privacy">{ICON_LOCK}<span>لا يُحفظ التسجيل دون إذنك</span></div>')

    path = st.session_state.audio_path
    error = None
    try:
        # 1) تجهيز الملف: نقرأ الملف ونتأكد إنه صوت صالح، ونطلع المدة ومعدل العينة
        render_steps(box, 0)
        st.session_state.meta = audio_meta(path)

        # 2) و 3) الموديل: استخراج الخصائص والتصنيف يصيرون داخل predict
        render_steps(box, 1)
        time.sleep(0.3)
        raw = predict(path)
        render_steps(box, 2)
        st.session_state.result = check_result(raw)
        render_steps(box, 3)
        time.sleep(0.4)
    except Exception as e:
        error = str(e)

    if error:
        fail("file", error)
    go("result")


def result_screen():
    statusbar()
    topbar("النتيجة", "result")
    res = st.session_state.result
    meta = st.session_state.meta
    is_fake = res["label"] == "fake"
    color = C["fake"] if is_fake else C["real"]
    word = "مزيف" if is_fake else "حقيقي"
    pct = round(res["confidence"] * 100)

    html(f'<div class="result-card" style="border:1px solid {color}55;'
         f'background: radial-gradient(circle at 50% 42%, {color}33 0%, {C["card"]} 60%);">'
         f'<div class="verdict-icon" style="background:{color}; box-shadow: 0 0 40px {color}66;">{ICON_WAVE}</div>'
         f'<div class="muted" style="font-size:.8rem">نتيجة التحليل</div>'
         f'<div class="verdict" style="color:{color}">{word}</div>'
         f'<div class="conf">{word} بنسبة ثقة تقريبية <b>{pct}%</b></div></div>')
    html('<div class="tiles">'
         f'<div class="tile"><div class="k">المدة</div><div class="v">{meta["duration"]}</div></div>'
         f'<div class="tile"><div class="k">الصيغة</div><div class="v">{meta["format"]}</div></div>'
         '</div>')
    html(f'<div class="note">{ICON_INFO}<span>يمكنك الاطلاع على المقاطع التي أثّرت أكثر على هذه النتيجة.</span></div>')

    if st.button("لماذا هذه النتيجة؟   ›", type="primary", width="stretch"):
        go("explaining" if st.session_state.explanation is None else "explain")
    if st.button("تحليل ملف آخر", width="stretch"):
        reset()


def error_screen():
    statusbar()
    topbar("تعذّر التحليل", "error")
    # التفاصيل التقنية تنطبع في الـ terminal (fail)، وهنا رسالة للمستخدم فقط.
    if st.session_state.error == "service":
        title, msg = "الخدمة غير متاحة حاليًا", "حدث خطأ أثناء تجهيز التطبيق. حاول مرة أخرى لاحقًا."
    else:
        title, msg = "الملف غير صالح", "تأكد أن الملف تسجيل صوتي صالح بصيغة WAV أو MP3، ثم حاول مرة أخرى."
    color = C["fake"]
    html(f'<div class="result-card" style="height:300px;padding:0 1.4rem;text-align:center;border:1px solid {color}55;'
         f'background: radial-gradient(circle at 50% 38%, {color}26 0%, {C["card"]} 60%);">'
         f'<div class="verdict-icon" style="background:{color}; box-shadow: 0 0 40px {color}55; font-size:2.2rem; font-weight:700">!</div>'
         f'<div class="h1" style="font-size:1.4rem;margin:0 0 .4rem">{title}</div>'
         f'<div class="muted">{msg}</div></div>')
    html('<div style="height:.9rem"></div>')
    if st.button("جرّب ملفًا آخر", type="primary", width="stretch"):
        reset()


def wav_data_uri(wave, sr):
    """يحوّل الصوت لـ WAV داخل الصفحة عشان نقدر نشغله ونشغل المقاطع."""
    import soundfile as sf
    buf = io.BytesIO()
    sf.write(buf, np.asarray(wave, dtype=np.float32), sr, format="WAV", subtype="PCM_16")
    return "data:audio/wav;base64," + base64.b64encode(buf.getvalue()).decode()


def explaining_screen():
    """شاشة انتظار: التفسير (GATR) أبطأ من التحليل، خصوصًا على جهاز بدون GPU."""
    statusbar()
    topbar("تفسير النتيجة", "explaining", back_to="result")
    html(f'<div class="pulse-wrap"><div><div class="pulse">{ICON_BARS}</div></div></div>'
         '<div class="h1 center" style="margin-bottom:0">نجهز التفسير...</div>'
         '<div class="muted center">نحدد المقاطع التي أثّرت على النتيجة</div>')
    html('<div class="bar" style="margin-top:1.6rem"><span style="width:100%;animation: load 1.6s ease-in-out infinite"></span></div>')
    html(f'<div class="privacy">{ICON_LOCK}<span>لا يُحفظ التسجيل دون إذنك</span></div>')

    path = st.session_state.audio_path
    error = None
    try:
        e = explain(path)
        # الصوت اللي يشغله التطبيق: نفس الصوت بعد التنظيف اللي شافه الموديل،
        # عشان أوقات المقاطع تطابق (الموديل يقص الصمت من البداية والنهاية).
        st.session_state.explanation = e
        st.session_state.play_wave = (clean_wave(load_wave(path)), SR)
    except Exception as ex:
        error = f"تعذّر تجهيز التفسير: {ex}"
    if error:
        fail("file", error)
    go("explain")


def explanation_widget(wave, sr, frames, segments, file_name, height):
    """الموجة الصوتية مع تظليل المقاطع المؤثرة + أزرار تشغيل كل مقطع (نفس تصميم فيقما)."""
    dur = len(wave) / sr
    n_bars = 46
    chunks = np.array_split(np.asarray(wave, dtype=np.float32), n_bars)
    amp = np.array([np.sqrt(np.mean(c ** 2)) if len(c) else 0.0 for c in chunks])
    amp = amp / (amp.max() + 1e-9)

    frames = np.asarray(frames, dtype=float) if frames is not None else np.zeros(1)
    fchunks = np.array_split(frames, n_bars)
    rel = np.array([c.mean() if len(c) else 0.0 for c in fchunks])
    rel = rel / (rel.max() + 1e-9) if rel.max() > 0 else rel

    bars = ""
    for i in range(n_bars):
        t0, t1 = i * dur / n_bars, (i + 1) * dur / n_bars
        in_seg = any(a < t1 and b > t0 for a, b in segments)  # العمود يتقاطع مع مقطع مؤثر
        h = 14 + 86 * amp[i]
        if in_seg:
            color, op = C["violet"], 1.0
        else:
            color, op = "#4a4b7c", 0.55 + 0.45 * rel[i]
        bars += f'<div class="b" style="height:{h:.0f}%;background:{color};opacity:{op:.2f}"></div>'

    ticks = "".join(f'<span>{dur * k / 4:.0f}ث</span>' for k in range(5))
    mm, ss = divmod(int(round(dur)), 60)

    tiles = ""
    for i, (a, b) in enumerate(segments):
        tiles += (f'<div class="tile" id="t{i}" onclick="playSeg({a:.3f},{b:.3f},{i})">'
                  f'<div class="pbtn">▶</div><div class="tt"><span dir="ltr">{a:.1f} - {b:.1f}</span> ث</div><div class="ul"></div></div>')
    if not segments:
        tiles = '<div class="empty">لم تُحدَّد مقاطع معيّنة أثّرت على النتيجة في هذا التسجيل.</div>'

    page = f"""
<html><head><style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Arabic:wght@400;600;700&display=swap');
* {{ box-sizing:border-box; margin:0; padding:0; }}
body {{ background:transparent; color:#fff; font-family:'IBM Plex Sans Arabic',sans-serif; direction:rtl; }}
.card {{ background:{C['card']}; border:1px solid {C['border']}; border-radius:20px; padding:16px; }}
.head {{ display:flex; align-items:center; gap:12px; direction:ltr; }}
.play {{ width:44px; height:44px; border-radius:50%; background:{C['lavender']}; color:#15122e; border:none;
  font-size:16px; cursor:pointer; display:flex; align-items:center; justify-content:center; }}
.fname {{ font-weight:700; font-size:13px; }} .fdur {{ color:{C['muted']}; font-size:11px; font-family:monospace; }}
.wave {{ display:flex; align-items:center; gap:3px; height:110px; margin:14px 0 6px; direction:ltr; }}
.b {{ flex:1; border-radius:3px; }}
.ticks {{ display:flex; justify-content:space-between; direction:ltr; color:{C['muted']}; font-size:10px; }}
.tiles {{ display:flex; gap:8px; margin-top:12px; }}
.tile {{ flex:1; background:{C['card']}; border:1px solid {C['border']}; border-radius:16px; padding:12px 6px;
  display:flex; flex-direction:column; align-items:center; gap:8px; cursor:pointer; }}
.tile.on {{ border-color:{C['violet']}; }}
.pbtn {{ width:34px; height:34px; border-radius:50%; background:#2e2f62; display:flex; align-items:center; justify-content:center; font-size:11px; }}
.tt {{ font-family:monospace; font-weight:700; font-size:11.5px; white-space:nowrap; }}
.ul {{ width:30px; height:3px; border-radius:2px; background:{C['violet']}; }}
.empty {{ color:{C['muted']}; font-size:13px; padding:14px; text-align:center; width:100%;
  background:{C['card']}; border:1px solid {C['border']}; border-radius:16px; }}
</style></head><body>
<div class="card">
  <div class="head"><button class="play" id="pa" onclick="playAll()">▶</button>
    <div><div class="fname">{file_name}</div><div class="fdur">{mm:02d}:{ss:02d}</div></div></div>
  <div class="wave">{bars}</div>
  <div class="ticks">{ticks}</div>
</div>
<div class="tiles">{tiles}</div>
<audio id="au" src="{wav_data_uri(wave, sr)}"></audio>
<script>
const au = document.getElementById('au'); let stopAt = null;
function clearOn() {{ document.querySelectorAll('.tile').forEach(t => t.classList.remove('on')); }}
function playAll() {{
  clearOn(); stopAt = null;
  if (!au.paused) {{ au.pause(); document.getElementById('pa').textContent='▶'; return; }}
  au.currentTime = 0; au.play(); document.getElementById('pa').textContent='❚❚';
}}
function playSeg(a, b, i) {{
  clearOn(); document.getElementById('t'+i).classList.add('on');
  au.currentTime = a; stopAt = b; au.play(); document.getElementById('pa').textContent='▶';
}}
au.addEventListener('timeupdate', () => {{ if (stopAt !== null && au.currentTime >= stopAt) {{ au.pause(); stopAt = null; clearOn(); }} }});
au.addEventListener('ended', () => {{ document.getElementById('pa').textContent='▶'; clearOn(); }});
</script></body></html>"""
    components.html(page, height=height)


def explain_screen():
    statusbar()
    topbar("تفسير النتيجة", "explain", back_to="result")
    e = st.session_state.explanation
    wave, sr = st.session_state.play_wave
    segments = [(float(a), float(b)) for a, b in (e.get("segments") or [])]

    html('<div class="h1-sm">المقاطع الأكثر تأثيرًا</div>'
         '<div class="muted" style="margin-bottom:.8rem">اضغط على أي مقطع لسماعه بشكل منفصل</div>')
    explanation_widget(wave, sr, e.get("frames"), segments, st.session_state.file_name or "",
                       height=335 if segments else 310)
    html(f'<div class="note">{ICON_INFO}<span>المقاطع المظللة هي الأكثر تأثيرًا على النتيجة، '
         'ولا تعني بالضرورة أن هذا الجزء تم التلاعب به.</span></div>')

    if st.button("تحليل ملف آخر", width="stretch"):
        reset()


# ─────────────────────────────── التشغيل ───────────────────────────────
init_state()
SCREENS = {
    "splash": splash_screen,
    "upload": upload_screen,
    "analyzing": analyzing_screen,
    "result": result_screen,
    "error": error_screen,
    "explaining": explaining_screen,
    "explain": explain_screen,
}
root = st.empty()
with root.container():
    SCREENS[st.session_state.screen]()
