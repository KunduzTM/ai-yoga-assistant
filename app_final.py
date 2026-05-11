import streamlit as st
import cv2
import numpy as np
import joblib
import mediapipe as mp
import plotly.graph_objects as go
from PIL import Image
import os
import re

# Import utils
from pose_utils import (
    extract_features_from_landmarks,
    compute_relevant_angles,
    generate_feedback,
    POSES_DB
)

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="AI Yoga Assistant",
    page_icon="🧘‍♀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Стили CSS для увеличения шрифта вкладок
st.markdown("""
<style>
    /* Увеличиваем шрифт заголовков вкладок */
    div[data-baseweb="tab-list"] p {
        font-size: 24px !important;
        font-weight: bold !important;
    }
    
    /* Опционально: увеличиваем кнопки вкладок */
    div[data-baseweb="tab-list"] button {
        padding-top: 10px !important;
        padding-bottom: 10px !important;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def format_text(text):
    """Breaks text into lines for 1. 2. 3. and bullets."""
    if not text: return ""
    text = re.sub(r'(\d+\.)', r'\n\n**\1**', text) 
    text = text.replace('•', '\n* ')
    return text

# ==========================================
# 3. TEXTS & TRANSLATIONS
# ==========================================
UI_TEXT = {
    "main_title": {
        "EN": "🧘 AI Yoga Assistant", "RU": "🧘 ИИ Йога-Ассистент", "KG": "🧘 AI Йога Жардамчысы"
    },
    "nav_header": {"EN": "Menu", "RU": "Меню", "KG": "Меню"},
    "nav_coach": {"EN": "AI Assistant", "RU": "AI Ассистент", "KG": "AI Жардамчы"},
    "nav_lib": {"EN": "Asana Library", "RU": "Библиотека асан", "KG": "Асаналар китепканасы"},
    "nav_about": {"EN": "About Project", "RU": "О проекте", "KG": "Долбоор жөнүндө"},
    
    # About Page
    "about_title": {"EN": "AI YOGA Assistant Project", "RU": "AI YOGA Ассистент", "KG": "AI YOGA Жардамчы"},
    "author_header": {"EN": "Author", "RU": "Автор", "KG": "Автор"},
    "author_name": {"EN": "Kunduz Tashmatova", "RU": "Кундуз Ташматова", "KG": "Кундуз Ташматова"},
    "author_role": {
        "EN": "\n* Student of **Data Science Pro Math II** course\n* AI Academy, Bishkek",
        "RU": "\n* Студентка курса **Data Science Pro Math II**\n* AI Academy, Бишкек",
        "KG": "\n* **Data Science Pro Math II** курсунун студенти\n* AI Academy, Бишкек"
    },
    "tech_stack": {
        "EN": "**🛠 Stack:**\n* Python\n* Streamlit\n* MediaPipe\n* Scikit-learn",
        "RU": "**🛠 Стек:**\n* Python\n* Streamlit\n* MediaPipe\n* Scikit-learn",
        "KG": "**🛠 Стек:**\n* Python\n* Streamlit\n* MediaPipe\n* Scikit-learn"
    },
    "ml_info": {
        "EN": "**🧠 Machine Learning:**\n* **Model:** CatBoost Classifier\n* **Optimization:** Optuna Tuning\n* **Features:** 111 geometric inputs",
        "RU": "**🧠 Машинное обучение:**\n* **Модель:** CatBoost Classifier\n* **Оптимизация:** Optuna Tuning\n* **Признаки:** 111 геометрических параметров",
        "KG": "**🧠 Машиналык окутуу:**\n* **Модель:** CatBoost Classifier\n* **Оптималдаштыруу:** Optuna Tuning\n* **Белгилер:** 111 геометриялык параметр"
    },
    "results_info": {
        "EN": "**🏆 Results:**\n* **Accuracy:** 90.2%",
        "RU": "**🏆 Результаты:**\n* **Точность:** 90.2%",
        "KG": "**🏆 Жыйынтыктар:**\n* **Тактык:** 90.2%"
    },

    # Library
    "lib_title": {"EN": "Pose Library", "RU": "Библиотека асан", "KG": "Асаналар китепканасы"},
    "select_pose": {"EN": "Select pose:", "RU": "Выберите позу:", "KG": "Позаны тандаңыз:"},
    "lib_tech": {"EN": "Technique", "RU": "Техника выполнения", "KG": "Аткаруу техникасы"},
    "lib_ben": {"EN": "Benefits", "RU": "Польза", "KG": "Пайдасы"},
    "lib_con": {"EN": "Contraindications", "RU": "Противопоказания", "KG": "Каршы көрсөтмөлөр"},

    # Coach UI
    "sidebar_settings": {"EN": "Settings", "RU": "Настройки", "KG": "Жөндөөлөр"},
    "input_mode": {"EN": "Mode", "RU": "Режим", "KG": "Режим"},
    "modes": {"Upload": {"EN": "Upload Photo", "RU": "Загрузить фото", "KG": "Сүрөт жүктөө"}, "Live": {"EN": "Webcam", "RU": "Веб-камера", "KG": "Веб-камера"}},
    "cv_header": {"EN": "Analysis", "RU": "Анализ", "KG": "Анализ"},
    "tech_details": {"EN": "Technical Details (Angles)", "RU": "Технические детали (Углы)", "KG": "Техникалык маалымат (Бурчтар)"},
    "pred_header": {"EN": "Result", "RU": "Результат", "KG": "Жыйынтык"},
    "feedback_header": {"EN": "Recommendations", "RU": "Рекомендации", "KG": "Кеңештер"},
    "ref_expand": {"EN": "How to do it properly?", "RU": "Как делать правильно?", "KG": "Кантип туура жасоо керек?"},
    "upload_text": {"EN": "Upload an image", "RU": "Выберите изображение", "KG": "Сүрөттү тандаңыз"},
    "welcome": {"EN": "👋 Please upload a photo to start.", "RU": "👋 Загрузите фото для начала.", "KG": "👋 Баштоо үчүн сүрөт жүктөңүз."},
    "unknown_pose": {"EN": "Unknown Pose", "RU": "Неизвестная поза", "KG": "Белгисиз поза"},
    "confidence": {"EN": "Confidence", "RU": "Уверенность", "KG": "Ишенимдүүлүк"},

    "good_job": {
        "EN": "✅ Good alignment!", 
        "RU": "✅ Хорошее положение!", 
        "KG": "✅ Абалы жакшы!"
    },
    "no_desc": {
        "EN": "Description unavailable for this pose.", 
        "RU": "Описание для этой позы отсутствует.", 
        "KG": "Бул поза үчүн маалымат жок."
    },
}

# ==========================================
# 4. LOAD MODELS
# ==========================================
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load("artifacts/yoga_model_final.pkl")
        scaler = joblib.load("artifacts/scaler.pkl")
        le = joblib.load("artifacts/label_encoder.pkl")
        return model, scaler, le
    except:
        return None, None, None

model, scaler, le = load_artifacts()

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# ==========================================
# 5. PROCESSING LOGIC
# ==========================================
def process_image(image_input, threshold, lang_code):
    image_rgb = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=threshold, model_complexity=1) as pose:
        results = pose.process(image_rgb)

    if not results.pose_landmarks: return None, None

    annotated_image = image_input.copy()
    mp_drawing.draw_landmarks(annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, 
                              landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

    features, _ = extract_features_from_landmarks(results.pose_landmarks.landmark)
    features_scaled = scaler.transform(features)
    probs = model.predict_proba(features_scaled)[0]
    pred_idx = np.argmax(probs)
    
    # Имя класса от модели
    pose_class_name = le.inverse_transform([pred_idx])[0]
    
    angles = compute_relevant_angles(results.pose_landmarks.landmark)
    
    # === ИСПРАВЛЕНИЕ ОШИБОК (RULE-BASED CORRECTION) ===
    # Если модель думает, что это "Стул", но ноги несимметричны -> это Дерево
    if pose_class_name == "utkatasana":
        lk = angles.get('left_knee', 180)
        rk = angles.get('right_knee', 180)
        
        # В стуле колени согнуты одинаково. В дереве - разница большая.
        # Если разница больше 40 градусов, меняем предсказание на Дерево.
        if abs(lk - rk) > 40:
            pose_class_name = "vriksasana"
            # Можно искусственно поднять уверенность для красивого графика
            # (необязательно, но для демо полезно)
            # probs[pred_idx] = 0.1 
            # tree_idx = list(le.classes_).index("vriksasana")
            # probs[tree_idx] = 0.9 
    # ==================================================
    
    # Generate feedback using the (possibly corrected) class name
    feedback = generate_feedback(pose_class_name, angles, lang_code)
    
    return annotated_image, {
        "pose_name": pose_class_name, 
        "confidence": probs[pred_idx], 
        "probs": probs, 
        "angles": angles, 
        "feedback": feedback
    }

# ==========================================
# 6. SIDEBAR MENU
# ==========================================
with st.sidebar:
    lang = st.radio("🌐 Language / Язык / Тил", ["KG", "RU", "EN"], horizontal=True)
    st.divider()
    
    st.subheader(UI_TEXT["nav_header"][lang])
    
    # Order: About -> Lib -> Coach
    menu = st.radio("", [
        UI_TEXT["nav_about"][lang], 
        UI_TEXT["nav_lib"][lang], 
        UI_TEXT["nav_coach"][lang]
    ], label_visibility="collapsed")
    
    st.divider()
    
    input_mode = "Upload"
    conf_threshold = 0.5
    
    # Settings only for AI Coach
    if menu == UI_TEXT["nav_coach"][lang]:
        st.subheader(UI_TEXT["sidebar_settings"][lang])
        m_sel = st.selectbox(UI_TEXT["input_mode"][lang], [UI_TEXT["modes"]["Upload"][lang], UI_TEXT["modes"]["Live"][lang]])
        input_mode = "Upload" if m_sel == UI_TEXT["modes"]["Upload"][lang] else "Live"
        conf_threshold = st.slider("AI Sensitivity", 0.0, 1.0, 0.5, 0.05)

# ==========================================
# PAGE: ABOUT PROJECT
# ==========================================
if menu == UI_TEXT["nav_about"][lang]:
    st.title(UI_TEXT["about_title"][lang])
    
    # Author Section
    c1, c2 = st.columns([1, 5], gap="small")
    with c1:
        if os.path.exists("assets/author.jpg"):
            st.image("assets/author.jpg", width=250)
        else:
            st.image("https://cdn-icons-png.flaticon.com/512/4140/4140048.png", width=120)
            
    with c2:
        st.markdown(f"### {UI_TEXT['author_header'][lang]}") 
        st.subheader(UI_TEXT["author_name"][lang])
        st.markdown(UI_TEXT["author_role"][lang])
    
    st.divider()
    
    # Tech Info
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(UI_TEXT["tech_stack"][lang])
    
    with col2:
        st.markdown(UI_TEXT["ml_info"][lang])
        
    with col3:
        st.markdown(UI_TEXT["results_info"][lang])

# ==========================================
# PAGE: LIBRARY
# ==========================================
elif menu == UI_TEXT["nav_lib"][lang]:
    st.title(UI_TEXT["lib_title"][lang])
    
    keys = list(POSES_DB.keys())
    display_names = []
    for k in keys:
        info = POSES_DB[k].get(lang, POSES_DB[k]["EN"])
        display_names.append(info['title'])
    
    sel_name = st.selectbox(UI_TEXT["select_pose"][lang], display_names)
    
    sel_key = keys[display_names.index(sel_name)]
    info = POSES_DB[sel_key].get(lang, POSES_DB[sel_key]["EN"])
    
    st.header(info['title'])
    
    c1, c2 = st.columns([1, 1.5])
    with c1:
        if os.path.exists(info['image']):
            st.image(info['image'], use_container_width=True)
        else:
            st.image("https://via.placeholder.com/600x400?text=No+Image", use_container_width=True)
            
    with c2:
        st.subheader(UI_TEXT["lib_tech"][lang])
        st.markdown(format_text(info['technique']))

    st.divider()
    
    c3, c4 = st.columns(2)
    with c3:
        st.success(f"**{UI_TEXT['lib_ben'][lang]}**\n\n{format_text(info['benefits'])}")
    with c4:
        st.warning(f"**{UI_TEXT['lib_con'][lang]}**\n\n{format_text(info['contra'])}")

# ==========================================
# PAGE: AI COACH
# ==========================================
elif menu == UI_TEXT["nav_coach"][lang]:
    st.title(UI_TEXT["main_title"][lang])
    
    if not model:
        st.error("⚠️ Error: Model artifacts not found.")
        st.stop()

    img_input = None
    if input_mode == "Upload":
        f = st.file_uploader(UI_TEXT["upload_text"][lang], type=['jpg','png','jpeg'])
        if f: img_input = cv2.imdecode(np.frombuffer(f.read(), np.uint8), 1)
    else:
        f = st.camera_input("Camera")
        if f: img_input = cv2.imdecode(np.frombuffer(f.read(), np.uint8), 1)

    if img_input is not None:
        res_img, data = process_image(img_input, conf_threshold, lang)
        
        if res_img is None:
            st.warning("⚠️ Pose not detected.")
        else:
            # --- 1. DATA PREP ---
            db_key = data['pose_name'].lower()
            pose_info = POSES_DB.get(db_key)
            
            if pose_info:
                info_text = pose_info.get(lang, pose_info["EN"])
                display_title = info_text["title"]
            else:
                info_text = None
                display_title = data['pose_name'].replace('_', ' ').title()

            # --- 2. DISPLAY ---
            c1, c2 = st.columns([1, 1])
            
            # LEFT COLUMN: Photo + Info Tabs
            with c1:
                st.subheader(UI_TEXT["cv_header"][lang])
                st.image(cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB), use_container_width=True)
                
                if info_text:
                    st.divider()
                    
                    t1, t2, t3 = st.tabs([UI_TEXT["lib_ben"][lang], UI_TEXT["lib_con"][lang], UI_TEXT["lib_tech"][lang]])
                    
                    with t1: # Benefits
                        st.success(format_text(info_text['benefits']))
                        
                    with t2: # Contraindications
                        st.warning(format_text(info_text['contra']))
                        
                    with t3: # Technique (Image + Large Text)
                        if os.path.exists(info_text['image']):
                            st.image(info_text['image'], caption=f"Эталон: {display_title}", use_container_width=True)
                        
                        # Increased font size for technique
                        technique_html = f'<div style="font-size: 18px;">{format_text(info_text["technique"])}</div>'
                        st.markdown(technique_html, unsafe_allow_html=True)
                else:
                    st.info("Description unavailable for this pose.")

            # RIGHT COLUMN: Results + Feedback
            with c2:
                st.markdown(f"## {display_title}")
                
                conf_val = data['confidence']
                conf_text = f"{UI_TEXT['confidence'][lang]}: {conf_val:.1%}"
                if conf_val > 0.8:
                    st.caption(f"✅ {conf_text}")
                else:
                    st.caption(f"⚠️ {conf_text}")

                st.divider()

                st.subheader(UI_TEXT["feedback_header"][lang])
                
                if data['feedback']:
                    for fb in data['feedback']:
                        if "✅" in fb:
                            st.success(fb)
                        elif "👉" in fb:
                            st.warning(fb)
                        else:
                            st.info(fb)
                else:
                    st.success("✅ Good alignment!")

                # Chart with translation
                chart_captions = {
                    "EN": "AI Confidence Distribution",
                    "RU": "Распределение уверенности ИИ",
                    "KG": "AI Ишенимдүүлүгүнүн бөлүштүрүлүшү"
                }
                st.caption(chart_captions[lang])

                sorted_indices = np.argsort(data['probs'])[-3:]
                
                fig = go.Figure(go.Bar(
                    x=data['probs'][sorted_indices], 
                    y=[le.classes_[i].replace('_',' ').title() for i in sorted_indices], 
                    orientation='h', 
                    marker=dict(color='#4B0082')
                ))
                fig.update_layout(margin=dict(l=0,r=0,t=0,b=0), height=150, yaxis=dict(tickfont=dict(size=10)))
                st.plotly_chart(fig, use_container_width=True)

    else:
        st.info(UI_TEXT["welcome"][lang])