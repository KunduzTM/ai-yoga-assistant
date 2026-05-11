# 🧘 AI Yoga Assistant — Localized AI-Powered Wellness App

> *"Building Inclusive AI for Real People"*  
> **Kunduz Tashmatova** | Data Science Pro Math II | AI Academy, Bishkek | Dec 2025

---

## 📌 Project Overview

An AI-powered yoga assistant designed specifically for **women in underserved communities** in Kyrgyzstan — with step-by-step guidance in **Kyrgyz, Russian, and English**.

The app uses computer vision to detect yoga poses in real time (via webcam or uploaded photo), classifies them with a machine learning model, and provides personalized feedback on body alignment — all without requiring a gym or instructor.

---

## 🎯 The Problem This Solves

| Dimension | Global AI Fitness Apps | AI Yoga (This Project) |
|-----------|----------------------|----------------------|
| Target audience | Urban, global users | Women in underserved communities |
| Product design | Fitness & performance | Health & safe movement |
| Cultural relevance | Generic | Locally adapted |
| Language | Optional translation | Core design element (KG/RU/EN) |
| Role of instructors | Replaced | Integrated and supported |

---

## 🗂️ Table of Contents

- [Demo](#-demo)
- [Features](#-features)
- [ML Pipeline](#-ml-pipeline)
- [Supported Poses](#-supported-poses-mvp)
- [Model Results](#-model-results)
- [Project Structure](#-project-structure)
- [How to Run](#-how-to-run)
- [Requirements](#-requirements)

---

## 🚀 Demo

The app has 3 sections:

**🤖 AI Assistant** — Upload a photo or use webcam → get pose detection + alignment feedback  
**📚 Asana Library** — Browse all 10 poses with technique, benefits, and contraindications  
**ℹ️ About Project** — Author info, tech stack, model results

---

## ✨ Features

- **Real-time pose detection** via MediaPipe (33 body landmarks)
- **ML classification** of 10 yoga poses (CatBoost, 90.2% accuracy)
- **Personalized feedback** on joint angles and body alignment
- **3 languages**: Kyrgyz 🇰🇬 | Russian 🇷🇺 | English 🇬🇧
- **Pose library** with technique, benefits, and contraindications for each pose
- **Confidence chart** showing top-3 model predictions
- **Rule-based corrections** on top of ML (e.g., Utkatasana vs Vriksasana disambiguation)

---

## 🔬 ML Pipeline

```
Kaggle Dataset          →  MediaPipe Pose          →  Feature Engineering
(1,834 images, 30 poses)   (33 landmarks extracted)   (111 geometric features:
                                                        angles + normalized coords)
        ↓
Torso-Based Normalization  →  Train/Val/Test Split   →  Optuna Hyperparameter Tuning
(scale invariance)            (60% / 20% / 20%)         (30 trials per model)
        ↓
Model Selection (CatBoost wins) → Retrain on full train set → Streamlit MVP
```

**Feature Engineering:**
- Joint angles (knee, hip, elbow, shoulder) calculated between MediaPipe landmarks
- Torso-based normalization for scale invariance (sitting vs. standing same person)
- 111 total features: 12 geometric + 99 raw coordinates (33 landmarks × 3 axes)

**Models Compared:**

| Model | Validation Accuracy |
|-------|-------------------|
| 🏆 **CatBoost** | **87.74%** |
| LightGBM | 86.65% |
| XGBoost | 86.38% |

**Final Test Set Accuracy: 90.22%** (retrained on full training set)

---

## 🧘 Supported Poses (MVP)

| Pose | Sanskrit Name | Category |
|------|--------------|----------|
| Mountain Pose | Tadasana | Standing |
| Tree Pose | Vriksasana | Balance |
| Warrior I | Virabhadrasana I | Standing |
| Warrior II | Virabhadrasana II | Standing |
| Downward Dog | Adho Mukha Svanasana | Inversion |
| Plank | Phalakasana | Core |
| Lotus Pose | Padmasana | Seated |
| Cobra | Bhujangasana | Backbend |
| Upward Dog | Urdhva Mukha Svanasana | Backbend |
| Shoulder Stand | Salamba Sarvangasana | Inversion |

---

## 📊 Model Results

```
Final Test Set Accuracy: 0.9022

              precision    recall  f1-score
   macro avg       0.90      0.90      0.90
weighted avg       0.90      0.90      0.90
```

---

## 📁 Project Structure

```
ai-yoga-assistant/
│
├── app_final.py                          # Main Streamlit application
├── pose_utils.py                         # Pose database, feature extraction, feedback logic
├── requirements.txt                      # Python dependencies
│
├── artifacts/                            # Trained ML model files
│   ├── yoga_model_final.pkl              # CatBoost classifier
│   ├── scaler.pkl                        # StandardScaler
│   ├── label_encoder.pkl                 # LabelEncoder for pose names
│   └── imputer.pkl                       # SimpleImputer
│
├── assets/                               # Pose reference images
│   ├── tadasana.png
│   ├── vriksasana.png
│   ├── virabhadrasana_I.png
│   ├── virabhadrasana_II.png
│   ├── adho_mukha_svanasana.png
│   ├── phalakasana.png
│   ├── padmasana.png
│   ├── bhujangasana.png
│   ├── urdhva_mukha_svanasana.png
│   ├── salamba_sarvangasana.png
│   └── author.jpg
│
├── notebook/
│   └── final-project-notebook-predefense_Dec13.ipynb   # Full ML pipeline
│
└── presentation/
    └── Kunduz_Tashmatova_Project_AI_YOGA.pdf            # Project presentation
```

---

## ▶️ How to Run

### Option 1 — Local (recommended)

```bash
# 1. Clone the repo
git clone https://github.com/KunduzTM/ai-yoga-assistant.git
cd ai-yoga-assistant

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app_final.py
```

> ⚠️ **Important**: Use `scikit-learn==1.2.2` (same version used for training) to avoid pickle compatibility warnings.

### Option 2 — Streamlit Cloud

1. Fork this repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set main file: `app_final.py`
5. Deploy!

---

## ⚙️ Requirements

```
streamlit>=1.28.0
opencv-python-headless>=4.8.0
mediapipe>=0.10.0
joblib>=1.3.0
scikit-learn==1.2.2
catboost>=1.2.0
plotly>=5.15.0
Pillow>=10.0.0
numpy>=1.24.0
```

Install:
```bash
pip install -r requirements.txt
```

---

## 🌍 Social Impact

This project contributes to:
- **Improved access** to basic physical activity and movement education
- **Early prevention** of musculoskeletal and stress-related issues common in rural women
- **Digital inclusion** through culturally and linguistically accessible AI tools

---

## 🚀 Future Roadmap

- [ ] Voice-based guidance for low-literacy users
- [ ] Expansion to back pain & postnatal recovery programs
- [ ] Adaptive difficulty based on user progress
- [ ] Collaboration with local yoga instructors
- [ ] Instructor dashboards for remote guidance
- [ ] NGO partnership programs

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Streamlit |
| Pose Detection | MediaPipe |
| ML Model | CatBoost |
| Hyperparameter Tuning | Optuna |
| Feature Engineering | NumPy, Scikit-learn |
| Visualization | Plotly |
| Language | Python 3.10+ |

---

## 👩‍💻 Author

**Kunduz Tashmatova**  
Student, Data Science Pro Math II | AI Academy, Bishkek  
📧 ktashmatova@gmail.com  
📱 t.me/kunduz_t

---

*Final project | Data Science Pro Math II | AI Academy | December 2025*
