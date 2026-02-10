import streamlit as st
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Netflix Content Recommender",
    page_icon="🎬",
    layout="wide"
)

# =========================
# LOAD DATA
# =========================
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "netflix_shows.csv"

df = pd.read_csv(DATA_PATH)

# =========================
# PREPARE DATA
# =========================
df_rec = df[['title', 'listed inside']].dropna().copy()

df_rec['listed inside'] = (
    df_rec['listed inside']
    .str.lower()
    .str.replace(',', ' ', regex=False)
)

# =========================
# TF-IDF + COSINE SIMILARITY
# =========================
tfidf = TfidfVectorizer(
    stop_words="english",
    max_features=5000
)

tfidf_matrix = tfidf.fit_transform(df_rec['listed inside'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

indices = pd.Series(df_rec.index, index=df_rec['title']).drop_duplicates()


def recommend(title, top_n=7):
    if title not in indices:
        return []
    idx = indices[title]
    scores = list(enumerate(cosine_sim[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:top_n + 1]
    movie_indices = [i[0] for i in scores]
    return df_rec['title'].iloc[movie_indices].tolist()


# =========================
# UI STYLING (POLISHED)
# =========================


st.markdown(
    """
    <style>
    /* Main heading */
    h1 {
        text-align: center;
        color: #e50914;
        margin-bottom: 10px;
    }

    /* Selectbox outer container */
    div[data-baseweb="select"] > div {
        border: 2px solid #e50914 !important;
        border-radius: 10px;
        min-height: 56px;              /* FIX: height increased */
        display: flex;
        align-items: center;           /* FIX: vertical centering */
        padding: 0 12px;               /* FIX: horizontal padding */
        font-size: 16px;
    }

    /* Selected value text */
    div[data-baseweb="select"] span {
        line-height: 1.4 !important;   /* FIX: text clipping */
        padding-top: 2px;
    }

    /* Dropdown arrow alignment */
    div[data-baseweb="select"] svg {
        margin-top: 2px;
    }

    /* Recommend button */
    div.stButton > button {
        background-color: #e50914;
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 8px;
        padding: 0.6em 1.6em;
        font-size: 16px;
        margin-top: 10px;
    }

    div.stButton > button:hover {
        background-color: #b20710;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================
# TITLE
# =========================
st.markdown("<h1>Netflix Content Recommender</h1>", unsafe_allow_html=True)

st.markdown(
    "<p style='text-align:center; color:#9aa0a6; font-size:15px;'>"
    "Type a movie or TV show name and discover similar content"
    "</p>",
    unsafe_allow_html=True
)

st.markdown("<br>", unsafe_allow_html=True)

# =========================
# SINGLE SEARCH LINE (ONLY ONE)
# =========================
movie_titles = sorted(df_rec['title'].unique())

selected_movie = st.selectbox(
    "",
    movie_titles,
    index=None,
    placeholder="Search a movie or TV show...",
    label_visibility="collapsed"
)

# =========================
# RECOMMEND BUTTON
# =========================
clicked = st.button("🎬 Recommend")

# =========================
# SHOW RECOMMENDATIONS
# =========================
if clicked and selected_movie:
    st.markdown(
        "<h3 style='color:#e50914; margin-top:25px;'>Recommended For You</h3>",
        unsafe_allow_html=True
    )

    for movie in recommend(selected_movie):
        st.markdown(f"🎞️ **{movie}**")

# =========================
# FOOTER
# =========================
st.markdown(
    """
    <div style="text-align:center; color:#7a7a7a; margin-top:50px; font-size:14px;">
        Built with ❤️ by Raam | Netflix-Style Recommendation System
    </div>
    """,
    unsafe_allow_html=True
)
