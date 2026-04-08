import os
import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
import anthropic
from dotenv import load_dotenv
from search import extract_attributes, rerank, hyde_expand_query

# --- SETUP ---
load_dotenv()

st.set_page_config(
    page_title="Movie Detective",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

CLAUDE_KEY = os.getenv("CLAUDE_API_KEY")
if not CLAUDE_KEY:
    st.error("CLAUDE_API_KEY is missing in your environment.")
    st.stop()

client_ai = anthropic.Anthropic(api_key=CLAUDE_KEY)

DB_PATH = "./db/movie_db"
COLLECTION_NAME = "movie_detective"


@st.cache_resource
def init():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    client_db = chromadb.PersistentClient(path=DB_PATH)
    collection = client_db.get_collection(name=COLLECTION_NAME)
    return model, collection


try:
    model, collection = init()
except Exception:
    st.error("Database not found. Run ingest.py first.")
    st.stop()

# --- STYLES ---
st.markdown("""
<style>
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(229,9,20,0.18), transparent 28%),
            radial-gradient(circle at top right, rgba(29,78,216,0.18), transparent 30%),
            linear-gradient(180deg, #05070d 0%, #0b1020 45%, #090d18 100%);
        color: #f8fafc;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #111827 100%);
        border-right: 1px solid rgba(255,255,255,0.06);
    }

    .hero-wrap {
        position: relative;
        overflow: hidden;
        padding: 2.7rem 2.2rem 2.2rem 2.2rem;
        border-radius: 30px;
        background:
            linear-gradient(135deg, rgba(229,9,20,0.22), rgba(59,130,246,0.12)),
            rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 24px 60px rgba(0,0,0,0.35);
        margin-bottom: 1.8rem;
    }

    .hero-title {
        font-size: 3.25rem;
        font-weight: 900;
        line-height: 1.0;
        margin-bottom: 0.8rem;
        letter-spacing: -0.02em;
        color: #ffffff;
    }

    .hero-subtitle {
        font-size: 1.08rem;
        color: #d1d5db;
        max-width: 920px;
        line-height: 1.65;
        margin-bottom: 1.25rem;
    }

    .pill-row {
        display: flex;
        gap: 0.65rem;
        flex-wrap: wrap;
        margin-top: 0.4rem;
    }

    .pill {
        display: inline-block;
        padding: 0.45rem 0.85rem;
        border-radius: 999px;
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.08);
        color: #e5e7eb;
        font-size: 0.88rem;
        font-weight: 600;
    }

    .section-title {
        font-size: 1.55rem;
        font-weight: 850;
        color: #ffffff;
        margin-top: 0.35rem;
        margin-bottom: 0.9rem;
    }

    .movie-card {
        background: linear-gradient(180deg, rgba(255,255,255,0.06), rgba(255,255,255,0.035));
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 22px;
        overflow: hidden;
        padding: 0.7rem;
        transition: transform 0.22s ease, box-shadow 0.22s ease, border-color 0.22s ease;
        box-shadow: 0 14px 34px rgba(0,0,0,0.24);
        backdrop-filter: blur(10px);
        height: 100%;
    }

    .movie-card:hover {
        transform: translateY(-6px) scale(1.01);
        box-shadow: 0 22px 46px rgba(0,0,0,0.34);
        border-color: rgba(229,9,20,0.35);
    }

    .movie-title {
        font-size: 1.08rem;
        font-weight: 800;
        color: #f8fafc;
        margin-top: 0.7rem;
        margin-bottom: 0.25rem;
        line-height: 1.25;
    }

    .movie-meta {
        color: #cbd5e1;
        font-size: 0.9rem;
        margin-bottom: 0.45rem;
    }

    .score-badge {
        display: inline-block;
        padding: 0.3rem 0.7rem;
        border-radius: 999px;
        background: linear-gradient(90deg, rgba(229,9,20,0.20), rgba(59,130,246,0.20));
        border: 1px solid rgba(255,255,255,0.09);
        color: #ffffff;
        font-weight: 800;
        font-size: 0.86rem;
        margin-bottom: 0.6rem;
    }

    .featured-card {
        padding: 1rem;
        border-radius: 24px;
        background: linear-gradient(180deg, rgba(255,255,255,0.05), rgba(255,255,255,0.03));
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 16px 40px rgba(0,0,0,0.28);
        margin-bottom: 1.4rem;
    }

    .info-box {
        padding: 1rem 1.1rem;
        border-radius: 18px;
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        color: #e5e7eb;
        margin-bottom: 1rem;
    }

    .stTextInput > div > div > input {
        background: rgba(255,255,255,0.07);
        border: 1px solid rgba(255,255,255,0.09);
        color: white;
        border-radius: 18px;
        padding: 0.95rem 1rem;
        font-size: 1rem;
    }

    .stButton button {
        background: linear-gradient(90deg, #e50914 0%, #b91c1c 100%);
        color: white;
        border: none;
        border-radius: 14px;
        font-weight: 700;
        padding: 0.7rem 1.05rem;
        transition: all 0.18s ease;
    }

    .stButton button:hover {
        transform: translateY(-1px);
        box-shadow: 0 10px 20px rgba(229,9,20,0.25);
    }

    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        padding: 0.7rem 0.8rem;
        border-radius: 18px;
    }

    .footer-note {
        color: #94a3b8;
        font-size: 0.86rem;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# --- SESSION STATE ---
defaults = {
    "input_query": "",
    "submitted_query": "",
    "results": [],
    "top_result": None,
    "attributes": {},
    "hyde_text": "",
    "do_search": False,
    "keep_rejections": False,
    "rejected_keys": [],
    "feedback_message": "",
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


def queue_quick_query(text: str):
    st.session_state.input_query = text
    st.session_state.keep_rejections = False
    st.session_state.do_search = True


def queue_search():
    st.session_state.keep_rejections = False
    st.session_state.do_search = True


def result_key(item: dict) -> str:
    return f"{item.get('title', '')}|{item.get('year', '')}|{item.get('media_type', '')}"


def current_result_keys(limit: int = 10) -> list[str]:
    return [result_key(item) for item in st.session_state.results[:limit]]


def vote_up_results():
    st.session_state.feedback_message = "Nice. I will keep this result set style for your query."


def vote_down_results():
    keys = current_result_keys(10)
    if not keys:
        return

    for key in keys:
        if key not in st.session_state.rejected_keys:
            st.session_state.rejected_keys.append(key)

    st.session_state.feedback_message = "Got it. I removed these results and searched again."
    st.session_state.keep_rejections = True
    st.session_state.do_search = True


def clear_search():
    st.session_state.input_query = ""
    st.session_state.submitted_query = ""
    st.session_state.results = []
    st.session_state.top_result = None
    st.session_state.attributes = {}
    st.session_state.hyde_text = ""
    st.session_state.rejected_keys = []
    st.session_state.feedback_message = ""
    st.session_state.keep_rejections = False
    st.session_state.do_search = False


def run_search(query_text: str, keep_rejections: bool = False):
    clean_query = (query_text or "").strip()
    if not clean_query:
        clear_search()
        return

    if not keep_rejections:
        st.session_state.rejected_keys = []

    with st.spinner("Analyzing memory clues and searching the movie + TV corpus..."):
        hyde_text = hyde_expand_query(client_ai, clean_query)
        final_query = clean_query if not hyde_text else f"{clean_query}. {hyde_text}"

        query_vector = model.encode(final_query).tolist()
        raw = collection.query(query_embeddings=[query_vector], n_results=30)

        candidates = []
        num_hits = len(raw["ids"][0])

        for i in range(num_hits):
            candidates.append({
                "rank": i + 1,
                "doc": raw["documents"][0][i] or "",
                "meta": raw["metadatas"][0][i] or {},
                "dist": raw["distances"][0][i] if raw.get("distances") else None,
            })

        attributes = extract_attributes(client_ai, clean_query)
        results = rerank(clean_query, candidates, attributes)

        rejected_set = set(st.session_state.rejected_keys)
        if rejected_set:
            results = [r for r in results if result_key(r) not in rejected_set]

        st.session_state.submitted_query = clean_query
        st.session_state.results = results
        st.session_state.top_result = results[0] if results else None
        st.session_state.attributes = attributes
        st.session_state.hyde_text = hyde_text
        st.session_state.keep_rejections = False
        st.session_state.do_search = False


# --- SIDEBAR ---
with st.sidebar:
    st.markdown("## 🎬 Movie Detective")
    st.caption("Hybrid movie + TV retrieval for vague human memory")

    st.metric("Titles Indexed", f"{collection.count():,}")

    st.markdown("### Quick Search Ideas")
    st.button(
        "Animated Bible movie about Moses",
        use_container_width=True,
        on_click=queue_quick_query,
        args=("animated Bible movie about Moses with great songs",)
    )
    st.button(
        "Scary doll movie",
        use_container_width=True,
        on_click=queue_quick_query,
        args=("scary doll horror movie with a possessed doll",)
    )
    st.button(
        "Dreams inside dreams",
        use_container_width=True,
        on_click=queue_quick_query,
        args=("movie about dreams inside dreams and stealing ideas",)
    )
    st.button(
        "Ship hits iceberg",
        use_container_width=True,
        on_click=queue_quick_query,
        args=("movie about a ship hitting an iceberg and tragic romance",)
    )

    st.markdown("### About the system")
    st.markdown("""
<div class="info-box">
Movie Detective was developed by Charles Appiah Manu and Bryan Chau.It combines semantic retrieval, HyDE query expansion, structured clue extraction, and metadata-aware reranking to identify movies and TV series from incomplete memories.
</div>
""", unsafe_allow_html=True)

    st.caption("Built for discovery, not exact-title lookup.")

# --- HERO ---
st.markdown("""
<div class="hero-wrap">
    <div class="hero-title">Movie Detective</div>
    <div class="hero-subtitle">
        Describe any movie or TV show the way you actually remember it — scenes, fragments, characters, setting, mood, soundtrack, or plot clues.
        Movie Detective interprets the memory and surfaces likely movie/TV matches as visual cards.
    </div>
    <div class="pill-row">
        <span class="pill">Semantic Search</span>
        <span class="pill">HyDE Expansion</span>
        <span class="pill">Metadata Reranking</span>
        <span class="pill">Visual Discovery</span>
    </div>
</div>
""", unsafe_allow_html=True)

st.text_input(
    "Search",
    key="input_query",
    placeholder="e.g. father fish searching the ocean for his lost son"
)

btn_col1, btn_col2 = st.columns(2)
with btn_col1:
    st.button("Search", use_container_width=True, on_click=queue_search)
with btn_col2:
    st.button("Clear", use_container_width=True, on_click=clear_search)

if st.session_state.do_search:
    run_search(st.session_state.input_query, keep_rejections=st.session_state.keep_rejections)

results = st.session_state.results
top = st.session_state.top_result
attributes = st.session_state.attributes

if results and top:
    st.markdown('<div class="section-title">Top Match</div>', unsafe_allow_html=True)

    if st.session_state.feedback_message:
        st.info(st.session_state.feedback_message)

    fc1, fc2 = st.columns([1.05, 1.95])

    with fc1:
        if top.get("poster"):
            st.image(top["poster"], use_container_width=True)

    with fc2:
        st.markdown(
            f"<div class='movie-title' style='font-size:1.9rem'>{top['title']} ({top['year']})</div>",
            unsafe_allow_html=True
        )
        if top.get("media_label"):
            st.caption(top["media_label"])
        st.markdown(
            f"<div class='score-badge'>Match Score: {top['score']:.1f}</div>",
            unsafe_allow_html=True
        )
        if top.get("genres"):
            st.markdown(
                f"<div class='movie-meta'>{top['genres']}</div>",
                unsafe_allow_html=True
            )
        st.write(top.get("overview", "No overview available."))

        why_bits = top.get("why", [])
        if why_bits:
            st.caption("Why it matched: " + " • ".join(why_bits[:5]))

    st.markdown('<div class="section-title">Rate These Results</div>', unsafe_allow_html=True)
    st.caption("These buttons apply to the current top 10 results, not only the featured card.")

    fb1, fb2 = st.columns(2)
    with fb1:
        st.button("👏 My movie is in the Top 5", use_container_width=True, on_click=vote_up_results)
    with fb2:
        st.button(" 🫤 Movie not in results, Re-search", use_container_width=True, on_click=vote_down_results)

    if attributes:
        clues = []
        for label in [
            "title_hint",
            "franchise",
            "genres",
            "themes",
            "setting",
            "release_period",
            "setting_period",
            "characters",
            "keywords"
        ]:
            value = attributes.get(label)
            if isinstance(value, list) and value:
                clues.append(f"**{label.replace('_', ' ').title()}:** {', '.join(value)}")
            elif isinstance(value, str) and value.strip():
                clues.append(f"**{label.replace('_', ' ').title()}:** {value}")

        if clues:
            st.markdown('<div class="section-title">Interpreted Memory Clues</div>', unsafe_allow_html=True)
            st.markdown(
                "<div class='info-box'>" + "<br>".join(clues) + "</div>",
                unsafe_allow_html=True
            )

    st.markdown('<div class="section-title">Browse Results</div>', unsafe_allow_html=True)

    cols = st.columns(3)
    display_results = results[:10]
    for i, movie in enumerate(display_results):
        with cols[i % 3]:
            if movie.get("poster"):
                st.image(movie["poster"], use_container_width=True)

            st.markdown(
                f"<div class='movie-title'>{movie['title']} ({movie['year']})</div>",
                unsafe_allow_html=True
            )

            if movie.get("media_label"):
                st.caption(movie["media_label"])

            meta_line = movie["genres"] if movie.get("genres") else "Metadata available"
            st.markdown(
                f"<div class='movie-meta'>{meta_line}</div>",
                unsafe_allow_html=True
            )

            st.markdown(
                f"<div class='score-badge'>Match Score: {movie['score']:.1f}</div>",
                unsafe_allow_html=True
            )

            safe_score = max(0.0, min(100.0, float(movie["score"])))
            st.progress(safe_score / 100.0)

            with st.expander("View details"):
                st.write(movie.get("overview", "No overview available."))
                if movie.get("keywords"):
                    st.caption(f"Keywords: {movie['keywords']}")
                if movie.get("why"):
                    st.caption("Why it matched: " + " • ".join(movie["why"][:5]))

    st.markdown(
        "<div class='footer-note'>Tip: add setting, release period, story period, franchise, relationship, or a very specific scene to improve accuracy.</div>",
        unsafe_allow_html=True
    )
else:
    if st.session_state.feedback_message:
        st.info(st.session_state.feedback_message)

    st.markdown("""
<div class="section-title">Start with a memory fragment</div>
<div class="info-box">
Examples:
<ul>
<li>animated Bible movie about Moses</li>
<li>scary doll horror movie</li>
<li>movie where a ship crashes into an iceberg</li>
<li>a Spanish series where red-suited robbers plan major heists</li>
<li>a detective hunts a serial killer in the rain</li>
<li>father fish searching the ocean for his lost son</li>
<li>an action movie about a retired assassin avenging his dog</li>
<li>a movie set in the 1980s with kids and supernatural events</li>
</ul>
</div>
""", unsafe_allow_html=True)