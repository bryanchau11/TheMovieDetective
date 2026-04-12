import os
from pathlib import Path
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


@st.cache_data(ttl=1800)
def get_unique_title_count() -> int:
    try:
        payload = collection.get(include=["metadatas"])
    except Exception:
        return 0

    metadatas = payload.get("metadatas") or []
    unique_keys = set()
    for meta in metadatas:
        if not isinstance(meta, dict):
            continue
        title = (meta.get("title") or "").strip().lower()
        year = (meta.get("year") or "").strip()
        media_type = (meta.get("media_type") or "").strip().lower()
        if not title:
            continue
        unique_keys.add((title, year, media_type))

    return len(unique_keys)


try:
    model, collection = init()
except Exception:
    st.error("Database not found. Run ingest.py first.")
    st.stop()

# --- STYLES ---
BASE_DIR = Path(__file__).resolve().parent


def inject_css() -> None:
    css_path = BASE_DIR / "styles" / "app.css"
    if not css_path.exists():
        st.warning("Stylesheet missing: styles/app.css")
        return

    st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


inject_css()

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
    st.session_state.feedback_message = "Thank you for your feedback 😊."


def vote_down_results():
    keys = current_result_keys(10)
    if not keys:
        return

    for key in keys:
        if key not in st.session_state.rejected_keys:
            st.session_state.rejected_keys.append(key)

    st.session_state.feedback_message = "Got it. Refining search results."
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
# --- QUICK IDEAS ---
quick_ideas = [
    ("Scary doll movie", "scary doll horror movie with a possessed doll"),
    ("Dreams inside dreams", "movie about dreams inside dreams and stealing ideas"),
    ("Ship hits iceberg", "movie about a ship hitting an iceberg and tragic romance"),
    ("Toys that come to life", "cartoon about toys that come to life"),
]

clue_order = [
    "genres",
    "themes",
    "setting",
    "characters",
    "keywords",
    "franchise",
    "release_period",
    "setting_period",
    "title_hint",
]

def close_html_div() -> None:
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    f"""
<div class="top-nav">
  <div>
    <div class="brand-title">MOVIE DETECTIVE</div>
    <div class="brand-sub">Movie Retrieval System</div>
  </div>
  <div>
    <div class="archive-stat">Movies and TV shows Indexed</div>
    <div class="archive-value">{get_unique_title_count():,} TITLES</div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

left_col, right_col = st.columns([1.05, 1.4], gap="large")

with left_col:
    st.markdown('<div class="left-rail">', unsafe_allow_html=True)
    st.markdown(
        """
<div class="panel-title">Your <span class="accent"> movie </span>search ends here.</div>
<div class="panel-copy">
Vague memory? No problem. Describe the vibe, a character, or the plot and we’ll help you find what you've been searching for..
</div>
""",
        unsafe_allow_html=True,
    )

    st.text_area(
        "Describe your memory",
        key="input_query",
        label_visibility="collapsed",
        placeholder="Type what you remember: was it a cartoon, anime, or live-action? Mention key scenes, characters, time period, or ending details.",
    )

    action_a, action_b = st.columns([1.2, 0.8])
    with action_a:
        st.button("Find Movie", use_container_width=True, on_click=queue_search)
    with action_b:
        st.button("Clear", use_container_width=True, on_click=clear_search)

    st.markdown('<div class="chip-label">Quick Memory Sparks</div>', unsafe_allow_html=True)
    q1, q2 = st.columns(2)
    with q1:
        st.button(quick_ideas[0][0], key="q0", use_container_width=True, on_click=queue_quick_query, args=(quick_ideas[0][1],))
        st.button(quick_ideas[2][0], key="q2", use_container_width=True, on_click=queue_quick_query, args=(quick_ideas[2][1],))
    with q2:
        st.button(quick_ideas[1][0], key="q1", use_container_width=True, on_click=queue_quick_query, args=(quick_ideas[1][1],))
        st.button(quick_ideas[3][0], key="q3", use_container_width=True, on_click=queue_quick_query, args=(quick_ideas[3][1],))

    st.caption("Built by Charles Appiah Manu and Bryan Chau")
    close_html_div()

if st.session_state.do_search:
    run_search(st.session_state.input_query, keep_rejections=st.session_state.keep_rejections)

results = st.session_state.results
top = st.session_state.top_result
attributes = st.session_state.attributes

with right_col:
    if st.session_state.feedback_message:
        st.info(st.session_state.feedback_message)

    if results and top:
        st.markdown('<div class="section-title">Primary Match</div>', unsafe_allow_html=True)
        st.markdown('<div class="featured-shell">', unsafe_allow_html=True)
        top_img_col, top_content_col = st.columns([0.9, 1.3], gap="medium")

        with top_img_col:
            if top.get("poster"):
                st.image(top["poster"], use_container_width=True)
            else:
                st.markdown('<div class="empty-box">Poster unavailable</div>', unsafe_allow_html=True)

        with top_content_col:
            safe_top_score = max(0.0, min(100.0, float(top.get("score", 0.0))))
            st.markdown(
                f"<div class='movie-title'>{top['title']} <span class='movie-meta'>({top['year']})</span></div>",
                unsafe_allow_html=True,
            )
            if top.get("media_label"):
                st.caption(top["media_label"])
            st.markdown(
                f"<div class='score-badge'>Reliability: {safe_top_score:.1f}%</div>",
                unsafe_allow_html=True,
            )
            if top.get("genres"):
                st.markdown(f"<div class='movie-meta'>{top['genres']}</div>", unsafe_allow_html=True)

            st.write(top.get("overview", "No overview available."))

            why_bits = top.get("why", [])
            if why_bits:
                st.markdown(
                    "<div class='detective-box'><strong>Detective Notes:</strong><br>"
                    + " • ".join(why_bits[:5])
                    + "</div>",
                    unsafe_allow_html=True,
                )

            st.markdown("<div class='feedback-prompt'>Was your movie found?</div>", unsafe_allow_html=True)
            r1, r2 = st.columns(2)
            with r1:
                st.button("Yes, it's in the top 5", use_container_width=True, on_click=vote_up_results)
            with r2:
                st.button("No, it's not in the results", use_container_width=True, on_click=vote_down_results)

        close_html_div()

        if attributes:
            rendered_clues = []
            for label in clue_order:
                value = attributes.get(label)
                if isinstance(value, list) and value:
                    rendered_clues.append(f"<span class='clue-chip'>{label.replace('_', ' ').title()}: {', '.join(value)}</span>")
                elif isinstance(value, str) and value.strip():
                    rendered_clues.append(f"<span class='clue-chip'>{label.replace('_', ' ').title()}: {value}</span>")
            if rendered_clues:
                st.markdown('<div class="section-title">Extracted Evidence</div>', unsafe_allow_html=True)
                st.markdown("<div class='clue-wrap'>" + "".join(rendered_clues) + "</div>", unsafe_allow_html=True)

        st.markdown('<div class="section-title">Other Potential Suspects</div>', unsafe_allow_html=True)
        lineup = results[1:9] if len(results) > 1 else []
        if lineup:
            grid_cols = st.columns(4)
            for idx, movie in enumerate(lineup):
                with grid_cols[idx % 4]:
                    st.markdown('<div class="small-card">', unsafe_allow_html=True)
                    if movie.get("poster"):
                        st.image(movie["poster"], use_container_width=True)
                    else:
                        st.markdown('<div class="empty-box">No poster</div>', unsafe_allow_html=True)
                    st.markdown(
                        f"<div class='small-card-title'>{movie['title']}</div>",
                        unsafe_allow_html=True,
                    )
                    media_label = movie.get("media_label", "Movie")
                    st.markdown(
                        f"<div class='small-card-meta'>{movie['year']} • {media_label}</div>",
                        unsafe_allow_html=True,
                    )
                    with st.expander("Investigate"):
                        st.write(movie.get("overview", "No overview available."))
                    close_html_div()
        else:
            st.markdown('<div class="empty-box">No additional suspects from this search yet.</div>', unsafe_allow_html=True)
    else:
        st.markdown(
            """
<div class="empty-box">
  <h3 style="margin:0;color:#f8fafc;">Awaiting Input</h3>
  <p style="margin-top:0.7rem;">Enter your memory fragments on the left to begin retrieval.</p>
</div>
""",
            unsafe_allow_html=True,
        )

st.markdown(
    "<div class='footer-note'>Powered by Semantic Retrieval Technology • Built for Discovery</div>",
    unsafe_allow_html=True,
)