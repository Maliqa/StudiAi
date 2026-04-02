import streamlit as st
import chromadb
from openai import OpenAI
import re
import os
import hashlib

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="StudyMate AI – Asisten Belajar",
    page_icon="📚",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

:root {
    --bg: #0a0a0f;
    --surface: #13131a;
    --surface2: #1c1c26;
    --border: #2a2a3a;
    --accent: #7c6bff;
    --accent2: #00e5a0;
    --text: #f0f0f8;
    --text-muted: #777;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg) !important;
    color: var(--text);
}

#MainMenu, footer, header { visibility: hidden; }
[data-testid="collapsedControl"] { display: none; }
.block-container {
    padding: 1rem 1rem 2rem 1rem !important;
    max-width: 640px !important;
}

/* TABS */
[data-testid="stTabs"] button {
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.88rem !important;
    color: var(--text-muted) !important;
    border-radius: 0 !important;
    border-bottom: 2px solid transparent !important;
    background: transparent !important;
    padding: 0.5rem 0.8rem !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom: 2px solid var(--accent) !important;
}
[data-testid="stTabsBar"] {
    background: var(--surface) !important;
    border-bottom: 1px solid var(--border) !important;
    border-radius: 10px 10px 0 0 !important;
}

/* INPUTS */
input, textarea, [data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
}
input::placeholder, textarea::placeholder { color: var(--text-muted) !important; }

label { color: var(--text-muted) !important; font-size: 0.82rem !important; }

/* BUTTONS */
.stButton > button {
    background: var(--accent) !important;
    color: #fff !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.6rem 1.5rem !important;
    width: 100% !important;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.85 !important; }

/* FILE UPLOADER */
[data-testid="stFileUploader"] {
    background: var(--surface2) !important;
    border: 2px dashed var(--border) !important;
    border-radius: 12px !important;
    padding: 12px !important;
}

/* CARDS */
.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 16px;
    margin: 10px 0;
}
.card-accent {
    border-left: 3px solid var(--accent);
}
.card-green {
    border-left: 3px solid var(--accent2);
}

/* SECTION HEADER */
.section-header {
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 0.75rem;
    color: var(--text-muted);
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin: 18px 0 8px 0;
}

/* BADGE */
.badge {
    display: inline-block;
    background: var(--accent);
    color: #fff;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 0.6rem;
    padding: 2px 8px;
    border-radius: 20px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-left: 6px;
    vertical-align: middle;
}
.badge-green {
    background: var(--accent2);
    color: #000;
}

/* QUIZ */
.quiz-question {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    border-radius: 10px;
    padding: 14px 16px;
    margin: 12px 0;
    font-size: 0.95rem;
    line-height: 1.6;
}
.quiz-answer {
    background: #0d2010;
    border: 1px solid #1a4020;
    border-radius: 8px;
    padding: 10px 14px;
    margin-top: 6px;
    font-size: 0.88rem;
    color: #90ffb0;
}

hr { border-color: var(--border) !important; margin: 12px 0 !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
FREE_MODELS = [
    "openrouter/free",
    "deepseek/deepseek-chat-v3-0324:free",
    "meta-llama/llama-3.3-70b-instruct:free",
]
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# ─────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────
for key, default in {
    "api_key": "",
    "doc_loaded": False,
    "doc_name": "",
    "doc_chunks": [],
    "qa_messages": [],
    "summary": "",
    "quiz": [],
    "chroma_collection": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────
def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Bagi teks jadi potongan kecil dengan overlap."""
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i+size])
        chunks.append(chunk)
        i += size - overlap
    return chunks

def call_ai(api_key, messages, max_tokens=1024):
    """Panggil AI via OpenRouter dengan fallback."""
    client = OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=api_key,
        timeout=60.0,
    )
    for model in FREE_MODELS:
        try:
            response = client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                messages=messages
            )
            content = response.choices[0].message.content
            if content and content.strip():
                # Strip HTML tags
                clean = re.sub(r'<[^>]+>', '', content).strip()
                return clean, None
        except Exception as e:
            continue
    return None, "Semua model gagal. Coba lagi."

def get_embedding_simple(text):
    """Simple hash-based pseudo embedding untuk ChromaDB tanpa API."""
    import struct
    words = text.lower().split()
    vec = [0.0] * 128
    for i, word in enumerate(words[:128]):
        h = int(hashlib.md5(word.encode()).hexdigest(), 16)
        vec[i % 128] += (h % 1000) / 1000.0
    norm = sum(x**2 for x in vec) ** 0.5
    if norm > 0:
        vec = [x/norm for x in vec]
    return vec

def load_document_to_chroma(text, doc_name):
    """Load dokumen ke ChromaDB."""
    try:
        chroma_client = chromadb.Client()
        # Hapus collection lama kalau ada
        try:
            chroma_client.delete_collection("studymate")
        except:
            pass
        collection = chroma_client.create_collection("studymate")
        chunks = chunk_text(text)
        embeddings = [get_embedding_simple(c) for c in chunks]
        ids = [f"chunk_{i}" for i in range(len(chunks))]
        collection.add(
            documents=chunks,
            embeddings=embeddings,
            ids=ids
        )
        return collection, chunks, None
    except Exception as e:
        return None, [], str(e)

def search_relevant_chunks(collection, query, n=3):
    """Cari chunk paling relevan dari ChromaDB."""
    try:
        query_embedding = get_embedding_simple(query)
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(n, collection.count())
        )
        return results['documents'][0] if results['documents'] else []
    except:
        return []

def read_uploaded_file(uploaded_file):
    """Baca isi file upload."""
    try:
        if uploaded_file.name.endswith('.txt'):
            return uploaded_file.read().decode('utf-8'), None
        elif uploaded_file.name.endswith('.pdf'):
            try:
                import PyPDF2
                import io
                reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                return text, None
            except ImportError:
                return None, "PyPDF2 tidak terinstall. Jalankan: pip install PyPDF2"
        else:
            return None, "Format tidak didukung. Gunakan .txt atau .pdf"
    except Exception as e:
        return None, str(e)

# ─────────────────────────────────────────────
#  BRAND HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding: 20px 0 8px 0;">
    <div style="font-family:'Syne',sans-serif; font-size:2rem; font-weight:800;
                background: linear-gradient(135deg, #7c6bff, #00e5a0);
                -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                display:inline-block;">
        📚 StudyMate AI
    </div>
    <div style="font-size:0.8rem; color:#555; margin-top:2px;">
        RAG-Powered Study Assistant · Upload & Learn
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  TABS
# ─────────────────────────────────────────────
tab_setup, tab_qa, tab_summary, tab_quiz = st.tabs([
    "⚙️ Setup", "💬 Tanya Jawab", "📝 Ringkasan", "🧠 Quiz"
])

# ══════════════════════════════════════════════
#  TAB 1 — SETUP
# ══════════════════════════════════════════════
with tab_setup:
    st.markdown('<div class="section-header">🔑 API Key</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="card card-accent" style="font-size:0.83rem; color:#777;">
        Gratis di <b style="color:#f0f0f8;">openrouter.ai</b> → Settings → Keys → Create Key
    </div>
    """, unsafe_allow_html=True)

    api_key = st.text_input(
        "api",
        type="password",
        placeholder="sk-or-v1-...",
        value=st.session_state.api_key,
        label_visibility="collapsed"
    )

    st.markdown('<div class="section-header">📄 Upload Materi</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.82rem; color:#555; margin-bottom:8px;">
        Format yang didukung: <b>.txt</b> dan <b>.pdf</b>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "upload",
        type=["txt", "pdf"],
        label_visibility="collapsed"
    )

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🚀 Proses Dokumen"):
        if not api_key:
            st.error("⚠️ Masukkan API Key dulu!")
        elif not uploaded_file:
            st.error("⚠️ Upload file dulu!")
        else:
            with st.spinner("📖 Membaca dokumen..."):
                text, err = read_uploaded_file(uploaded_file)

            if err:
                st.error(f"❌ {err}")
            elif not text or len(text.strip()) < 50:
                st.error("❌ Dokumen terlalu pendek atau kosong.")
            else:
                with st.spinner("🧠 Memproses ke ChromaDB..."):
                    collection, chunks, err2 = load_document_to_chroma(text, uploaded_file.name)

                if err2:
                    st.error(f"❌ ChromaDB error: {err2}")
                else:
                    st.session_state.api_key = api_key
                    st.session_state.doc_loaded = True
                    st.session_state.doc_name = uploaded_file.name
                    st.session_state.doc_chunks = chunks
                    st.session_state.chroma_collection = collection
                    st.session_state.qa_messages = []
                    st.session_state.summary = ""
                    st.session_state.quiz = []
                    st.success(f"✅ Dokumen siap! {len(chunks)} chunk dimuat ke ChromaDB.")
                    st.markdown(f"""
                    <div class="card card-green" style="font-size:0.85rem; margin-top:8px;">
                        📄 <b>{uploaded_file.name}</b><br>
                        <span style="color:#777;">
                            {len(text)} karakter · {len(chunks)} chunk · ChromaDB ready
                        </span>
                    </div>
                    """, unsafe_allow_html=True)

    if st.session_state.doc_loaded:
        st.markdown(f"""
        <div class="card card-green" style="font-size:0.85rem;">
            ✅ Dokumen aktif: <b>{st.session_state.doc_name}</b><br>
            <span style="color:#777;">{len(st.session_state.doc_chunks)} chunk di ChromaDB</span><br>
            <span style="color:#555; font-size:0.78rem;">Buka tab lain untuk mulai belajar!</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style="color:#333; font-size:0.7rem; text-align:center; margin-top:24px;">
        Powered by ChromaDB + OpenRouter · StudyMate AI v1.0
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════
#  TAB 2 — TANYA JAWAB
# ══════════════════════════════════════════════
with tab_qa:
    if not st.session_state.doc_loaded:
        st.markdown("""
        <div style="text-align:center; padding:50px 20px; color:#444;">
            <div style="font-size:2.5rem;">⚙️</div>
            <div style="margin-top:12px; font-size:0.92rem; line-height:1.8;">
                Belum ada dokumen.<br>
                Buka tab <b style="color:#7c6bff;">⚙️ Setup</b> dan upload materi dulu.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="font-family:'Syne',sans-serif; font-size:1rem; font-weight:800; margin-bottom:12px;">
            💬 Tanya tentang <span style="color:#7c6bff;">{st.session_state.doc_name}</span>
            <span class="badge">RAG</span>
        </div>
        """, unsafe_allow_html=True)

        # Chat history
        if not st.session_state.qa_messages:
            st.markdown("""
            <div style="text-align:center; padding:30px 16px; color:#444;">
                <div style="font-size:1.8rem;">💬</div>
                <div style="margin-top:8px; font-size:0.85rem; line-height:1.6;">
                    Tanyakan apapun tentang materi yang kamu upload.<br>
                    AI akan jawab berdasarkan isi dokumen.
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            for msg in st.session_state.qa_messages:
                if msg["role"] == "user":
                    with st.chat_message("user"):
                        st.write(msg["content"])
                else:
                    with st.chat_message("assistant"):
                        st.write(msg["content"])

        # Input
        col_in, col_btn = st.columns([4, 1])
        with col_in:
            question = st.text_input(
                "q",
                placeholder="Tanya sesuatu tentang materi...",
                key="qa_input",
                label_visibility="collapsed"
            )
        with col_btn:
            ask = st.button("➤", use_container_width=True, key="ask_btn")

        if st.session_state.qa_messages:
            if st.button("🔄 Reset", key="reset_qa"):
                st.session_state.qa_messages = []
                st.rerun()

        if ask and question.strip():
            st.session_state.qa_messages.append({"role": "user", "content": question.strip()})

            with st.spinner("🔍 Mencari di dokumen..."):
                # RAG: cari chunk relevan
                relevant = search_relevant_chunks(
                    st.session_state.chroma_collection,
                    question.strip(),
                    n=3
                )
                context = "\n\n---\n\n".join(relevant) if relevant else "Tidak ada konteks ditemukan."

                messages = [
                    {
                        "role": "system",
                        "content": f"""Kamu adalah asisten belajar yang membantu menjawab pertanyaan berdasarkan materi yang diberikan.

KONTEKS DARI DOKUMEN:
{context}

Instruksi:
- Jawab HANYA berdasarkan konteks di atas
- Jika informasi tidak ada di konteks, katakan "Informasi ini tidak ada di materi yang diupload"
- Jawab dalam Bahasa Indonesia
- Gunakan teks biasa, JANGAN gunakan HTML atau tag apapun
- Jawab dengan jelas dan mudah dipahami"""
                    },
                    {"role": "user", "content": question.strip()}
                ]

                reply, error = call_ai(st.session_state.api_key, messages)

            if reply:
                st.session_state.qa_messages.append({"role": "assistant", "content": reply})
            else:
                st.error(f"❌ {error}")
                st.session_state.qa_messages.pop()

            st.rerun()

# ══════════════════════════════════════════════
#  TAB 3 — RINGKASAN
# ══════════════════════════════════════════════
with tab_summary:
    if not st.session_state.doc_loaded:
        st.markdown("""
        <div style="text-align:center; padding:50px 20px; color:#444;">
            <div style="font-size:2.5rem;">⚙️</div>
            <div style="margin-top:12px; font-size:0.92rem; line-height:1.8;">
                Belum ada dokumen.<br>
                Buka tab <b style="color:#7c6bff;">⚙️ Setup</b> dan upload materi dulu.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="font-family:'Syne',sans-serif; font-size:1rem; font-weight:800; margin-bottom:12px;">
            📝 Ringkasan <span style="color:#7c6bff;">{st.session_state.doc_name}</span>
        </div>
        """, unsafe_allow_html=True)

        if st.session_state.summary:
            st.markdown(f"""
            <div class="card card-accent" style="font-size:0.92rem; line-height:1.8; white-space:pre-wrap;">
                {st.session_state.summary}
            </div>
            """, unsafe_allow_html=True)
            if st.button("🔄 Generate Ulang"):
                st.session_state.summary = ""
                st.rerun()
        else:
            st.markdown("""
            <div style="text-align:center; padding:20px; color:#444; font-size:0.88rem;">
                Klik tombol di bawah untuk generate ringkasan otomatis dari materi.
            </div>
            """, unsafe_allow_html=True)

            if st.button("✨ Buat Ringkasan"):
                with st.spinner("📝 Merangkum dokumen..."):
                    # Ambil sample chunk dari awal, tengah, akhir
                    chunks = st.session_state.doc_chunks
                    n = len(chunks)
                    sample = []
                    if n > 0: sample.append(chunks[0])
                    if n > 2: sample.append(chunks[n//2])
                    if n > 1: sample.append(chunks[-1])
                    sample_text = "\n\n".join(sample)

                    messages = [
                        {
                            "role": "system",
                            "content": """Kamu adalah asisten belajar yang ahli membuat ringkasan.
Buat ringkasan yang jelas dan terstruktur dalam Bahasa Indonesia.
Gunakan teks biasa saja. JANGAN gunakan HTML atau tag apapun.
Format: tulis poin-poin penting dengan awalan nomor (1. 2. 3. dst)"""
                        },
                        {
                            "role": "user",
                            "content": f"Buat ringkasan dari materi berikut:\n\n{sample_text}"
                        }
                    ]
                    summary, error = call_ai(st.session_state.api_key, messages, max_tokens=800)

                if summary:
                    st.session_state.summary = summary
                    st.rerun()
                else:
                    st.error(f"❌ {error}")

# ══════════════════════════════════════════════
#  TAB 4 — QUIZ
# ══════════════════════════════════════════════
with tab_quiz:
    if not st.session_state.doc_loaded:
        st.markdown("""
        <div style="text-align:center; padding:50px 20px; color:#444;">
            <div style="font-size:2.5rem;">⚙️</div>
            <div style="margin-top:12px; font-size:0.92rem; line-height:1.8;">
                Belum ada dokumen.<br>
                Buka tab <b style="color:#7c6bff;">⚙️ Setup</b> dan upload materi dulu.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="font-family:'Syne',sans-serif; font-size:1rem; font-weight:800; margin-bottom:12px;">
            🧠 Quiz dari <span style="color:#7c6bff;">{st.session_state.doc_name}</span>
            <span class="badge badge-green">AUTO-GENERATED</span>
        </div>
        """, unsafe_allow_html=True)

        if st.session_state.quiz:
            for i, item in enumerate(st.session_state.quiz):
                st.markdown(f"""
                <div class="quiz-question">
                    <b>Soal {i+1}:</b> {item.get('question', '')}
                </div>
                """, unsafe_allow_html=True)
                with st.expander(f"💡 Lihat Jawaban #{i+1}"):
                    st.markdown(f"""
                    <div class="quiz-answer">
                        {item.get('answer', '')}
                    </div>
                    """, unsafe_allow_html=True)

            if st.button("🔄 Generate Quiz Baru"):
                st.session_state.quiz = []
                st.rerun()
        else:
            st.markdown("""
            <div style="text-align:center; padding:20px; color:#444; font-size:0.88rem;">
                Klik tombol di bawah untuk generate soal quiz otomatis dari materi.
            </div>
            """, unsafe_allow_html=True)

            num_q = st.slider("Jumlah soal", min_value=3, max_value=10, value=5)

            if st.button("🧠 Generate Quiz"):
                with st.spinner("🎯 Membuat soal quiz..."):
                    chunks = st.session_state.doc_chunks
                    n = len(chunks)
                    # Ambil sample dari dokumen
                    indices = [0, n//4, n//2, 3*n//4, n-1] if n >= 5 else list(range(n))
                    sample = "\n\n".join([chunks[i] for i in indices if i < n])

                    messages = [
                        {
                            "role": "system",
                            "content": f"""Kamu adalah guru yang membuat soal quiz dari materi pelajaran.
Buat {num_q} soal pilihan ganda atau esai singkat dalam Bahasa Indonesia.
Format WAJIB seperti ini (tanpa HTML, tanpa markdown, teks biasa):

SOAL 1: [pertanyaan]
JAWABAN 1: [jawaban lengkap]

SOAL 2: [pertanyaan]
JAWABAN 2: [jawaban lengkap]

Dan seterusnya. JANGAN gunakan format lain."""
                        },
                        {
                            "role": "user",
                            "content": f"Buat {num_q} soal quiz dari materi berikut:\n\n{sample}"
                        }
                    ]
                    raw_quiz, error = call_ai(st.session_state.api_key, messages, max_tokens=1200)

                if raw_quiz:
                    # Parse soal & jawaban
                    quiz_items = []
                    soal_pattern = re.findall(
                        r'SOAL\s*\d+\s*:\s*(.*?)\s*JAWABAN\s*\d+\s*:\s*(.*?)(?=SOAL\s*\d+|$)',
                        raw_quiz, re.DOTALL | re.IGNORECASE
                    )
                    for q, a in soal_pattern:
                        quiz_items.append({
                            "question": q.strip(),
                            "answer": a.strip()
                        })

                    if quiz_items:
                        st.session_state.quiz = quiz_items
                    else:
                        # Fallback: split manual
                        lines = [l.strip() for l in raw_quiz.split('\n') if l.strip()]
                        fallback = []
                        i = 0
                        while i < len(lines) - 1:
                            fallback.append({"question": lines[i], "answer": lines[i+1]})
                            i += 2
                        st.session_state.quiz = fallback[:num_q] if fallback else [{"question": raw_quiz, "answer": "Lihat materi"}]
                    st.rerun()
                else:
                    st.error(f"❌ {error}")

