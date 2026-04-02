<div align="center">

```
███████╗████████╗██╗   ██╗██████╗ ██╗   ██╗███╗   ███╗ █████╗ ████████╗███████╗
██╔════╝╚══██╔══╝██║   ██║██╔══██╗╚██╗ ██╔╝████╗ ████║██╔══██╗╚══██╔══╝██╔════╝
███████╗   ██║   ██║   ██║██║  ██║ ╚████╔╝ ██╔████╔██║███████║   ██║   █████╗  
╚════██║   ██║   ██║   ██║██║  ██║  ╚██╔╝  ██║╚██╔╝██║██╔══██║   ██║   ██╔══╝  
███████║   ██║   ╚██████╔╝██████╔╝   ██║   ██║ ╚═╝ ██║██║  ██║   ██║   ███████╗
╚══════╝   ╚═╝    ╚═════╝ ╚═════╝    ╚═╝   ╚═╝     ╚═╝╚═╝  ╚═╝   ╚═╝   ╚══════╝
                              AI  ·  Asisten Belajar Berbasis RAG
```

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_DB-7C6BFF?style=for-the-badge)](https://trychroma.com)
[![OpenRouter](https://img.shields.io/badge/OpenRouter-Free_LLM-00E5A0?style=for-the-badge)](https://openrouter.ai)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

**StudyMate AI** adalah aplikasi asisten belajar berbasis **RAG (Retrieval Augmented Generation)** — upload dokumen PDF atau TXT, lalu tanya jawab, buat ringkasan, dan generate quiz otomatis menggunakan AI.

[🚀 Demo](#demo) · [✨ Fitur](#fitur) · [⚙️ Instalasi](#instalasi) · [📖 Cara Pakai](#cara-pakai) · [🧠 Arsitektur RAG](#arsitektur-rag)

</div>

---

## 🎯 Tentang Project

**StudyMate AI** mendemonstrasikan implementasi penuh pipeline RAG dari nol tanpa framework berat — hanya Python, ChromaDB, dan OpenRouter API.

Upload materi kuliah, buku, modul, atau dokumen apapun → AI akan memahami isinya dan siap menjawab pertanyaan, meringkas, serta membuat soal quiz secara otomatis.

> **Kenapa project ini penting untuk portfolio AI Engineer?**
> RAG adalah skill paling dicari di industri AI saat ini. Project ini membuktikan pemahaman end-to-end: document ingestion → chunking → embedding → vector search → generation.

---

## ✨ Fitur

| Tab | Fitur | Deskripsi |
|-----|-------|-----------|
| ⚙️ **Setup** | Upload Dokumen | Support file `.txt` dan `.pdf` |
| ⚙️ **Setup** | ChromaDB Ingestion | Auto-chunk → embed → store ke vector DB |
| 💬 **Tanya Jawab** | Contextual Q&A | Jawaban AI berdasarkan isi dokumen (RAG) |
| 📝 **Ringkasan** | Auto Summary | Ringkasan terstruktur dari materi |
| 🧠 **Quiz** | Quiz Generator | Generate soal otomatis, bisa pilih jumlah soal |

---

## 🧠 Arsitektur RAG

```
┌─────────────────────────────────────────────────────────────────┐
│                        RAG PIPELINE                             │
│                                                                 │
│  [Upload PDF/TXT]                                               │
│        │                                                        │
│        ▼                                                        │
│  [Text Extraction]  ──►  [Chunking (800 words + 100 overlap)]   │
│                                   │                             │
│                                   ▼                             │
│                          [Embedding Generation]                 │
│                                   │                             │
│                                   ▼                             │
│                          [ChromaDB Vector Store]                │
│                                                                 │
│  User Query ──► [Query Embedding]                               │
│                        │                                        │
│                        ▼                                        │
│               [Semantic Search → Top-3 Chunks]                  │
│                        │                                        │
│                        ▼                                        │
│          [Augmented Prompt = Context + Query]                   │
│                        │                                        │
│                        ▼                                        │
│             [LLM via OpenRouter → Response]                     │
└─────────────────────────────────────────────────────────────────┘
```

**Kenapa RAG lebih baik dari langsung tanya LLM?**

- ✅ Jawaban berdasarkan dokumen spesifik lo, bukan data training model
- ✅ Tidak hallucinate karena ada konteks nyata
- ✅ Bisa dipakai untuk dokumen privat/internal
- ✅ Lebih hemat token daripada kirim seluruh dokumen

---

## 🛠️ Tech Stack

```yaml
Frontend:       Streamlit 1.32+
Vector DB:      ChromaDB (local, in-memory)
LLM Engine:     OpenRouter API (Free Tier)
Models:         openrouter/free → DeepSeek V3 → Llama 3.3 70B (fallback chain)
PDF Parsing:    PyPDF2
Language:       Python 3.9+
Embedding:      Custom hash-based vector (no API cost)
```

---

## ⚙️ Instalasi

### Prerequisites
- Python 3.9+
- OpenRouter API Key gratis di [openrouter.ai](https://openrouter.ai)

### Steps

```bash
# 1. Clone repository
git clone https://github.com/Maliqa/studi_asisten.git
cd studi_asisten

# 2. Install dependencies
pip install streamlit openai chromadb PyPDF2

# 3. Jalankan aplikasi
streamlit run studi_asisten.py
```

### Dependencies

```txt
streamlit>=1.32.0
openai>=1.0.0
chromadb>=0.4.0
PyPDF2>=3.0.0
```

---

## 📖 Cara Pakai

### 1. Setup
- Buka tab **⚙️ Setup**
- Masukkan **OpenRouter API Key** (gratis di openrouter.ai)
- Upload file **`.pdf`** atau **`.txt`**
- Klik **🚀 Proses Dokumen** — dokumen akan di-chunk dan disimpan ke ChromaDB

### 2. Tanya Jawab (RAG)
- Buka tab **💬 Tanya Jawab**
- Ketik pertanyaan tentang isi materi
- AI akan mencari bagian relevan di ChromaDB, lalu menjawab berdasarkan konteks tersebut

### 3. Buat Ringkasan
- Buka tab **📝 Ringkasan**
- Klik **✨ Buat Ringkasan**
- AI menganalisis bagian awal, tengah, dan akhir dokumen untuk membuat ringkasan terstruktur

### 4. Generate Quiz
- Buka tab **🧠 Quiz**
- Pilih jumlah soal (3–10)
- Klik **🧠 Generate Quiz**
- Soal dan jawaban di-generate otomatis dari isi materi
- Klik **💡 Lihat Jawaban** per soal untuk reveal jawaban

---

## 🗂️ Struktur Project

```
studi_asisten/
│
├── studi_asisten.py      # Main application (RAG pipeline + UI)
├── requirements.txt      # Python dependencies
└── README.md             # Dokumentasi
```

---

## 🔍 Cara Kerja Teknis

### Document Chunking
```python
# Dokumen dipotong jadi chunk 800 kata dengan overlap 100 kata
# Overlap mencegah informasi terpotong di tengah kalimat
CHUNK_SIZE    = 800
CHUNK_OVERLAP = 100
```

### Vector Search
```python
# Query dikonversi ke vector → dicari top-3 chunk paling relevan
# Menggunakan cosine similarity di ChromaDB
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=3
)
```

### Augmented Prompt
```python
# Konteks dari ChromaDB digabung dengan pertanyaan user
# Memastikan LLM menjawab berdasarkan dokumen, bukan data training
messages = [
    {"role": "system", "content": f"KONTEKS:\n{context}\n\nJawab berdasarkan konteks di atas."},
    {"role": "user", "content": question}
]
```

---

## 🚀 Roadmap

- [x] RAG pipeline dengan ChromaDB
- [x] Q&A berbasis dokumen
- [x] Auto-summarization
- [x] Quiz generator otomatis
- [x] Fallback model chain (3 model)
- [ ] Support multi-dokumen sekaligus
- [ ] Embedding dengan model proper (sentence-transformers)
- [ ] Export hasil quiz ke PDF
- [ ] Memory chat history antar sesi
- [ ] Support format DOCX & PPTX

---

## 💡 Konsep yang Didemonstrasikan

Project ini menunjukkan pemahaman tentang:

- **RAG Architecture** — end-to-end pipeline dari document ke response
- **Vector Database** — penyimpanan dan retrieval berbasis similarity
- **Chunking Strategy** — teknik memotong dokumen dengan overlap
- **Prompt Engineering** — menyusun augmented prompt yang efektif
- **LLM Integration** — memanggil API dengan fallback handling
- **Production UI** — Streamlit app yang mobile-friendly dan siap pakai

---

## 👨‍💻 Author

**Malik Abdul Azis**
- GitHub: [@Maliqa](https://github.com/Maliqa)
- LinkedIn: [malik-abdul-aziz](https://www.linkedin.com/in/malik-abdul-aziz-153129176/)
- Email: Maliqaaziz11@gmail.com

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

---

<div align="center">

**📚 Built with Python · ChromaDB · OpenRouter · Streamlit**

*StudyMate AI — Belajar lebih cerdas dengan RAG.*

</div>

