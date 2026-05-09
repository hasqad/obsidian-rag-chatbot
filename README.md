# Privat Notat-AI

En fullstendig lokal RAG-chatbot (Retrieval-Augmented Generation) som lar deg chatte med dine egne Obsidian- eller Notion-notater. Ingen sky, ingen API-nøkler, ingen deling — alt kjører på din egen maskin.

![Demo](https://img.shields.io/badge/status-fungerende-brightgreen) ![Python](https://img.shields.io/badge/python-3.10%2B-blue) ![Ollama](https://img.shields.io/badge/LLM-Ollama-orange)

---

## Hvordan det fungerer

```
Notater (.md/.txt)
        ↓
   ingest.py
   • Leser og renser markdown
   • Deler i overlappende chunks
   • Genererer embeddings (all-MiniLM-L6-v2)
        ↓
   ChromaDB (lokal vektordatabase på disk)
        ↓
   app.py (FastAPI)
   • Tar imot spørsmål fra brukeren
   • Embedder spørsmålet
   • Henter topp 5 relevante chunks
   • Sender kontekst + spørsmål til Ollama
        ↓
   Ollama (lokal LLM: llama3, qwen3, mistral, osv.)
        ↓
   Svar med kildevisning i nettleseren
```

**Teknologier:**
- `sentence-transformers` — lokale embeddings (~80 MB, all-MiniLM-L6-v2)
- `ChromaDB` — lokal vektordatabase med cosine-søk
- `Ollama` — lokal LLM-kjøring
- `FastAPI` + Server-Sent Events — streaming backend
- Vanilla HTML/CSS/JS — chat-grensesnitt uten eksterne avhengigheter

---

## Oppsett

### 1. Installer Ollama

```bash
# macOS / Linux
curl -fsSL https://ollama.com/install.sh | sh

# Last ned en modell (velg én)
ollama pull llama3        # Anbefalt — god balanse mellom kvalitet og hastighet
ollama pull qwen3:8b      # Alternativ med sterk resonnering
ollama pull mistral       # Raskere, litt svakere
ollama pull phi3:mini     # For eldre maskiner med lite RAM

# Start Ollama
ollama serve
```

### 2. Klon og installer avhengigheter

```bash
git clone https://github.com/hasqad/obsidian-rag-chatbot.git
cd obsidian-rag-chatbot/rag-chatbot

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Indekser notatene dine

```bash
# Obsidian Vault
python ingest.py --notes-dir ~/Documents/"Obsidian Vault"

# Notion (eksporter til Markdown fra Notion-innstillinger først)
python ingest.py --notes-dir ~/Downloads/Notion-eksport

# Bygg på nytt etter nye notater
python ingest.py --notes-dir ~/Documents/"Obsidian Vault" --reset
```

Første kjøring laster ned embedding-modellen (~80 MB). Indeksering av 100+ filer tar typisk 10–30 sekunder.

### 4. Start chatboten

```bash
python app.py
# Åpne http://localhost:8000
```

---

## Konfigurasjon

```bash
# Bruk en annen Ollama-modell
OLLAMA_MODEL=mistral python app.py

# Ollama kjører på en annen maskin i nettverket
OLLAMA_URL=http://192.168.1.10:11434 python app.py
```

| Parameter | Standard | Beskrivelse |
|---|---|---|
| `--chunk-size` | 400 | Tegn per chunk ved indeksering |
| `--chunk-overlap` | 60 | Overlapp mellom chunks |
| `--reset` | false | Slett og bygg databasen på nytt |
| `TOP_K` (i app.py) | 5 | Antall chunks hentet per spørsmål |

---

## Prosjektstruktur

```
obsidian-rag-chatbot/
├── rag-chatbot/
│   ├── app.py            # FastAPI backend med RAG-logikk og streaming
│   ├── ingest.py         # Indekserer notater til ChromaDB
│   ├── requirements.txt
│   └── static/
│       └── index.html    # Chat-grensesnitt
└── README.md
```

---

## Vanlige problemer

**"Vektorbase ikke funnet"**
→ Kjør `ingest.py` før du starter `app.py`.

**"Kan ikke koble til Ollama"**
→ Kjør `ollama serve` i et separat terminalvindu.

**Modellen svarer ikke på norsk / ignorerer notatene**
→ Systemprompteten sender konteksten på engelsk for å unngå sikkerhetsfiltre i enkelte modeller. Svarene vil likevel komme på norsk hvis du skriver norsk.

**Finner ikke riktig kurs (f.eks. DATS2300)**
→ Kjør `ingest.py --reset` på nytt. Nyere versjoner legger til mappestruktur som prefiks i hvert chunk slik at kurskoder blir søkbare.

**Tregt**
→ Bruk `phi3:mini` eller `mistral` i stedet for llama3. Reduser `TOP_K` i `app.py`.
