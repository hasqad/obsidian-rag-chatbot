# Lokal RAG-chatbot over egne notater

En fullstendig privat AI-assistent som svarer på spørsmål fra dine egne
Obsidian/Notion-notater. Ingen sky, ingen deling, alt kjører lokalt.

## Arkitektur

```
Notater (.md) → ingest.py → ChromaDB (lokal)
                                  ↓
Bruker → FastAPI → hent chunks → Ollama (lokal LLM) → svar med kilder
```

**Komponenter:**
- `sentence-transformers` → lokale embeddings (all-MiniLM-L6-v2, ~80 MB)
- `ChromaDB` → lokal vektordatabase (persistent på disk)
- `Ollama` → lokal LLM (llama3, mistral, phi3, osv.)
- `FastAPI` → backend med streaming SSE
- `static/index.html` → chat-grensesnitt med kildevisning

---

## Oppsett

### 1. Installer Ollama

```bash
# macOS / Linux
curl -fsSL https://ollama.com/install.sh | sh

# Last ned en modell (velg én)
ollama pull llama3          # Anbefalt, god balanse
ollama pull mistral         # Raskere, litt svakere
ollama pull phi3:mini       # Svært liten, for eldre maskiner

# Start Ollama (kjør i bakgrunnen)
ollama serve
```

### 2. Installer Python-avhengigheter

```bash
cd rag-chatbot
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 3. Indekser notatene dine

```bash
# Obsidian
python ingest.py --notes-dir ~/Obsidian/MinVault

# Notion (eksporter til Markdown først fra Notion-innstillinger)
python ingest.py --notes-dir ~/Downloads/Notion-eksport

# Med egne innstillinger
python ingest.py --notes-dir ~/notater --chunk-size 500 --chunk-overlap 80

# Bygg på nytt (etter du har lagt til nye notater)
python ingest.py --notes-dir ~/Obsidian/MinVault --reset
```

Første gang laster dette ned embedding-modellen (~80 MB). Deretter tar
indeksering vanligvis 1-5 minutter avhengig av antall notater.

### 4. Start chatboten

```bash
python app.py
# Åpne http://localhost:8000 i nettleseren
```

---

## Konfigurasjon

Sett miljøvariabler for å endre standardinnstillinger:

```bash
# Bruk annen Ollama-modell
OLLAMA_MODEL=mistral python app.py

# Ollama kjører på annen maskin
OLLAMA_URL=http://192.168.1.10:11434 python app.py
```

---

## Oppdatering av notater

Når du legger til nye notater, kjør ingest på nytt:

```bash
# Legg til nye notater uten å slette eksisterende
python ingest.py --notes-dir ~/Obsidian/MinVault

# Full rebuild
python ingest.py --notes-dir ~/Obsidian/MinVault --reset
```

---

## Vanlige problemer

**"Vektorbase ikke funnet"**
→ Kjør `ingest.py` først.

**"Kan ikke koble til Ollama"**
→ Kjør `ollama serve` i et separat terminalvindu.

**Dårlig svarkvallitet**
→ Prøv større `--chunk-size` (600-800) eller bruk en større modell.
→ Sjekk at notatene er i ren tekst/markdown-format.

**Tregt**
→ `phi3:mini` er raskere enn llama3 men noe svakere.
→ Reduser `TOP_K` i `app.py` (standard: 5).
