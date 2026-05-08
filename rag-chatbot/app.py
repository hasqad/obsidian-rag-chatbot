"""
app.py — Lokal RAG-chatbot backend med FastAPI + Ollama + ChromaDB.

Start:
    python app.py
    # eller:
    uvicorn app:app --reload --port 8000
"""

import json
import os
from pathlib import Path
from typing import AsyncGenerator

import chromadb
import httpx
from chromadb.config import Settings
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

# --- Konfigurasjon ---
DB_DIR = "./chroma_db"
COLLECTION_NAME = "notater"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
OLLAMA_BASE_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
TOP_K = 5  # Antall chunks å hente per spørring

# --- FastAPI-app ---
app = FastAPI(title="Lokal RAG-chatbot", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Global tilstand ---
embedding_modell: SentenceTransformer | None = None
samling = None


@app.on_event("startup")
async def oppstart():
    """Last modeller og koble til ChromaDB ved oppstart."""
    global embedding_modell, samling

    print("🚀 Starter RAG-chatbot ...")

    if not Path(DB_DIR).exists():
        raise RuntimeError(
            f"Vektorbase ikke funnet i '{DB_DIR}'. "
            "Kjør 'python ingest.py --notes-dir /sti/til/notater' først."
        )

    print(f"🧠 Laster embedding-modell: {EMBEDDING_MODEL}")
    embedding_modell = SentenceTransformer(EMBEDDING_MODEL)

    print(f"💾 Kobler til ChromaDB: {DB_DIR}")
    klient = chromadb.PersistentClient(
        path=DB_DIR,
        settings=Settings(anonymized_telemetry=False),
    )
    samling = klient.get_collection(COLLECTION_NAME)
    antall = samling.count()

    print(f"✅ Klar! {antall} chunks indeksert. Åpne http://localhost:8000\n")


# --- Pydantic-modeller ---
class ChatMelding(BaseModel):
    role: str   # "user" eller "assistant"
    content: str


class ChatForespørsel(BaseModel):
    meldinger: list[ChatMelding]
    modell: str = OLLAMA_MODEL
    top_k: int = TOP_K


class KildeInfo(BaseModel):
    tittel: str
    fil: str
    utdrag: str


# --- Hjelpefunksjoner ---
def hent_relevante_chunks(spørring: str, top_k: int) -> list[dict]:
    """Embed spørringen og finn de mest relevante notat-chunkene."""
    embedding = embedding_modell.encode([spørring]).tolist()
    resultater = samling.query(
        query_embeddings=embedding,
        n_results=min(top_k, samling.count()),
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for doc, meta, dist in zip(
        resultater["documents"][0],
        resultater["metadatas"][0],
        resultater["distances"][0],
    ):
        relevans = 1 - dist  # Cosine-avstand → relevans-score
        if relevans > 0.25:  # Filtrer bort svakt relevante chunks
            chunks.append({
                "tekst": doc,
                "tittel": meta.get("tittel", "Ukjent"),
                "fil": meta.get("kilde_fil", ""),
                "tags": meta.get("tags", ""),
                "relevans": round(relevans, 3),
            })

    return chunks


def bygg_system_prompt(chunks: list[dict]) -> str:
    """Bygg systemprompt med hentet kontekst fra notater."""
    if not chunks:
        kontekst = "Ingen relevante notater ble funnet for dette spørsmålet."
    else:
        kontekst_deler = []
        for i, chunk in enumerate(chunks, 1):
            kontekst_deler.append(
                f"[Kilde {i}: {chunk['tittel']}]\n{chunk['tekst']}"
            )
        kontekst = "\n\n---\n\n".join(kontekst_deler)

    return f"""Du er en personlig AI-assistent som hjelper brukeren med å finne informasjon fra sine egne notater.

KONTEKST FRA NOTATER:
{kontekst}

INSTRUKSJONER:
- Svar alltid basert på konteksten over hvis den er relevant
- Referer til kilder med [Kilde N]-notasjonen når du bruker informasjon fra dem
- Hvis konteksten ikke inneholder svaret, si det tydelig
- Svar på samme språk som brukeren skriver på
- Vær konsis og presis
"""


async def stream_ollama(
    meldinger: list[dict],
    modell: str,
) -> AsyncGenerator[str, None]:
    """Stream svar fra Ollama API."""
    async with httpx.AsyncClient(timeout=120.0) as klient:
        try:
            async with klient.stream(
                "POST",
                f"{OLLAMA_BASE_URL}/api/chat",
                json={
                    "model": modell,
                    "messages": meldinger,
                    "stream": True,
                    "options": {
                        "temperature": 0.3,
                        "num_ctx": 4096,
                    },
                },
            ) as resp:
                if resp.status_code != 200:
                    feil = await resp.aread()
                    yield f"data: {json.dumps({'error': f'Ollama feil: {resp.status_code}'})}\n\n"
                    return

                async for linje in resp.aiter_lines():
                    if not linje:
                        continue
                    try:
                        data = json.loads(linje)
                        if data.get("done"):
                            yield f"data: {json.dumps({'done': True})}\n\n"
                            break
                        innhold = data.get("message", {}).get("content", "")
                        if innhold:
                            yield f"data: {json.dumps({'token': innhold})}\n\n"
                    except json.JSONDecodeError:
                        continue

        except httpx.ConnectError:
            yield f"data: {json.dumps({'error': 'Kan ikke koble til Ollama. Er den kjørende? (ollama serve)'})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"


# --- API-endepunkter ---
@app.get("/api/helse")
async def helse():
    """Sjekk status på alle systemkomponenter."""
    status = {
        "embedding_modell": embedding_modell is not None,
        "vektorbase": samling is not None,
        "antall_chunks": samling.count() if samling else 0,
        "ollama_modell": OLLAMA_MODEL,
    }

    # Sjekk Ollama
    try:
        async with httpx.AsyncClient(timeout=3.0) as klient:
            resp = await klient.get(f"{OLLAMA_BASE_URL}/api/tags")
            modeller = [m["name"] for m in resp.json().get("models", [])]
            status["ollama_tilkoblet"] = True
            status["tilgjengelige_modeller"] = modeller
    except Exception:
        status["ollama_tilkoblet"] = False
        status["tilgjengelige_modeller"] = []

    return status


@app.post("/api/chat")
async def chat(forespørsel: ChatForespørsel):
    """
    RAG-chat med streaming.
    Returnerer Server-Sent Events med tokens og kildeinfo.
    """
    if not embedding_modell or not samling:
        raise HTTPException(503, "Serveren er ikke klar ennå")

    # Hent siste brukermelding for RAG-søk
    siste_bruker = next(
        (m.content for m in reversed(forespørsel.meldinger) if m.role == "user"),
        ""
    )

    # Finn relevante chunks
    relevante_chunks = hent_relevante_chunks(siste_bruker, forespørsel.top_k)

    # Bygg system-prompt
    system_prompt = bygg_system_prompt(relevante_chunks)

    # Sett opp meldingshistorikk for Ollama
    ollama_meldinger = [{"role": "system", "content": system_prompt}]
    for m in forespørsel.meldinger:
        ollama_meldinger.append({"role": m.role, "content": m.content})

    # Serialiser kildeinfo for frontend
    kilder = [
        {
            "tittel": c["tittel"],
            "fil": Path(c["fil"]).name if c["fil"] else "",
            "utdrag": c["tekst"][:200] + "..." if len(c["tekst"]) > 200 else c["tekst"],
            "relevans": c["relevans"],
        }
        for c in relevante_chunks
    ]

    async def generator():
        # Send kildeinfo først
        yield f"data: {json.dumps({'kilder': kilder})}\n\n"

        # Stream LLM-svar
        async for chunk in stream_ollama(ollama_meldinger, forespørsel.modell):
            yield chunk

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/modeller")
async def hent_modeller():
    """Hent tilgjengelige Ollama-modeller."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as klient:
            resp = await klient.get(f"{OLLAMA_BASE_URL}/api/tags")
            return {"modeller": [m["name"] for m in resp.json().get("models", [])]}
    except Exception:
        return {"modeller": [], "feil": "Kunne ikke nå Ollama"}


# --- Serve frontend ---
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
