"""
ingest.py — Les Obsidian/Notion-notater og bygg en lokal vektorbase.

Bruk:
    python ingest.py --notes-dir /sti/til/notater
    python ingest.py --notes-dir ~/Obsidian/Vault --chunk-size 400
"""

import argparse
import os
import re
import sys
import time
from pathlib import Path
from typing import Generator

import frontmatter
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# --- Konfigurasjon ---
DB_DIR = "./chroma_db"
COLLECTION_NAME = "notater"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Liten, rask, god kvalitet (~80 MB)
DEFAULT_CHUNK_SIZE = 400   # tegn per chunk
DEFAULT_CHUNK_OVERLAP = 60  # overlapp mellom chunks


def finn_markdown_filer(rot_sti: Path) -> list[Path]:
    """Finn alle .md og .txt filer rekursivt."""
    filer = []
    for ext in ("*.md", "*.txt", "*.markdown"):
        filer.extend(rot_sti.rglob(ext))
    # Filtrer bort Obsidian-systemfiler
    filer = [f for f in filer if not any(
        del_sti in str(f) for del_sti in [".obsidian", ".trash", ".git"]
    )]
    return sorted(filer)


def les_notat(fil: Path) -> tuple[str, dict]:
    """Les en markdown-fil og returner (innhold, metadata)."""
    try:
        post = frontmatter.load(str(fil))
        innhold = post.content
        metadata = dict(post.metadata)
    except Exception:
        innhold = fil.read_text(encoding="utf-8", errors="ignore")
        metadata = {}

    # Rens Obsidian-spesifikk syntaks
    innhold = re.sub(r"!\[\[.*?\]\]", "", innhold)           # Embedded images
    innhold = re.sub(r"\[\[([^\]|]+)\|?[^\]]*\]\]", r"\1", innhold)  # Wiki-lenker
    innhold = re.sub(r"```[\s\S]*?```", "", innhold)         # Kodeblokker (valgfritt)
    innhold = re.sub(r"#+ ", "", innhold)                    # Header-symboler
    innhold = re.sub(r"\n{3,}", "\n\n", innhold)             # Ekstra linjeskift
    innhold = innhold.strip()

    metadata["kilde_fil"] = str(fil)
    metadata["tittel"] = metadata.get("title", fil.stem)
    metadata["tags"] = ", ".join(str(t) for t in metadata.get("tags", []))

    return innhold, metadata


def lag_chunks(
    tekst: str,
    metadata: dict,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> Generator[tuple[str, dict], None, None]:
    """
    Del tekst i overlappende chunks.
    Prøver å kutte ved avsnitt for å bevare kontekst.
    """
    if not tekst:
        return

    avsnitt = tekst.split("\n\n")
    gjeldende = ""
    chunk_nr = 0

    for avsnitt_tekst in avsnitt:
        avsnitt_tekst = avsnitt_tekst.strip()
        if not avsnitt_tekst:
            continue

        if len(gjeldende) + len(avsnitt_tekst) > chunk_size and gjeldende:
            if len(gjeldende.strip()) > 30:
                chunk_meta = {**metadata, "chunk_nr": chunk_nr}
                yield gjeldende.strip(), chunk_meta
                chunk_nr += 1
            # Behold overlapp
            ord_liste = gjeldende.split()
            overlap_ord = ord_liste[-max(1, overlap // 6):]
            gjeldende = " ".join(overlap_ord) + "\n\n" + avsnitt_tekst
        else:
            gjeldende = (gjeldende + "\n\n" + avsnitt_tekst).lstrip()

    if gjeldende.strip() and len(gjeldende.strip()) > 30:
        yield gjeldende.strip(), {**metadata, "chunk_nr": chunk_nr}


def main():
    parser = argparse.ArgumentParser(
        description="Indekser Obsidian/Notion-notater til lokal vektorbase"
    )
    parser.add_argument(
        "--notes-dir", required=True,
        help="Sti til notat-mappen (f.eks. ~/Obsidian/Vault)"
    )
    parser.add_argument(
        "--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE,
        help=f"Tegn per chunk (standard: {DEFAULT_CHUNK_SIZE})"
    )
    parser.add_argument(
        "--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP,
        help=f"Overlapp mellom chunks (standard: {DEFAULT_CHUNK_OVERLAP})"
    )
    parser.add_argument(
        "--reset", action="store_true",
        help="Slett eksisterende database og bygg på nytt"
    )
    args = parser.parse_args()

    notes_dir = Path(args.notes_dir).expanduser().resolve()
    if not notes_dir.exists():
        print(f"[FEIL] Mappen finnes ikke: {notes_dir}")
        sys.exit(1)

    print(f"\n📂  Notatmappe : {notes_dir}")
    print(f"💾  Vektorbase  : {os.path.abspath(DB_DIR)}")
    print(f"🧠  Modell      : {EMBEDDING_MODEL}\n")

    # Last embedding-modell
    print("Laster embedding-modell ...")
    modell = SentenceTransformer(EMBEDDING_MODEL)

    # Sett opp ChromaDB
    klient = chromadb.PersistentClient(
        path=DB_DIR,
        settings=Settings(anonymized_telemetry=False),
    )

    if args.reset:
        try:
            klient.delete_collection(COLLECTION_NAME)
            print("Eksisterende samling slettet.\n")
        except Exception:
            pass

    samling = klient.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # Finn og prosesser filer
    filer = finn_markdown_filer(notes_dir)
    print(f"Fant {len(filer)} filer. Starter indeksering...\n")

    totalt_chunks = 0
    start_tid = time.time()

    for i, fil in enumerate(filer):
        relativ = fil.relative_to(notes_dir)
        print(f"  [{i+1:>4}/{len(filer)}] {relativ}", end="", flush=True)

        try:
            innhold, metadata = les_notat(fil)
            if not innhold:
                print("  (tom)")
                continue

            chunks = list(lag_chunks(innhold, metadata, args.chunk_size, args.chunk_overlap))
            if not chunks:
                print("  (ingen chunks)")
                continue

            tekster = [c[0] for c in chunks]
            meta_liste = [c[1] for c in chunks]

            # Lag unike IDer basert på filsti og chunk-nummer
            id_liste = [
                f"{fil.stem}_{m['chunk_nr']}_{hash(fil)}"
                for m in meta_liste
            ]

            # Konverter metadata-verdier til strenger (ChromaDB-krav)
            for m in meta_liste:
                for k, v in m.items():
                    if not isinstance(v, (str, int, float, bool)):
                        m[k] = str(v)

            # Generer embeddings
            embeddings = modell.encode(tekster, show_progress_bar=False).tolist()

            # Lagre i ChromaDB (upsert for å unngå duplikater)
            samling.upsert(
                ids=id_liste,
                documents=tekster,
                embeddings=embeddings,
                metadatas=meta_liste,
            )

            totalt_chunks += len(chunks)
            print(f"  → {len(chunks)} chunks")

        except Exception as e:
            print(f"  [FEIL] {e}")

    elapsed = time.time() - start_tid
    print(f"\n✅  Ferdig! {totalt_chunks} chunks fra {len(filer)} filer ({elapsed:.1f}s)")
    print(f"📊  Database: {DB_DIR}")
    print(f"\nStart chatboten med:\n   python app.py\n")


if __name__ == "__main__":
    main()
