from __future__ import annotations

import json
import math
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZipFile
from typing import List, Tuple

import numpy as np
from pypdf import PdfReader

from app.db import get_connection
from app.openai_client import embed_texts
from app.settings import settings
from app.time_utils import now_kst_iso


PDF_NAMES = {
    "1. 치료자 교육서.pdf",
    "2. 참가자 학습서 청소년청년용.pdf",
    "3. 치료자 안내서 청소년청년용.pdf",
}
SUPPORTED_DOC_SUFFIXES = {".pdf", ".docx"}


def get_upload_source_dir() -> Path:
    return Path(settings.db_path).resolve().parent / "rag_sources"


def ensure_upload_source_dir() -> Path:
    upload_dir = get_upload_source_dir()
    upload_dir.mkdir(parents=True, exist_ok=True)
    return upload_dir


def list_document_paths(source_dir: str) -> List[Path]:
    base = Path(source_dir)
    if not base.exists():
        return []
    docs = [p for p in base.iterdir() if p.suffix.lower() in SUPPORTED_DOC_SUFFIXES]
    return sorted(
        docs,
        key=lambda item: (
            0 if item.suffix.lower() == ".pdf" and item.name in PDF_NAMES else 1,
            item.name,
        ),
    )


def extract_text_from_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    texts = []
    for page in reader.pages:
        texts.append(page.extract_text() or "")
    return "\n".join(texts)


def extract_text_from_docx(path: Path) -> str:
    try:
        with ZipFile(path, "r") as archive:
            xml = archive.read("word/document.xml")
    except (KeyError, OSError, ValueError):
        return ""

    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return ""

    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    paragraphs = []
    for paragraph in root.findall(".//w:p", ns):
        tokens = [node.text for node in paragraph.findall(".//w:t", ns) if node.text]
        if tokens:
            paragraphs.append("".join(tokens))
    return "\n".join(paragraphs)


def extract_text_from_path(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return extract_text_from_pdf(path)
    if suffix == ".docx":
        return extract_text_from_docx(path)
    return ""


def should_preprocess(text: str) -> bool:
    if not text:
        return False
    newline_ratio = text.count("\n") / max(1, len(text))
    return newline_ratio > 0.01


def preprocess_text(text: str) -> str:
    # 기본 전처리: 하이픈 줄바꿈 제거, 다중 공백 정리
    text = text.replace("-\n", "")
    text = text.replace("\n", " ")
    text = " ".join(text.split())
    return text


def chunk_text(text: str, chunk_size: int = 600, overlap: int = 120) -> List[str]:
    if not text:
        return []
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = min(length, start + chunk_size)
        chunks.append(text[start:end])
        start = end - overlap
        if start < 0:
            start = 0
        if end == length:
            break
    return chunks


def reindex_documents() -> dict:
    upload_dir = ensure_upload_source_dir()
    doc_paths = list_document_paths(str(upload_dir))
    source_mode = "upload"
    source_dir = str(upload_dir)
    if not doc_paths:
        doc_paths = list_document_paths(settings.pdf_dir)
        source_mode = "fallback"
        source_dir = settings.pdf_dir
    if not doc_paths:
        return {
            "count": 0,
            "message": "PDF/DOCX 문서를 찾지 못했습니다.",
            "source_mode": source_mode,
            "source_dir": source_dir,
            "total_documents": 0,
            "indexed_documents": 0,
            "skipped_documents": [],
        }

    with get_connection() as conn:
        conn.execute("DELETE FROM embeddings WHERE source_type = 'doc_chunk'")
        conn.execute("DELETE FROM doc_chunks")
        conn.execute("DELETE FROM documents")
        conn.commit()

    total_chunks = 0
    indexed_documents = 0
    skipped_documents: list[dict[str, str]] = []
    for doc_path in doc_paths:
        try:
            raw = extract_text_from_path(doc_path)
        except Exception as exc:
            skipped_documents.append(
                {
                    "title": doc_path.name,
                    "reason": f"extract_error:{exc.__class__.__name__}",
                }
            )
            continue
        if should_preprocess(raw):
            raw = preprocess_text(raw)
        chunks = chunk_text(raw)
        if not chunks:
            skipped_documents.append(
                {
                    "title": doc_path.name,
                    "reason": "empty_or_unreadable_text",
                }
            )
            continue

        now = now_kst_iso()
        with get_connection() as conn:
            cur = conn.execute(
                "INSERT INTO documents (source, title, created_at) VALUES (?, ?, ?)",
                (str(doc_path), doc_path.name, now),
            )
            doc_id = int(cur.lastrowid)
            conn.commit()

        embeddings = embed_texts(chunks)
        with get_connection() as conn:
            for idx, chunk in enumerate(chunks):
                cur = conn.execute(
                    "INSERT INTO doc_chunks (doc_id, chunk_text, chunk_idx) VALUES (?, ?, ?)",
                    (doc_id, chunk, idx),
                )
                chunk_id = int(cur.lastrowid)
                conn.execute(
                    "INSERT INTO embeddings (source_type, source_id, vector_json, created_at) VALUES (?, ?, ?, ?)",
                    ("doc_chunk", chunk_id, json.dumps(embeddings[idx]), now),
                )
            conn.commit()
        total_chunks += len(chunks)
        indexed_documents += 1

    return {
        "count": total_chunks,
        "message": "인덱싱 완료",
        "source_mode": source_mode,
        "source_dir": source_dir,
        "total_documents": len(doc_paths),
        "indexed_documents": indexed_documents,
        "skipped_documents": skipped_documents,
    }


def index_message_embedding(message_id: int, content: str) -> None:
    vectors = embed_texts([content])
    now = now_kst_iso()
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO embeddings (source_type, source_id, vector_json, created_at) VALUES (?, ?, ?, ?)",
            ("message", message_id, json.dumps(vectors[0]), now),
        )
        conn.commit()


def _load_embeddings(source_type: str) -> List[Tuple[int, np.ndarray]]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT source_id, vector_json FROM embeddings WHERE source_type = ?",
            (source_type,),
        ).fetchall()
        data = []
        for row in rows:
            vec = np.array(json.loads(row["vector_json"]), dtype=np.float32)
            data.append((row["source_id"], vec))
        return data


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def search_rag(query: str, top_k: int = 4, user_id: int | None = None) -> dict:
    query_vec = np.array(embed_texts([query])[0], dtype=np.float32)

    doc_embeddings = _load_embeddings("doc_chunk")
    msg_embeddings = _load_embeddings("message")

    doc_scores = [(chunk_id, _cosine_sim(query_vec, vec)) for chunk_id, vec in doc_embeddings]
    doc_scores.sort(key=lambda x: x[1], reverse=True)

    msg_scores = [(msg_id, _cosine_sim(query_vec, vec)) for msg_id, vec in msg_embeddings]
    msg_scores.sort(key=lambda x: x[1], reverse=True)

    doc_ids = [cid for cid, _ in doc_scores[:top_k]]
    msg_ids = [mid for mid, _ in msg_scores[:top_k]]

    docs = []
    if doc_ids:
        with get_connection() as conn:
            rows = conn.execute(
                f"SELECT doc_chunks.id, doc_chunks.chunk_text, documents.title FROM doc_chunks JOIN documents ON documents.id = doc_chunks.doc_id WHERE doc_chunks.id IN ({','.join('?' for _ in doc_ids)})",
                doc_ids,
            ).fetchall()
            docs = [dict(r) for r in rows]

    msgs = []
    if msg_ids:
        with get_connection() as conn:
            rows = conn.execute(
                f"SELECT messages.id, messages.content, sessions.user_id FROM messages JOIN sessions ON sessions.id = messages.session_id WHERE messages.id IN ({','.join('?' for _ in msg_ids)})",
                msg_ids,
            ).fetchall()
            msgs = [dict(r) for r in rows if user_id is None or r["user_id"] == user_id]

    return {"doc_chunks": docs, "messages": msgs}
