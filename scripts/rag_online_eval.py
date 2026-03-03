#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag import search_rag
from app.time_utils import now_kst_iso


DEFAULT_OUT = "/tmp/rag_online_eval_2026-03-03.json"
GOLDEN_QUERIES = [
    {
        "query": "호흡이 너무 가빠질 때 바로 할 수 있는 방법",
        "expected_tags": ["호흡", "복식호흡", "호흡 조절", "숨"],
    },
    {
        "query": "마음챙김 기본 기술 3가지를 알려줘",
        "expected_tags": ["마음챙김", "관찰", "묘사", "몰입", "비판단"],
    },
    {
        "query": "복식호흡을 언제 쓰는지",
        "expected_tags": ["복식호흡", "호흡", "긴장", "불안"],
    },
    {
        "query": "스트레스 올라올 때 몸부터 안정시키는 법",
        "expected_tags": ["스트레스", "호흡", "이완", "안정"],
    },
    {
        "query": "자살 생각이 올라올 때 즉시 할 안전 행동",
        "expected_tags": ["안전", "1393", "112", "위기", "자살"],
    },
    {
        "query": "마음챙김으로 생각을 다루는 방법",
        "expected_tags": ["마음챙김", "생각", "수용", "관찰"],
    },
    {
        "query": "DBT 기반으로 충동을 낮추는 기술",
        "expected_tags": ["DBT", "충동", "고통 감내", "마음챙김"],
    },
    {
        "query": "불면이 심할 때 자기 돌보기 항목",
        "expected_tags": ["수면", "자기 돌보기", "스트레스", "호흡"],
    },
    {
        "query": "긴장할 때 4-6 호흡 같은 훈련이 있나",
        "expected_tags": ["호흡", "복식호흡", "긴장", "훈련"],
    },
    {
        "query": "현재 감각에 집중하는 연습이 뭐야",
        "expected_tags": ["마음챙김", "현재", "감각", "주의"],
    },
]


def _normalize_text(value: str) -> str:
    return (value or "").lower()


def _evaluate_query(query: str, expected_tags: list[str], top_k: int) -> dict:
    try:
        rag_result = search_rag(query, top_k=top_k)
    except Exception as exc:  # pragma: no cover - runtime/network path
        return {
            "query": query,
            "expected_tags": expected_tags,
            "status": "error",
            "error_type": exc.__class__.__name__,
            "error": str(exc),
            "matched": False,
            "matched_tags": [],
            "doc_chunks": [],
        }

    docs = rag_result.get("doc_chunks", [])
    matched_tags: set[str] = set()
    serialized_docs = []
    for item in docs:
        title = str(item.get("title", ""))
        chunk_text = str(item.get("chunk_text", ""))
        corpus = f"{title}\n{chunk_text}"
        normalized = _normalize_text(corpus)
        for tag in expected_tags:
            if _normalize_text(tag) in normalized:
                matched_tags.add(tag)
        serialized_docs.append(
            {
                "title": title,
                "chunk_preview": chunk_text.replace("\n", " ")[:180],
            }
        )

    matched = len(matched_tags) > 0
    return {
        "query": query,
        "expected_tags": expected_tags,
        "status": "ok",
        "matched": matched,
        "matched_tags": sorted(matched_tags),
        "doc_chunks": serialized_docs,
    }


def run_eval(top_k: int, out_path: Path) -> dict:
    per_query = [_evaluate_query(item["query"], item["expected_tags"], top_k) for item in GOLDEN_QUERIES]
    ok_queries = [item for item in per_query if item["status"] == "ok"]
    error_queries = [item for item in per_query if item["status"] != "ok"]
    matched_queries = [item for item in ok_queries if item["matched"]]
    unrelated_queries = [item for item in ok_queries if not item["matched"]]

    summary = {
        "timestamp": now_kst_iso(),
        "top_k": top_k,
        "total_queries": len(per_query),
        "ok_queries": len(ok_queries),
        "error_queries": len(error_queries),
        "matched_queries": len(matched_queries),
        "unrelated_queries": len(unrelated_queries),
        "pass_threshold": len(matched_queries) >= 8 and len(unrelated_queries) == 0 and len(error_queries) == 0,
    }
    payload = {
        "summary": summary,
        "results": per_query,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="RAG online relevance evaluation")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--out", type=str, default=DEFAULT_OUT)
    args = parser.parse_args()

    output_path = Path(args.out).expanduser().resolve()
    payload = run_eval(top_k=max(1, int(args.top_k)), out_path=output_path)
    summary = payload["summary"]
    print(
        (
            "RAG online eval done | total={total_queries} ok={ok_queries} "
            "error={error_queries} matched={matched_queries} unrelated={unrelated_queries} "
            "pass={pass_threshold} out={out}"
        ).format(**summary, out=str(output_path))
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
