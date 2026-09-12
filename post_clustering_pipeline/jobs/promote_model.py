"""Promote a validated embedding adapter without mutating the active model blindly."""
from __future__ import annotations

import argparse

from ..db import get_db_connection


def promote(model_version: str, adapter_path: str, precision: float, recall: float, noise_fp: float,
            min_precision: float, min_recall: float, max_noise_fp: float) -> str:
    if precision < min_precision or recall < min_recall or noise_fp > max_noise_fp:
        return "rejected: validation quality gates failed"
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT validation_precision, validation_recall, noise_false_positive_rate FROM model_registry WHERE status = 'active' ORDER BY promoted_at DESC LIMIT 1")
            current = cur.fetchone()
            if current:
                current_precision = float(current["validation_precision"] or 0)
                current_recall = float(current["validation_recall"] or 0)
                current_noise = float(current["noise_false_positive_rate"] or 1)
                if precision < current_precision and recall < current_recall:
                    return "rejected: candidate is worse than active model"
                if noise_fp > current_noise and precision <= current_precision:
                    return "rejected: candidate increases noise errors without precision gain"
            cur.execute("UPDATE model_registry SET status = 'retired' WHERE status = 'active'")
            cur.execute("""INSERT INTO model_registry
                (model_version, adapter_path, validation_precision, validation_recall,
                 noise_false_positive_rate, status, promoted_at)
                VALUES (%s, %s, %s, %s, %s, 'active', NOW())
                ON CONFLICT (model_version) DO UPDATE SET adapter_path = EXCLUDED.adapter_path,
                  validation_precision = EXCLUDED.validation_precision, validation_recall = EXCLUDED.validation_recall,
                  noise_false_positive_rate = EXCLUDED.noise_false_positive_rate, status = 'active', promoted_at = NOW()
            """, (model_version, adapter_path, precision, recall, noise_fp))
        conn.commit()
    finally:
        conn.close()
    return "promoted"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("model_version")
    p.add_argument("adapter_path")
    p.add_argument("--precision", type=float, required=True)
    p.add_argument("--recall", type=float, required=True)
    p.add_argument("--noise-fp", type=float, required=True)
    p.add_argument("--min-precision", type=float, default=0.95)
    p.add_argument("--min-recall", type=float, default=0.80)
    p.add_argument("--max-noise-fp", type=float, default=0.02)
    args = p.parse_args()
    print(promote(args.model_version, args.adapter_path, args.precision, args.recall, args.noise_fp, args.min_precision, args.min_recall, args.max_noise_fp))


if __name__ == "__main__":
    main()
