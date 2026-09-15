import numpy as np
from skopt import gp_minimize
from skopt.space import Real

from ..db import get_db_cursor


def optimize_threshold():
    with get_db_cursor(commit=True) as cur:
        cur.execute(
            """
            SELECT initial_similarity_score, feedback_type
            FROM clustering_feedback_log
            WHERE created_at >= NOW() - INTERVAL '7 days';
            """
        )
        rows = cur.fetchall()

        if len(rows) < 10:
            return

        scores = np.array([
            float(r["initial_similarity_score"] if isinstance(r, dict) else r[0])
            for r in rows
        ])
        labels = np.array([
            1 if (r["feedback_type"] if isinstance(r, dict) else r[1]) in {"auto_confirmed", "user_confirmed"} else 0
            for r in rows
        ])

        def objective(threshold_vec):
            thresh = threshold_vec[0]
            predictions = (scores >= thresh).astype(int)

            fp = np.sum((predictions == 1) & (labels == 0))
            fn = np.sum((predictions == 0) & (labels == 1))

            cost = (fp * 5.0) + (fn * 1.0)
            return cost

        res = gp_minimize(objective, [Real(0.40, 0.95)], n_calls=25, random_state=42)
        optimal_threshold = round(float(res.x[0]), 3)

        cur.execute(
            """
            INSERT INTO system_config (key, value)
            VALUES ('global_similarity_threshold', %s)
            ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value;
            """,
            (optimal_threshold,)
        )


if __name__ == "__main__":
    optimize_threshold()
