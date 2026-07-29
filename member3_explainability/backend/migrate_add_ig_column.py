"""
member3_explainability/backend/migrate_add_ig_column.py
=========================================================
One-off migration: adds the `ig_attribution` column to the existing
`shap_logs` table.

Why this is needed: SQLAlchemy's `Base.metadata.create_all()` (called on
app startup in database.py) only creates tables that don't exist yet --
it does NOT add new columns to a table that's already there. Since
models.py now declares `SHAPLog.ig_attribution`, the dev SQLite database
needs this one-time ALTER TABLE to catch up. Existing rows get NULL
(sessions created before this migration simply have no IG data, which the
frontend already handles gracefully).

Safe to run multiple times -- it checks whether the column already exists
first.

Usage
-----
    python -m member3_explainability.backend.migrate_add_ig_column

Author : Adshaya Balarajah (214024V)
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

# Matches DATABASE_URL in .env: sqlite+aiosqlite:///./medoracle.db
_DB_PATH = Path(__file__).resolve().parent.parent.parent / "medoracle.db"


def migrate(db_path: Path = _DB_PATH) -> None:
    if not db_path.exists():
        print(f"No database found at {db_path} -- nothing to migrate "
              "(it will be created with the new column on first run).")
        return

    con = sqlite3.connect(str(db_path))
    try:
        cur = con.cursor()
        cur.execute("PRAGMA table_info(shap_logs)")
        columns = {row[1] for row in cur.fetchall()}

        if "ig_attribution" in columns:
            print("ig_attribution column already exists -- nothing to do.")
            return

        cur.execute("ALTER TABLE shap_logs ADD COLUMN ig_attribution JSON")
        con.commit()
        print(f"Added ig_attribution column to shap_logs in {db_path}")
    finally:
        con.close()


if __name__ == "__main__":
    migrate()
