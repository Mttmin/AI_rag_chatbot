"""SQLite mock-bank data layer.

Single demo user. All accessors return plain dicts (JSON-friendly for tool returns).
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "bank.sqlite"
FIXTURES = ROOT / "data" / "fixtures"

SCHEMA = """
CREATE TABLE accounts (
    id TEXT PRIMARY KEY,
    label TEXT NOT NULL,
    iban TEXT,
    balance REAL NOT NULL,
    currency TEXT NOT NULL,
    type TEXT NOT NULL
);
CREATE TABLE cards (
    id TEXT PRIMARY KEY,
    label TEXT NOT NULL,
    last_four TEXT NOT NULL,
    linked_account TEXT NOT NULL,
    status TEXT NOT NULL,
    monthly_limit REAL NOT NULL,
    FOREIGN KEY (linked_account) REFERENCES accounts(id)
);
CREATE TABLE transactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    account_id TEXT NOT NULL,
    date TEXT NOT NULL,
    amount REAL NOT NULL,
    label TEXT NOT NULL,
    FOREIGN KEY (account_id) REFERENCES accounts(id)
);
CREATE TABLE products (
    id TEXT PRIMARY KEY,
    label TEXT NOT NULL,
    type TEXT NOT NULL,
    payload TEXT NOT NULL
);
CREATE TABLE counselor (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    payload TEXT NOT NULL
);
CREATE TABLE client (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    payload TEXT NOT NULL
);
"""


def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def reset_and_seed() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    if DB_PATH.exists():
        DB_PATH.unlink()
    conn = connect()
    conn.executescript(SCHEMA)

    accounts = json.loads((FIXTURES / "accounts.json").read_text())
    cards = json.loads((FIXTURES / "cards.json").read_text())
    txns = json.loads((FIXTURES / "transactions.json").read_text())
    products = json.loads((FIXTURES / "products.json").read_text())
    counselor = json.loads((FIXTURES / "counselor.json").read_text())
    client = json.loads((FIXTURES / "client.json").read_text())

    conn.executemany(
        "INSERT INTO accounts (id,label,iban,balance,currency,type) VALUES (:id,:label,:iban,:balance,:currency,:type)",
        accounts,
    )
    conn.executemany(
        "INSERT INTO cards (id,label,last_four,linked_account,status,monthly_limit) VALUES (:id,:label,:last_four,:linked_account,:status,:monthly_limit)",
        cards,
    )
    conn.executemany(
        "INSERT INTO transactions (account_id,date,amount,label) VALUES (:account_id,:date,:amount,:label)",
        txns,
    )
    for p in products:
        conn.execute(
            "INSERT INTO products (id,label,type,payload) VALUES (?,?,?,?)",
            (p["id"], p["label"], p["type"], json.dumps(p)),
        )
    conn.execute("INSERT INTO counselor (id,payload) VALUES (1, ?)", (json.dumps(counselor),))
    conn.execute("INSERT INTO client (id,payload) VALUES (1, ?)", (json.dumps(client),))
    conn.commit()
    conn.close()


def list_accounts() -> list[dict[str, Any]]:
    conn = connect()
    rows = conn.execute("SELECT * FROM accounts").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_account(account_id: str) -> dict[str, Any] | None:
    conn = connect()
    row = conn.execute("SELECT * FROM accounts WHERE id = ?", (account_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def get_transactions(account_id: str, limit: int = 10) -> list[dict[str, Any]]:
    conn = connect()
    rows = conn.execute(
        "SELECT date, amount, label FROM transactions WHERE account_id = ? ORDER BY date DESC, id DESC LIMIT ?",
        (account_id, limit),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def transfer_between_accounts(from_id: str, to_id: str, amount: float) -> dict[str, Any]:
    if amount <= 0:
        raise ValueError("amount must be positive")
    conn = connect()
    try:
        src = conn.execute("SELECT balance FROM accounts WHERE id = ?", (from_id,)).fetchone()
        dst = conn.execute("SELECT balance FROM accounts WHERE id = ?", (to_id,)).fetchone()
        if src is None or dst is None:
            raise ValueError("unknown account")
        if src["balance"] < amount:
            raise ValueError("insufficient funds")
        conn.execute("UPDATE accounts SET balance = balance - ? WHERE id = ?", (amount, from_id))
        conn.execute("UPDATE accounts SET balance = balance + ? WHERE id = ?", (amount, to_id))
        today = _today()
        conn.execute(
            "INSERT INTO transactions (account_id,date,amount,label) VALUES (?,?,?,?)",
            (from_id, today, -amount, f"Virement vers {to_id}"),
        )
        conn.execute(
            "INSERT INTO transactions (account_id,date,amount,label) VALUES (?,?,?,?)",
            (to_id, today, amount, f"Virement depuis {from_id}"),
        )
        conn.commit()
        new_src = conn.execute("SELECT balance FROM accounts WHERE id = ?", (from_id,)).fetchone()
        new_dst = conn.execute("SELECT balance FROM accounts WHERE id = ?", (to_id,)).fetchone()
        return {
            "from": {"id": from_id, "new_balance": new_src["balance"]},
            "to": {"id": to_id, "new_balance": new_dst["balance"]},
            "amount": amount,
        }
    finally:
        conn.close()


def list_cards() -> list[dict[str, Any]]:
    conn = connect()
    rows = conn.execute("SELECT * FROM cards").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def set_card_status(card_id: str, status: str) -> dict[str, Any]:
    conn = connect()
    try:
        cur = conn.execute("UPDATE cards SET status = ? WHERE id = ?", (status, card_id))
        if cur.rowcount == 0:
            raise ValueError("unknown card")
        conn.commit()
        row = conn.execute("SELECT * FROM cards WHERE id = ?", (card_id,)).fetchone()
        return dict(row)
    finally:
        conn.close()


def list_products() -> list[dict[str, Any]]:
    conn = connect()
    rows = conn.execute("SELECT payload FROM products").fetchall()
    conn.close()
    return [json.loads(r["payload"]) for r in rows]


def get_counselor() -> dict[str, Any]:
    conn = connect()
    row = conn.execute("SELECT payload FROM counselor WHERE id = 1").fetchone()
    conn.close()
    return json.loads(row["payload"])


def get_client() -> dict[str, Any]:
    conn = connect()
    row = conn.execute("SELECT payload FROM client WHERE id = 1").fetchone()
    conn.close()
    return json.loads(row["payload"])


def _today() -> str:
    from datetime import date

    return date.today().isoformat()
