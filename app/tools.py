"""Banking tool registry exposed to the model.

Each entry: a Python callable + a JSON schema (Ollama tool format) describing it.
The agent loop dispatches model-emitted tool calls through `run_tool`, which
applies guardrails first.
"""
from __future__ import annotations

import inspect
from typing import Any, Callable

from app import db, guardrails, rag


def _t_get_account_balance(account_id: str) -> dict[str, Any]:
    acc = db.get_account(account_id)
    if not acc:
        return {"error": f"unknown account_id '{account_id}'"}
    return {
        "id": acc["id"],
        "label": acc["label"],
        "balance": acc["balance"],
        "currency": acc["currency"],
    }


def _t_list_transactions(account_id: str, limit: int = 10) -> dict[str, Any]:
    txns = db.get_transactions(account_id, limit=limit)
    return {"account_id": account_id, "transactions": txns}


def _t_list_accounts() -> dict[str, Any]:
    return {"accounts": db.list_accounts()}


def _t_transfer_internal(from_account: str, to_account: str, amount: float, confirmed: bool = False) -> dict[str, Any]:
    return db.transfer_between_accounts(from_account, to_account, amount)


def _t_transfer_external(iban: str, amount: float) -> dict[str, Any]:
    # Always blocked at guardrail layer; this body is unreachable.
    return {"error": "blocked"}


def _t_lock_card(card_id: str) -> dict[str, Any]:
    return db.set_card_status(card_id, "locked")


def _t_list_cards() -> dict[str, Any]:
    return {"cards": db.list_cards()}


def _t_get_counselor_contact() -> dict[str, Any]:
    c = db.get_counselor()
    return {
        "name": c["name"],
        "email": c["email"],
        "phone": c["phone"],
        "agency": c["agency"]["name"],
    }


def _t_get_agency_hours() -> dict[str, Any]:
    c = db.get_counselor()
    return {"agency": c["agency"]["name"], "address": c["agency"]["address"], "hours": c["agency"]["hours"]}


def _t_list_products() -> dict[str, Any]:
    return {"products": db.list_products()}


def _t_search_contracts(query: str) -> dict[str, Any]:
    hits = rag.retrieve(query, k=4)
    return {"query": query, "results": hits}


def _t_get_client_profile() -> dict[str, Any]:
    c = db.get_client()
    accounts = db.list_accounts()
    products = db.list_products()
    cards = db.list_cards()
    return {
        "identity": {
            "first_name": c["first_name"],
            "last_name": c["last_name"],
            "segment": c.get("segment"),
            "client_since": c.get("client_since"),
            "language": c.get("language", "fr"),
        },
        "accounts_summary": [{"id": a["id"], "label": a["label"]} for a in accounts],
        "products_summary": [{"id": p["id"], "label": p["label"]} for p in products],
        "cards_summary": [{"id": k["id"], "label": k["label"], "status": k["status"]} for k in cards],
    }


REGISTRY: dict[str, Callable[..., dict[str, Any]]] = {
    "get_client_profile": _t_get_client_profile,
    "list_accounts": _t_list_accounts,
    "get_account_balance": _t_get_account_balance,
    "list_transactions": _t_list_transactions,
    "transfer_internal": _t_transfer_internal,
    "transfer_external": _t_transfer_external,
    "list_cards": _t_list_cards,
    "lock_card": _t_lock_card,
    "list_products": _t_list_products,
    "get_counselor_contact": _t_get_counselor_contact,
    "get_agency_hours": _t_get_agency_hours,
    "search_contracts": _t_search_contracts,
}


# Ollama / OpenAI-style tool schemas
SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_client_profile",
            "description": "Return the authenticated client's identity (first name, last name, segment, language) and a summary of their accounts, products and cards. Call this once at the start of a conversation, or whenever you need to address the client by name or check what products/accounts they own.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_accounts",
            "description": "List all of the client's accounts (id, label, balance, IBAN, type).",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_account_balance",
            "description": "Return the current balance of a specific account owned by the client.",
            "parameters": {
                "type": "object",
                "properties": {"account_id": {"type": "string", "description": "Account id, e.g. 'checking', 'livret_a', 'portfolio'."}},
                "required": ["account_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_transactions",
            "description": "List recent transactions on an account, most recent first.",
            "parameters": {
                "type": "object",
                "properties": {
                    "account_id": {"type": "string"},
                    "limit": {"type": "integer", "description": "Max number of transactions to return.", "default": 10},
                },
                "required": ["account_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "transfer_internal",
            "description": "Transfer funds between two accounts owned by the client. Use only after the client clearly asked for it.",
            "parameters": {
                "type": "object",
                "properties": {
                    "from_account": {"type": "string"},
                    "to_account": {"type": "string"},
                    "amount": {"type": "number", "description": "Amount in EUR, must be positive."},
                    "confirmed": {"type": "boolean", "description": "Must be true to authorize amounts above 10000 EUR.", "default": False},
                },
                "required": ["from_account", "to_account", "amount"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "transfer_external",
            "description": "Send a SEPA wire transfer to an external IBAN. ALWAYS BLOCKED by guardrails — do not try to bypass.",
            "parameters": {
                "type": "object",
                "properties": {
                    "iban": {"type": "string"},
                    "amount": {"type": "number"},
                },
                "required": ["iban", "amount"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_cards",
            "description": "List the client's bank cards with current status.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lock_card",
            "description": "Temporarily lock a bank card. Reversible. Safe to call when the client reports loss/theft.",
            "parameters": {
                "type": "object",
                "properties": {"card_id": {"type": "string"}},
                "required": ["card_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_products",
            "description": "List subscribed banking products (insurance, loans, packages).",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_counselor_contact",
            "description": "Return the client's BNP advisor contact details.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_agency_hours",
            "description": "Return the client's home agency address and weekly opening hours.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_contracts",
            "description": "Semantic search over the client's contract corpus (Esprit Libre, Livret A, carte bancaire, assurance auto, prêt études, portefeuille). Use this for any question about contractual terms, fees, ceilings, conditions.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Natural-language query in French or English."}},
                "required": ["query"],
            },
        },
    },
]


_TRUTHY = {"true", "yes", "1", "oui"}
_FALSY = {"false", "no", "0", "non", ""}


def _coerce(value: Any, target: type) -> Any:
    """Best-effort coercion for arg types that small models often get wrong.
    Raises ValueError if the value can't be coerced."""
    if value is None:
        return None
    if target is bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            v = value.strip().lower()
            if v in _TRUTHY:
                return True
            if v in _FALSY:
                return False
            raise ValueError(f"cannot coerce {value!r} to bool")
        raise ValueError(f"cannot coerce {value!r} to bool")
    if target is int:
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        if isinstance(value, str):
            return int(float(value.strip()))
        raise ValueError(f"cannot coerce {value!r} to int")
    if target is float:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        if isinstance(value, str):
            return float(value.strip().replace(",", ".").replace(" ", ""))
        raise ValueError(f"cannot coerce {value!r} to float")
    if target is str:
        return str(value)
    return value


_ANN_MAP = {
    int: int, "int": int,
    float: float, "float": float,
    bool: bool, "bool": bool,
    str: str, "str": str,
}


def _normalize_args(fn: Callable[..., Any], raw: dict[str, Any]) -> dict[str, Any]:
    """Filter unknown kwargs and coerce typed ones based on the function's signature.

    Handles `from __future__ import annotations`: annotations are strings,
    so we look up by string OR by class.
    """
    sig = inspect.signature(fn)
    out: dict[str, Any] = {}
    for pname, param in sig.parameters.items():
        if pname not in raw:
            continue
        v = raw[pname]
        target = _ANN_MAP.get(param.annotation)
        if target is not None:
            try:
                v = _coerce(v, target)
            except ValueError:
                pass
        out[pname] = v
    return out


def run_tool(name: str, args: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    """Apply guardrails, then dispatch. Returns (allowed, result_payload)."""
    fn = REGISTRY.get(name)
    if fn is None:
        return False, {"error": f"unknown tool '{name}'"}
    if not isinstance(args, dict):
        args = {}

    clean_args = _normalize_args(fn, args)

    decision = guardrails.check(name, clean_args)
    if not decision.allowed:
        return False, {"blocked": True, "reason": decision.reason}

    try:
        result = fn(**clean_args)
        return True, result
    except TypeError as e:
        return False, {"error": f"bad arguments: {e}"}
    except ValueError as e:
        return False, {"error": str(e)}
