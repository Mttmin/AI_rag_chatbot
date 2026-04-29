"""Tool-level guardrails. Data-coded rules, not prompt-coded.

The agent loop must call check() before executing any tool. If allowed=False,
the agent must surface `reason` to the user and not retry the same call.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app import db

INTERNAL_TRANSFER_CONFIRM_THRESHOLD = 10_000.0


@dataclass
class Decision:
    allowed: bool
    reason: str = ""


def _owned_account_ids() -> set[str]:
    return {a["id"] for a in db.list_accounts()}


def check(tool_name: str, args: dict[str, Any]) -> Decision:
    if tool_name == "transfer_external":
        return Decision(
            False,
            "Les virements externes (vers un IBAN tiers) sont bloqués par la politique de sécurité. "
            "Pour ce type d'opération, je dois vous orienter vers votre conseiller ou l'application "
            "BNP Paribas avec authentification forte.",
        )

    if tool_name == "transfer_internal":
        owned = _owned_account_ids()
        from_id = args.get("from_account")
        to_id = args.get("to_account")
        try:
            amount = float(args.get("amount", 0))
        except (TypeError, ValueError):
            return Decision(False, "Le montant fourni n'est pas un nombre valide.")
        if from_id not in owned or to_id not in owned:
            return Decision(False, "Compte source ou destination inconnu pour ce client.")
        if from_id == to_id:
            return Decision(False, "Le compte source et le compte destinataire sont identiques.")
        if amount <= 0:
            return Decision(False, "Le montant doit être strictement positif.")
        if amount > INTERNAL_TRANSFER_CONFIRM_THRESHOLD and not args.get("confirmed"):
            return Decision(
                False,
                f"Montant supérieur à {INTERNAL_TRANSFER_CONFIRM_THRESHOLD:.0f} €. "
                "Confirmation explicite du client requise (paramètre `confirmed=true`).",
            )
        src = db.get_account(from_id)
        if src and src["balance"] - amount < 0:
            return Decision(
                False,
                f"Solde insuffisant sur le compte source ({src['balance']:.2f} €). "
                "Le découvert n'est pas autorisé pour ce virement.",
            )
        return Decision(True)

    if tool_name in {"get_account_balance", "list_transactions"}:
        owned = _owned_account_ids()
        acc = args.get("account_id")
        if acc and acc not in owned:
            return Decision(False, "Ce compte n'appartient pas au client.")
        return Decision(True)

    return Decision(True)
