"""Agent orchestration loop. Talks to local Ollama, dispatches tool calls.

Yields a stream of events for the SSE layer:
  {"type": "token", "text": "..."}            -- model text token
  {"type": "tool_start", "name": "...", "args": {...}}
  {"type": "tool_result", "name": "...", "allowed": bool, "result": {...}}
  {"type": "done"}
  {"type": "error", "message": "..."}
"""
from __future__ import annotations

import json
import os
from typing import Any, Iterator

import ollama

from app import tools

MODEL = os.environ.get("MISTRAL_MODEL", "ministral-3:8b")
MAX_TOOL_ITERATIONS = 5

SYSTEM_PROMPT = """Tu es l'assistant virtuel BNP Paribas pour la clientèle particulière. Tu accompagnes le client dans la gestion quotidienne de ses comptes, ses produits et ses contrats. Tu réponds en français par défaut, et en anglais si la question est posée en anglais.

Identité du client
- Au début d'une conversation, ou dès que tu as besoin du nom ou de la liste des comptes/produits/cartes du client, appelle l'outil `get_client_profile`. Ne suppose jamais l'identité, les comptes détenus, ni les produits souscrits — récupère-les.
- Utilise les identifiants techniques renvoyés par `get_client_profile` (par ex. l'`id` de chaque compte, l'`id` de chaque carte) lorsque tu appelles les autres outils. N'invente aucun identifiant.

Comportement
- Utilise systématiquement les outils pour obtenir une information factuelle (solde, transaction, contrat, horaire, identité). Ne devine jamais un chiffre, un nom ou une référence.
- Pour toute question contractuelle (frais, plafonds, conditions, garanties), appelle `search_contracts` et cite la source dans la réponse (par ex. « selon `esprit_libre.md` »).
- Tu peux exécuter directement les actions à risque faible et réversibles : virement entre comptes du même client, blocage temporaire d'une carte.
- Les virements externes (vers un IBAN tiers) sont bloqués par sécurité. Si le client demande, redirige vers son conseiller BNP ou vers l'application BNP Paribas avec authentification forte. N'essaie jamais de contourner ce blocage.
- Pour un virement interne supérieur à 10 000 €, demande une confirmation explicite au client avant de relancer l'outil avec `confirmed=true`.
- Si un outil renvoie `blocked: true`, explique brièvement la raison du blocage et propose l'alternative (conseiller, agence, application avec auth forte).

Style
- Sois concis. Donne le chiffre, l'action effectuée, la source. Pas de paragraphes inutiles, pas de formules creuses.
- Adresse-toi au client par son prénom une fois que tu l'as récupéré via `get_client_profile`.
"""


def _normalize_tool_call(tc: Any) -> tuple[str, dict[str, Any]]:
    """Ollama tool calls come as objects with .function.name and .function.arguments (dict or json str)."""
    fn = tc["function"] if isinstance(tc, dict) else tc.function
    name = fn["name"] if isinstance(fn, dict) else fn.name
    raw_args = fn["arguments"] if isinstance(fn, dict) else fn.arguments
    if isinstance(raw_args, str):
        try:
            args = json.loads(raw_args) if raw_args else {}
        except json.JSONDecodeError:
            args = {}
    elif isinstance(raw_args, dict):
        args = raw_args
    else:
        args = dict(raw_args)
    return name, args


def run_turn(history: list[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    """Run a single user turn. `history` is the full message history including
    the new user message at the end. System prompt is prepended automatically."""
    messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}, *history]

    for _ in range(MAX_TOOL_ITERATIONS):
        try:
            stream = ollama.chat(
                model=MODEL,
                messages=messages,
                tools=tools.SCHEMAS,
                stream=True,
            )
        except Exception as e:
            yield {"type": "error", "message": f"Ollama error: {e}"}
            return

        accumulated_content = ""
        accumulated_tool_calls: list[Any] = []
        try:
            for chunk in stream:
                msg = chunk.get("message") if isinstance(chunk, dict) else chunk.message
                if msg is None:
                    continue
                content = msg.get("content") if isinstance(msg, dict) else msg.content
                if content:
                    accumulated_content += content
                    yield {"type": "token", "text": content}
                tc = msg.get("tool_calls") if isinstance(msg, dict) else getattr(msg, "tool_calls", None)
                if tc:
                    accumulated_tool_calls.extend(tc)
        except Exception as e:
            yield {"type": "error", "message": f"Stream error: {e}"}
            return

        assistant_msg: dict[str, Any] = {"role": "assistant", "content": accumulated_content}
        if accumulated_tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "function": {
                        "name": _normalize_tool_call(tc)[0],
                        "arguments": _normalize_tool_call(tc)[1],
                    }
                }
                for tc in accumulated_tool_calls
            ]
        messages.append(assistant_msg)

        if not accumulated_tool_calls:
            yield {"type": "done"}
            return

        for tc in accumulated_tool_calls:
            name, args = _normalize_tool_call(tc)
            yield {"type": "tool_start", "name": name, "args": args}
            allowed, result = tools.run_tool(name, args)
            yield {"type": "tool_result", "name": name, "allowed": allowed, "result": result}
            messages.append({"role": "tool", "name": name, "content": json.dumps(result, ensure_ascii=False)})

    yield {"type": "error", "message": "Max tool iterations reached."}
