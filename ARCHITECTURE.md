# Architecture & Stack — BNP × Mistral assistant

## 1. Use case

24/7 conversational assistant for a retail-banking client that:

1. **Answers** broad customer-service questions — balance, transactions, contract terms, agency hours, advisor contact.
2. **Acts** on low-risk reversible operations — internal transfers between the client's own accounts, temporary card lock.
3. **Escalates** anything risky or irreversible — outbound wires, profile changes — to a human advisor. Enforced at the **system layer**, not just the prompt.

The prototype models a single client (me) with realistic BNP-style products: Compte de chèques, Livret A, portefeuille investissement, Visa Premier, prêt études, assurance auto, et le contrat **Esprit Libre**.

---

## 2. Stack at a glance

```
┌──────────────────────────────────────────────────────────────────┐
│ Browser (vanilla HTML/CSS/JS, BNP theme)                         │  static/index.html
│   • SSE consumer · streaming markdown bubbles                    │  static/app.js
│   • tool-call chips · blocked indicator                          │  static/style.css
│   • LIVE accounts/cards/products sidebar (polls /state)          │
└────────────────────────────┬─────────────────────────────────────┘
                             │ HTTP · text/event-stream
┌────────────────────────────▼─────────────────────────────────────┐
│ FastAPI                                                          │  app/main.py
│   GET  /         → static/index.html                             │
│   GET  /state    → live JSON snapshot of mock bank               │
│   POST /chat     → SSE stream from agent loop                    │
└────────────────────────────┬─────────────────────────────────────┘
                             │
┌────────────────────────────▼─────────────────────────────────────┐
│ Agent loop                                                       │  app/agent.py
│   • Ollama chat client (streaming + native tool-calling)         │
│   • normalize tool calls → guardrails → tool → re-prompt         │
│   • Hard cap: 5 tool iterations / turn                           │
└────────┬───────────────────┬─────────────────────────────────────┘
         │                   │
   ┌─────▼─────┐       ┌─────▼──────────────────────────┐
   │ Ollama    │       │ Tool dispatcher                │  app/tools.py
   │ (local)   │       │   • arg coercion + filtering   │
   │ ministral │       │   • registry of 12 tools       │
   │   -3:8b   │       │   • routes through guardrails  │  app/guardrails.py
   └───────────┘       └───────┬────────────┬───────────┘
                               │            │
                       ┌───────▼──────┐  ┌──▼─────────────┐
                       │ SQLite       │  │ Chroma RAG     │  app/rag.py
                       │ mock bank    │  │  hybrid scoring│
                       │ (5 tables)   │  │ over .md docs  │  data/contracts/*.md
                       └──────────────┘  └────────────────┘
                       app/db.py
                       data/bank.sqlite
```

| Layer | Choice | One-line justification |
|-------|--------|------------------------|
| Inference | Ollama, local | Data sovereignty, GDPR posture, no token cost, no model leakage |
| Model | `ministral-3:8b` (default) / `:14b` | Mistral family, native function-calling. 8B fits ~6 GB VRAM, 14B ~10 GB |
| Orchestration | Plain Python loop | Easier to demo and audit than LangChain/LangGraph for a prototype |
| Tool wiring | In-process Python | One process, no IPC; can be swapped for MCP later (same schemas) |
| Bank data | SQLite | File-based, zero setup, easy to reset and seed |
| RAG store | Chroma (local persistent) | No server, no network call, embeds & queries from disk |
| Embeddings | `nomic-embed-text` (Ollama) | Same local-only story as the LLM; multilingual |
| API | FastAPI + SSE (`sse-starlette`) | Native streaming, type-safe, minimal boilerplate |
| Frontend | Vanilla HTML/JS + `marked` (CDN) | No build step, transparent for a panel walkthrough |

---

## 3. Why local Ollama + Ministral

Three reasons line up:

1. **Regulatory / GDPR.** A retail-bank assistant handles position data, transaction labels, IBANs. Routing that to a hosted endpoint either crosses borders or requires a complex DPA. Local inference keeps personal data on the bank's infrastructure.
2. **No training feedback loop.** Anything sent to a hosted provider is governed by their retention policy. A local model has zero risk of contributing customer data back into training.
3. **Latency control.** TTFT depends on the network only when the model is hosted. Locally we control the entire path; on a consumer GPU `ministral-3:8b` lands well under the 4 s ceiling and exceeds reading speed.

Trade-off: an 8B local model is weaker at deep reasoning than a frontier hosted model. We mitigate by **forcing tool use for all factual answers** — the model orchestrates, the tools provide ground truth, and a robust dispatcher (next section) absorbs the model's imperfect arg shapes.

Plus corresponds to what could be done by the bank on-premises.
---

## 4. The agent loop

`app/agent.py:run_turn` is a generator. It streams from Ollama, intercepts emitted tool calls, runs them through guardrails + tools, feeds results back, and loops.

```python
def run_turn(history):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, *history]

    for _ in range(MAX_TOOL_ITERATIONS):                   # cap = 5
        stream = ollama.chat(MODEL, messages=messages,
                             tools=tools.SCHEMAS, stream=True)
        accumulated_content, accumulated_tool_calls = "", []
        for chunk in stream:
            msg = chunk["message"] or chunk.message
            if msg.content:
                accumulated_content += msg.content
                yield {"type": "token", "text": msg.content}
            if msg.tool_calls:
                accumulated_tool_calls.extend(msg.tool_calls)

        messages.append({"role": "assistant", ...})
        if not accumulated_tool_calls:
            yield {"type": "done"}; return

        for tc in accumulated_tool_calls:
            name, args = _normalize_tool_call(tc)
            yield {"type": "tool_start", "name": name, "args": args}
            allowed, result = tools.run_tool(name, args)
            yield {"type": "tool_result", "name": name,
                   "allowed": allowed, "result": result}
            messages.append({"role": "tool", "name": name,
                             "content": json.dumps(result)})
```

Why streaming: TTFT is what the user perceives. Streaming makes the first token visible the moment Ollama produces it.

Why `_normalize_tool_call`: the Ollama Python client emits tool-call shapes that vary across versions and model behaviours — sometimes `arguments` is a dict, sometimes a JSON-encoded string, sometimes empty/malformed. Normalizing in one place keeps the rest of the loop trivial:

```python
def _normalize_tool_call(tc):
    fn = tc["function"] if isinstance(tc, dict) else tc.function
    name = fn["name"] if isinstance(fn, dict) else fn.name
    raw_args = fn["arguments"] if isinstance(fn, dict) else fn.arguments
    if isinstance(raw_args, str):
        try: args = json.loads(raw_args) if raw_args else {}
        except json.JSONDecodeError: args = {}
    elif isinstance(raw_args, dict):
        args = raw_args
    else:
        args = dict(raw_args)
    return name, args
```

Hard cap: a misaligned model can bounce between tools forever. 5-20 iterations covers any realistic chain (RAG → balance read → action) and fails loudly if exceeded.

---

## 5. Tool dispatcher — robustness as a feature

A small open-weight model gets argument shapes wrong often: extra keys, numbers as strings, French decimals (`"100,50"`), boolean strings (`"oui"`), unknown account names. If the dispatcher is brittle, the agent loop dies and the user sees a failure message. We make it **forgiving by design**.

`app/tools.py:run_tool` is the only path to tool execution. It does three things, in order:

```python
def run_tool(name, args):
    fn = REGISTRY.get(name)
    if fn is None:
        return False, {"error": f"unknown tool '{name}'"}
    if not isinstance(args, dict):
        args = {}

    clean_args = _normalize_args(fn, args)        # 1. filter + coerce
    decision = guardrails.check(name, clean_args) # 2. data-coded rules
    if not decision.allowed:
        return False, {"blocked": True, "reason": decision.reason}

    try:
        return True, fn(**clean_args)             # 3. invoke
    except TypeError as e:
        return False, {"error": f"bad arguments: {e}"}
    except ValueError as e:
        return False, {"error": str(e)}
```

`_normalize_args` reads each tool's signature, drops unknown kwargs silently, and coerces by annotation:

```python
def _normalize_args(fn, raw):
    sig = inspect.signature(fn)
    out = {}
    for pname, param in sig.parameters.items():
        if pname not in raw: continue
        v = raw[pname]
        target = _ANN_MAP.get(param.annotation)   # handles `from __future__`
        if target is not None:                    # → annotations are strings
            try: v = _coerce(v, target)
            except ValueError: pass
        out[pname] = v
    return out
```

`_coerce` accepts the shapes a real Mistral 8B has been observed to emit:

- `float`: `100`, `100.0`, `"100"`, `"100.50"`, `"100,50"` (French decimal)
- `int`: `"3"`, `4.0`, `True`
- `bool`: `True`, `"true"`/`"True"`, `"yes"`/`"oui"`, `"1"`, `1` (and falsy mirrors)

The 12 tools exposed to the model (see `tools.SCHEMAS`):

| Read | Write | RAG / Reference |
|------|-------|-----------------|
| `get_client_profile` | `transfer_internal` | `search_contracts` |
| `list_accounts` | `lock_card` | `get_counselor_contact` |
| `get_account_balance` | `transfer_external` *(always blocked)* | `get_agency_hours` |
| `list_transactions` | | |
| `list_cards` | | |
| `list_products` | | |

`get_client_profile` is the entry point: the system prompt instructs the model to call it whenever it needs the client's name, owned accounts, or product list. The agent never has identity hardcoded.

---

## 6. Mock banking layer — and the seam to a real one

`app/db.py` is a thin SQLite layer over JSON fixtures (`data/fixtures/*.json`). The agent doesn't know it's mocked — it sees the tool API. To swap in a real backend, replace the bodies of `db.list_accounts`, `db.get_transactions`, etc. The tool schemas in `app/tools.py:SCHEMAS` stay identical, the guardrails stay identical, the system prompt stays identical.

In a production BNP context, the seams would map to:

- **Read tools** → existing core-banking system / Cobol mainframe APIs, typically already exposed via an internal REST gateway.
- **Action tools** → the real movement engine and card-management system, with idempotency keys and audit logging.
- **Reference data** → the CRM.

Since the tool layer is the contract, swapping in MCP later is a refactor — each of the 12 tools becomes one MCP `tool` definition, and human-escalation routing becomes a second MCP server querying the CRM by topic / availability.

---

## 7. RAG design — hybrid retrieval

**Why per-product contracts.** The corpus mirrors the products the client owns (six markdown files). Index small, retrieval precise, citations meaningful. In production, gate retrieval by what the authenticated client is actually subscribed to — never let a model surface a clause for a product the client doesn't have.

**Pipeline:**

1. **Ingestion** (`rag.py:build_index`) — paragraph-aware chunking, ~500 chars with 80-char overlap. Source filename embedded in the chunk text (`[esprit_libre] # Découvert autorisé …`) so the embedding picks up document identity.
2. **Embeddings** — `nomic-embed-text` via Ollama (multilingual, ~270 MB).
3. **Store** — Chroma persistent client at `data/chroma/`, cosine distance.

```python
def retrieve(query, k=4):
    if not query.strip(): return []
    qtoks = _tokens(query)

    # Vector candidates (top 3k)
    res = coll.query(query_embeddings=[_embed([query])[0]],
                     n_results=max(k * 3, 12))
    seen = {f"{m['source']}#{m['chunk']}": {...} for ...}

    # Lexical pass: pull ALL chunks whose filename matches a query token.
    # Catches "Esprit Libre", "Livret A" when vector recall buries them.
    for doc, meta in coll.get(...):
        if any(t in meta["source"].lower() for t in qtoks):
            seen.setdefault(...)

    # Combine: vec_score + 0.05 * token_overlap + 0.6 * src_match
    candidates = [...]
    candidates.sort(key=lambda c: c["score"], reverse=True)
    return candidates[:k]
```

The agent prompt requires it to cite sources from `search_contracts` (e.g. *« selon `esprit_libre.md` »*).

In production, replace Chroma with the bank's vector index of choice (pgvector, OpenSearch), gate by client identity, and add a learned re-ranker if the latency budget permits.

---

## 8. Guardrails — three layers, code-coded rules

Defense in depth, each layer fails differently:

1. **Prompt layer** — the system prompt tells the model what's blocked and instructs it not to bypass.
2. **Schema layer** — `transfer_external` exists in the tool schema (so the model can attempt the call and learn it fails) but its dispatcher always returns a structured `blocked` payload.
3. **Code layer** — `app/guardrails.py:check` is the only path through `run_tool`.

```python
def check(tool_name, args) -> Decision:
    if tool_name == "transfer_external":
        return Decision(False, "Les virements externes (...) sont bloqués...")

    if tool_name == "transfer_internal":
        owned = _owned_account_ids()
        from_id, to_id = args.get("from_account"), args.get("to_account")
        try: amount = float(args.get("amount", 0))
        except (TypeError, ValueError):
            return Decision(False, "Le montant fourni n'est pas un nombre valide.")
        if from_id not in owned or to_id not in owned:
            return Decision(False, "Compte source ou destination inconnu.")
        if from_id == to_id:
            return Decision(False, "Compte source et destinataire identiques.")
        if amount <= 0:
            return Decision(False, "Le montant doit être strictement positif.")
        if amount > 10_000 and not args.get("confirmed"):
            return Decision(False, "Montant > 10 000 €. Confirmation requise.")
        src = db.get_account(from_id)
        if src and src["balance"] - amount < 0:
            return Decision(False, f"Solde insuffisant ({src['balance']:.2f} €). "
                                   "Découvert non autorisé.")
        return Decision(True)

    if tool_name in {"get_account_balance", "list_transactions"}:
        if (acc := args.get("account_id")) and acc not in _owned_account_ids():
            return Decision(False, "Ce compte n'appartient pas au client.")
        return Decision(True)

    return Decision(True)
```

| Rule | Effect |
|------|--------|
| `transfer_external` | Always blocked. Routes user to advisor / strong auth. |
| `transfer_internal` ∧ unknown account | Blocked. |
| `transfer_internal` ∧ same source/destination | Blocked. |
| `transfer_internal` ∧ amount ≤ 0 | Blocked. |
| `transfer_internal` ∧ amount > 10 000 € ∧ ¬confirmed | Blocked. |
| `transfer_internal` ∧ source balance − amount < 0 | Blocked (no overdraft). |
| Account id not owned by the client | Blocked. |
| All read tools, `lock_card`, `search_contracts`, `get_*` | Allowed. |

When a guardrail fires, the UI surfaces a red `⛔ bloqué: <tool>` chip on the assistant message

The 10 000 € confirm gate is intentionally below the overdraft check: a 50 000 € transfer with `confirmed=true` is still blocked on overdraft. Two independent gates, both must pass.

---

## 9. Latency budget

Targets from the brief:

- **TTFT < 2 s** — perceived first-token latency.
- **Tokens/sec > reading speed** — roughly > 5 tok/s.

What contributes to TTFT, in order:

1. Model load (cold) — Ollama keeps the model resident after first request; first call after boot pays a one-time cost.
2. Prompt prefix — system prompt + 12 tool schemas + history. Caches well in Ollama once warm.
3. First inference token — depends on GPU.

Measured on this machine: a one-tool turn (`Solde Livret A ?`) returns the full streamed response over HTTP in **~290 ms** warm.

If TTFT exceeds budget: (a) keep `ministral-3:8b` as the default and reserve 14B for tool-heavy turns; (b) shorten the system prompt; (c) trim tool schemas to the ones needed for the current intent (router pattern).

---

## 10. What changes in production

Out of scope for the prototype, listed so you can answer "what next" in the room:

- **Authentication & authorization.** Today: single hardcoded client via `get_client_profile`. Real: SSO / mobile auth, per-tool scopes, session timeouts, step-up auth for any write action.
- **Multi-tenant data isolation.** Today: one SQLite. Real: per-client row-level security, every tool scoped by authenticated client id.
- **Audit log.** Every tool call (allowed *and* blocked) persisted with user id, args, timestamp, model id, prompt hash. Required for regulatory traceability — the agent loop already emits these as events; you just need a sink.
- **Risk-model integration.** Hook the bank's existing fraud / risk engine into `guardrails.check()` — that function is exactly the right insertion point.
- **Observability.** Latency histograms (TTFT, tok/s, tool-call duration), cost-equivalent token counts, refusal rate, guardrail-fire rate.
- **Eval harness.** Golden set of ~100 user queries with expected tool calls and answer rubrics; regress on every model upgrade. The pytest scaffold is already in place.
- **Human escalation routing.** The brainstorm's MCP routing idea: a second MCP server that, given the user's topic and frustration signal, picks the right advisor or queue.
- **Compliance review.** Any RAG citation that contradicts a current legal document is a liability — versioning of the contract corpus, signed snapshots.
- **UX.** Confirmation modals for any write action, transcript export, language toggle, accessibility, dark mode.

---

## 11. Quick demo script

`./run.sh` launches everything (Ollama daemon if needed, missing model pulls, venv/conda setup, seed, uvicorn). Then in the browser:

1. *« Liste mes comptes. »* — tools: `get_client_profile` → `list_accounts`.
2. *« Quel est le solde de mon Livret A ? »* — single tool call, fast. Sidebar matches.
3. *« Mes 5 dernières transactions sur le compte de chèques. »* — formatted markdown list.
4. *« Que dit mon contrat Esprit Libre sur le découvert autorisé ? »* — RAG hit, citation `esprit_libre.md`.
5. *« Vire 1000 € de mon compte de chèques vers mon Livret A. »* — write executes, **sidebar flashes green and updates**.
6. *« Vire 50 000 € de mon compte de chèques vers mon Livret A. »* — guardrail blocks (overdraft), red ⛔ chip, sidebar unchanged.
7. *« Envoie 5000 € à l'IBAN FR76… »* — guardrail blocks (external), red chip, model pivots to advisor.
8. *« Bloque ma carte Visa Premier. »* — reversible action, card pill flips active → Bloquée.
9. *« Quels sont les horaires de mon agence le samedi ? »* — static lookup.

Each step exercises a different layer (identity / read / RAG / write / overdraft guard / external guard / card / static). The whole story in 90 seconds.
