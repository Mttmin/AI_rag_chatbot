const chatEl = document.getElementById("chat");
const formEl = document.getElementById("composer");
const inputEl = document.getElementById("input");
const sendBtn = document.getElementById("send");
const accountsEl = document.getElementById("accounts");
const cardsEl = document.getElementById("cards");
const productsEl = document.getElementById("products");
const refreshDot = document.getElementById("refresh-dot");

const history = [];
let welcomeRemoved = false;
let lastBalances = {};
let lastCardStatus = {};

function removeWelcome() {
  if (welcomeRemoved) return;
  const w = document.querySelector(".welcome");
  if (w) w.remove();
  welcomeRemoved = true;
}

if (window.marked) {
  marked.setOptions({ breaks: true, gfm: true });
}

function renderBubble(bubble) {
  const md = bubble.dataset.md || "";
  if (window.marked && bubble.dataset.markdown === "1") {
    bubble.innerHTML = marked.parse(md);
  } else {
    bubble.textContent = md;
  }
}

function addMessage(role, text = "") {
  const wrap = document.createElement("div");
  wrap.className = `msg ${role}`;
  const col = document.createElement("div");
  col.className = "msg-col";
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.dataset.md = text;
  bubble.dataset.markdown = role === "bot" ? "1" : "0";
  renderBubble(bubble);
  col.appendChild(bubble);
  wrap.appendChild(col);
  chatEl.appendChild(wrap);
  chatEl.scrollTop = chatEl.scrollHeight;
  return bubble;
}

function appendToBubble(bubble, text) {
  bubble.dataset.md = (bubble.dataset.md || "") + text;
  renderBubble(bubble);
}

function addToolChip(parentBubble, label, blocked) {
  const col = parentBubble.parentElement; // .msg-col
  let row = col.querySelector(".tool-row");
  if (!row) {
    row = document.createElement("div");
    row.className = "tool-row";
    col.insertBefore(row, parentBubble);
  }
  const chip = document.createElement("span");
  chip.className = "tool-chip" + (blocked ? " blocked" : "");
  chip.textContent = label;
  row.appendChild(chip);
}

function toolLabel(name, args) {
  const s = JSON.stringify(args || {});
  const trimmed = s.length > 60 ? s.slice(0, 57) + "…" : s;
  return `🔧 ${name}${trimmed === "{}" ? "" : " " + trimmed}`;
}

function fmtEUR(n) {
  return new Intl.NumberFormat("fr-FR", { style: "currency", currency: "EUR" }).format(n);
}

function renderAccounts(accounts) {
  accountsEl.innerHTML = "";
  for (const a of accounts) {
    const div = document.createElement("div");
    div.className = "account";
    if (lastBalances[a.id] !== undefined && lastBalances[a.id] !== a.balance) {
      div.classList.add("flash");
      setTimeout(() => div.classList.remove("flash"), 1500);
    }
    const balCls = a.balance < 0 ? "balance negative" : "balance";
    div.innerHTML = `
      <div class="row">
        <span class="label">${a.label}</span>
        <span class="${balCls}">${fmtEUR(a.balance)}</span>
      </div>
      ${a.iban ? `<div class="iban">${a.iban}</div>` : ""}
    `;
    accountsEl.appendChild(div);
    lastBalances[a.id] = a.balance;
  }
}

function renderCards(cards) {
  cardsEl.innerHTML = "";
  for (const c of cards) {
    const div = document.createElement("div");
    div.className = "card-item";
    if (lastCardStatus[c.id] !== undefined && lastCardStatus[c.id] !== c.status) {
      div.classList.add("flash");
      setTimeout(() => div.classList.remove("flash"), 1500);
    }
    div.innerHTML = `
      <div class="row">
        <span><strong>${c.label}</strong> · •••• ${c.last_four}</span>
        <span class="status-pill ${c.status}">${c.status === "active" ? "Active" : "Bloquée"}</span>
      </div>
    `;
    cardsEl.appendChild(div);
    lastCardStatus[c.id] = c.status;
  }
}

function renderProducts(products) {
  productsEl.innerHTML = "";
  for (const p of products) {
    const div = document.createElement("div");
    div.className = "product-item";
    let meta = "";
    if (p.monthly_fee) meta = `${fmtEUR(p.monthly_fee)} / mois`;
    if (p.remaining !== undefined) meta = `Restant : ${fmtEUR(p.remaining)} / ${fmtEUR(p.principal)}`;
    div.innerHTML = `
      <span class="label">${p.label}</span>
      ${meta ? `<span class="meta">${meta}</span>` : ""}
    `;
    productsEl.appendChild(div);
  }
}

async function refreshState() {
  try {
    const r = await fetch("/state");
    if (!r.ok) return;
    const s = await r.json();
    renderAccounts(s.accounts);
    renderCards(s.cards);
    renderProducts(s.products);
    refreshDot.classList.add("flash");
    setTimeout(() => refreshDot.classList.remove("flash"), 600);
  } catch (e) {
    console.error("state refresh failed", e);
  }
}

async function sendMessage(text) {
  removeWelcome();
  if (!text.trim()) return;
  history.push({ role: "user", content: text });
  addMessage("user", text);
  inputEl.value = "";
  sendBtn.disabled = true;

  const botBubble = addMessage("bot", "");
  botBubble.classList.add("cursor");
  let assistantText = "";

  try {
    const resp = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ messages: history }),
    });
    if (!resp.ok || !resp.body) throw new Error(`HTTP ${resp.status}`);

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    const SEP = /\r\n\r\n|\n\n/;
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let m;
      while ((m = SEP.exec(buffer))) {
        const raw = buffer.slice(0, m.index);
        buffer = buffer.slice(m.index + m[0].length);
        const dataLines = raw
          .split(/\r?\n/)
          .filter(l => l.startsWith("data:"))
          .map(l => l.slice(5).replace(/^ /, ""));
        if (!dataLines.length) continue;
        const dataStr = dataLines.join("\n");
        let payload;
        try {
          payload = JSON.parse(dataStr);
        } catch (err) {
          console.warn("SSE parse failed:", dataStr, err);
          continue;
        }
        handleEvent(payload, botBubble);
        if (payload.type === "token") assistantText += payload.text;
      }
    }
  } catch (e) {
    appendToBubble(botBubble, `\n\n_[erreur: ${e.message}]_`);
  } finally {
    botBubble.classList.remove("cursor");
    sendBtn.disabled = false;
    if (assistantText) history.push({ role: "assistant", content: assistantText });
    inputEl.focus();
    refreshState();
  }
}

const WRITE_TOOLS = new Set(["transfer_internal", "lock_card"]);

function handleEvent(ev, botBubble) {
  if (ev.type === "token") {
    appendToBubble(botBubble, ev.text);
    chatEl.scrollTop = chatEl.scrollHeight;
  } else if (ev.type === "tool_start") {
    addToolChip(botBubble, toolLabel(ev.name, ev.args), false);
  } else if (ev.type === "tool_result") {
    if (!ev.allowed) {
      addToolChip(botBubble, `⛔ bloqué: ${ev.name}`, true);
    } else if (WRITE_TOOLS.has(ev.name)) {
      refreshState();
    }
  } else if (ev.type === "error") {
    appendToBubble(botBubble, `\n\n_[erreur agent: ${ev.message}]_`);
  }
}

formEl.addEventListener("submit", (e) => {
  e.preventDefault();
  sendMessage(inputEl.value);
});

document.querySelectorAll(".chip-btn").forEach(btn => {
  btn.addEventListener("click", () => sendMessage(btn.dataset.q));
});

refreshState();
inputEl.focus();
