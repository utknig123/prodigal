const $ = (s) => document.querySelector(s);
const FIELDS = ["full_name","phone","email","college","degree","branch","graduation_year","cgpa","preferred_location","available_from","internship_duration_months","linkedin_url","github_url","skills","collection_status","attempt_count","last_contacted_at","notes"];
let token = null, busy = false, idleTimer = null, selected = null;
const IDLE_MS = 5 * 60 * 1000;

const API = String(window.ACME_API || "").replace(/\/$/, "");
async function api(path, body) {
  const r = await fetch(API + "/api/" + path, { method: body ? "POST" : "GET", headers: { "content-type": "application/json" }, body: body ? JSON.stringify(body) : undefined });
  const j = await r.json().catch(() => ({ error: "Bad response" }));
  if (!r.ok) throw new Error(j.error || r.statusText);
  return j;
}
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const fmt = (v) => (v === null || v === undefined || v === "" ? "—" : typeof v === "object" ? JSON.stringify(v) : String(v));

async function health() {
  const el = $("#health");
  try {
    const h = await api("health");
    const ok = h.llm.configured && h.candidate_api_auth === "ok";
    const chain = (h.llm.chain || []).map((c) => c.model).join(" → ");
    el.textContent = ok ? `Ready · ${chain || h.llm.model}` : !h.llm.configured ? "LLM key missing" : "Candidate API: " + h.candidate_api_auth;
    el.className = "pill " + (ok ? "ok" : "bad");
  } catch (e) { el.textContent = "Server unreachable"; el.className = "pill bad"; }
}

async function loadCands() {
  const box = $("#cands");
  try {
    const { candidates } = await api("candidates");
    box.innerHTML = candidates.map((c) => `<button class="cand ${c.id === selected ? "sel" : ""}" data-id="${esc(c.id)}">
      <div class="r"><b>${esc(c.name)}</b><span class="chip ${esc(c.status)}">${esc(c.status)}</span></div>
      <small>${esc(c.id)} · ${c.todo.length ? c.todo.length + " to collect" : "complete"} · attempts ${c.attempts ?? 0}</small></button>`).join("") || '<p class="muted">No candidates</p>';
  } catch (e) { box.innerHTML = `<p class="muted">${esc(e.message)}</p>`; }
}

function addMsg(role, text, ts) {
  const d = document.createElement("div");
  d.className = "m " + role;
  const t = ts ? new Date(ts).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }) : "";
  d.innerHTML = esc(text) + (t ? `<time>${t}</time>` : "");
  $("#msgs").appendChild(d);
  $("#msgs").scrollTop = 1e9;
}
function typing(on) {
  document.querySelector(".typing")?.remove();
  if (on) { const d = document.createElement("div"); d.className = "typing"; d.innerHTML = "<i></i><i></i><i></i>"; $("#msgs").appendChild(d); $("#msgs").scrollTop = 1e9; }
}
function render(v) {
  token = v.token;
  $("#msgs").innerHTML = "";
  v.transcript.forEach((m) => addMsg(m.role, m.text, m.ts));
  $("#ch-name").textContent = v.record.full_name;
  $("#av").textContent = (v.record.full_name || "?")[0];
  $("#ch-sub").textContent = `${v.candidate_id} · ${v.ended ? "chat ended" : "in conversation"}`;
  const focus = $("#focus");
  if (focus) {
    if (!v.ended && v.focus) { focus.hidden = false; focus.textContent = "Asking for " + v.focus.replaceAll("_", " "); }
    else focus.hidden = true;
  }
  const open = (v.plan.missing || []).length + (v.plan.invalid || []).length + (v.plan.confirm || []).length;
  const bar = $("#progress");
  if (bar) bar.style.width = (v.ended ? 100 : Math.max(8, Math.round(100 * (1 - open / 9)))) + "%";
  $("#leave").hidden = v.ended;
  $("#text").disabled = $("#composer button").disabled = v.ended;
  const b = $("#banner");
  if (v.ended) {
    const o = v.outcome;
    b.hidden = false;
    b.className = "banner " + (o === "completed" ? "completed" : o === "needs_human" || o === "refused" ? "bad" : "");
    const hand = v.handoff;
    b.textContent = `Outcome recorded: ${o}` + (v.final_write_ok === false ? " — WARNING: final database write failed" : "");
    if (hand && hand.next_question) {
      const extra = document.createElement("div");
      extra.className = "handoff";
      extra.textContent = `Handoff: ${hand.next_question} Still needed: ${(hand.still_needed || []).join(", ") || "none"}.`;
      b.appendChild(extra);
    }
  } else b.hidden = true;
  renderTrace(v.trace); renderRecord(v); renderPlan(v.plan); renderLedger(v.trace);
  clearTimeout(idleTimer);
  if (!v.ended) idleTimer = setTimeout(() => leave("no reply / idle timeout"), IDLE_MS);
}
function renderTrace(tr) {
  const out = [];
  for (const e of tr) {
    if (e.t === "tool") {
      const bad = e.result && (e.result.ok === false || (e.result.results || []).some((r) => r.status !== "ok" && r.status !== "unchanged"));
      out.push(`<div class="ev tool ${bad ? "warn" : ""}"><b>TOOL · ${esc(e.name)}</b><pre>${esc(JSON.stringify(e.args, null, 1))}\n→ ${esc(JSON.stringify(e.result, null, 1))}</pre></div>`);
    } else if (e.t === "db") {
      if (e.method === "GET" && !e.attempt && e.status === 200) out.push(`<div class="ev db"><b>DB · GET ${esc(e.path)} → ${e.status} (${e.ms}ms)</b></div>`);
      else out.push(`<div class="ev db ${e.status >= 400 || e.status === 0 ? "bad" : ""}"><b>DB · ${esc(e.method)} ${esc(e.path)} → ${e.status} (${e.ms}ms)</b>${e.request ? `<pre>${esc(JSON.stringify(e.request, null, 1))}</pre>` : ""}</div>`);
    } else if (e.t === "guardrail_reply_blocked") out.push(`<div class="ev warn"><b>GUARDRAIL · reply blocked</b>${esc(e.reason)}</div>`);
    else if (e.t === "conflict_resolved") out.push(`<div class="ev warn"><b>409 conflict · re-read and merged</b>${esc(JSON.stringify(e.dropped))}</div>`);
    else if (e.t === "llm_error" || e.t === "tool_format_error") out.push(`<div class="ev bad"><b>${esc(e.t)}</b>${esc(e.error || e.failed || "")}</div>`);
    else if (e.t === "abandoned") out.push(`<div class="ev warn"><b>CHAT ABANDONED → ${esc(e.outcome)}</b>${esc(e.reason)}</div>`);
  }
  $("#t-act").innerHTML = out.join("") || '<p class="muted">No activity yet.</p>';
}
function renderRecord(v) {
  const rows = FIELDS.map((f) => {
    const a = v.initial[f], b = v.record[f];
    const changed = JSON.stringify(a) !== JSON.stringify(b);
    const miss = (b === null || b === "") && !["linkedin_url","github_url","skills","notes","last_contacted_at"].includes(f);
    return `<tr class="${changed ? "chg" : miss ? "miss" : ""}"><td>${f}</td><td>${changed ? `<s>${esc(fmt(a))}</s> → ` : ""}${esc(fmt(b))}</td></tr>`;
  });
  $("#t-rec").innerHTML = `<table class="rec">${rows.join("")}</table><p class="muted">Green = written by the agent this chat. Red = still missing.</p>`;
}
function renderPlan(p) {
  const li = (a) => a.length ? a.map((x) => `<li>${esc(typeof x === "string" ? x : `${x.field}: ${x.value} — ${x.reason}`)}</li>`).join("") : "<li class='muted'>none</li>";
  $("#t-plan").innerHTML = `<h4>Missing</h4><ul>${li(p.missing)}</ul><h4>Existing data that looks wrong</h4><ul>${li(p.invalid)}</ul><h4>Needs confirmation</h4><ul>${li(p.confirm)}</ul><h4>Auto-normalised</h4><ul>${li(Object.entries(p.autofix || {}).map(([k, val]) => `${k} → ${val}`))}</ul><h4>Optional, ask once</h4><ul>${li(p.optional_missing)}</ul>`;
}
function renderLedger(trace) {
  const box = $("#t-ledger");
  if (!box) return;
  const cards = [];
  for (const e of trace || []) {
    if (e.t !== "tool" || e.name !== "save_fields") continue;
    for (const row of (e.result && e.result.results) || []) {
      const quote = ((e.args && e.args.fields) || []).find((f) => f.field === row.field);
      cards.push(`<div class="ledger"><b>${esc(row.field)} · ${esc(row.status)}</b><div>${esc(fmt(row.value))}</div>${quote && quote.evidence ? `<q>“${esc(quote.evidence)}”</q>` : ""}</div>`);
    }
  }
  box.innerHTML = cards.join("") || '<p class="muted">Nothing has been written yet. Each save shows the value and the candidate’s exact words.</p>';
}

async function start(id) {
  if (busy || !id) return;
  busy = true; selected = id; token = null;
  $("#msgs").innerHTML = ""; typing(true); $("#banner").hidden = true;
  setPanel("p-chat");
  try {
    const v = await api("start", { candidate_id: id });
    typing(false);
    if (v.no_action) {
      $("#msgs").innerHTML = `<div class="empty"><b>${esc(v.record.full_name)}</b>'s record is already complete and valid, so no conversation was started.<br>Status set to <b>completed</b> without contacting them.</div>`;
      $("#ch-name").textContent = v.record.full_name; $("#ch-sub").textContent = id + " · nothing to collect";
      renderTrace(v.trace); renderRecord(v); renderPlan(v.plan); renderLedger(v.trace); $("#text").disabled = true; $("#leave").hidden = true;
    } else { render(v); $("#text").focus(); }
  } catch (e) { typing(false); $("#msgs").innerHTML = `<div class="empty">${esc(e.message)}</div>`; }
  busy = false; loadCands();
}
async function send(text) {
  if (busy || !token) return;
  busy = true; addMsg("candidate", text, new Date()); typing(true);
  $("#text").value = ""; $("#text").disabled = true;
  try { render(await api("message", { token, text })); }
  catch (e) { typing(false); addMsg("agent", "⚠ " + e.message); }
  busy = false;
  if (!$("#text").disabled || !$("#banner").hidden === false) $("#text").focus();
  loadCands();
}
async function leave(reason) {
  if (!token) return;
  try { render(await api("leave", { token, reason })); loadCands(); } catch (e) { /* ignore */ }
}
function setPanel(id) {
  document.querySelectorAll(".panel").forEach((p) => p.classList.toggle("show", p.id === id));
  document.querySelectorAll("#mtabs button").forEach((b) => b.classList.toggle("on", b.dataset.p === id));
}

$("#cands").addEventListener("click", (e) => { const b = e.target.closest(".cand"); if (b) start(b.dataset.id); });
$("#manual").addEventListener("submit", (e) => { e.preventDefault(); start($("#cid").value.trim()); });
$("#composer").addEventListener("submit", (e) => { e.preventDefault(); const t = $("#text").value.trim(); if (t) send(t); });
$("#leave").addEventListener("click", () => leave("candidate left the chat"));
$("#reset").addEventListener("click", async () => { if (confirm("Reset your workspace to the original data?")) { await api("reset", {}); token = null; $("#msgs").innerHTML = '<div class="empty">Workspace reset.</div>'; loadCands(); } });
document.querySelectorAll(".itabs button").forEach((b) => b.addEventListener("click", () => {
  document.querySelectorAll(".itabs button,.tab").forEach((x) => x.classList.remove("on")); b.classList.add("on"); $("#" + b.dataset.t).classList.add("on");
}));
document.querySelectorAll("#mtabs button").forEach((b) => b.addEventListener("click", () => setPanel(b.dataset.p)));
window.addEventListener("pagehide", () => { if (token) navigator.sendBeacon(API + "/api/leave", new Blob([JSON.stringify({ token, reason: "candidate closed the page" })], { type: "application/json" })); });
if (matchMedia("(max-width:760px)").matches) setPanel("p-list");
health(); loadCands();
