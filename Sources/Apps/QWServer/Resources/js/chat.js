/* Dedicated local Chat & Agent workbench controller */

const CHAT_SESSIONS_STORAGE_KEY = "qwg.webui.chats.v1";
const CHAT_PRESETS_STORAGE_KEY = "qwg.webui.presets.v1";
const CHAT_UI_STATE_STORAGE_KEY = "qwg.webui.chat_ui.v1";
const CHAT_PARAM_INPUT_BINDINGS = [
  ["chat-param-temp", "temperature"], ["chat-param-top-p", "topP"], ["chat-param-top-k", "topK"],
  ["chat-param-max-tokens", "maxTokens"], ["chat-param-rep-pen", "repetitionPenalty"], ["chat-param-seed", "seed"]
];
const chatState = {
  initialized: false, sessions: [], activeSessionId: null, customPresets: [],
  sidebarCollapsed: false, openDrawer: null, pendingImages: [], pendingVideos: [], editingMsgId: null,
  clearAllArmedUntil: 0, isGenerating: false, activeAbortController: null, activeStatusLabel: "Ready",
  autoFollowScroll: true, lastTranscriptScrollTop: 0, ignoreScrollUntil: 0,
  transcriptListenersBound: false, reasoningOpenById: {},
  reasoningQuip: { msgId: null, index: -1, text: "", timerId: null }
};
const CHAT_REASONING_QUIP_INTERVAL_MS = 3500;

function chatEl(id) { return typeof document !== "undefined" ? document.getElementById(id) : null; }
function generateChatId(prefix) { return `${prefix}-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 7)}`; }
function notifyChat(msg, isError) { if (typeof showToast === "function") showToast(msg, isError); }
function createDefaultChatParams() {
  return {
    temperature: null, topP: null, topK: null, maxTokens: null, repetitionPenalty: null, seed: null,
    enableThinking: "inherit", reasoningEffort: "inherit", stream: true, responseFormat: "text",
    jsonSchema: "", stop: "", toolsJson: "",
    cachePrompt: true, promptCacheKey: "", store: true, exactSpeculation: "inherit", customMetadata: ""
  };
}
function createBlankChatSession(title, presetId, systemPrompt, workspacePath) {
  const now = Date.now();
  return {
    id: generateChatId("chat"), title: title || "New Chat", presetId: presetId || "server-default",
    systemPrompt: typeof systemPrompt === "string" ? systemPrompt : "",
    workspacePath: typeof workspacePath === "string" ? workspacePath : "",
    harnessEnabled: true, useMemoryPackHarness: true,
    params: createDefaultChatParams(),
    messages: [], createdAt: now, updatedAt: now
  };
}
function safeReadStorage(key) {
  if (typeof localStorage === "undefined") return null;
  try { const raw = localStorage.getItem(key); return raw ? JSON.parse(raw) : null; } catch (_) { return null; }
}
function safeWriteStorage(key, value) {
  if (typeof localStorage === "undefined") return;
  try { localStorage.setItem(key, JSON.stringify(value)); } catch (err) { notifyChat("Storage quota warning: " + err.message, true); }
}
function loadChatPersistence() {
  const savedPresets = safeReadStorage(CHAT_PRESETS_STORAGE_KEY);
  chatState.customPresets = Array.isArray(savedPresets) ? savedPresets.filter((p) => p && p.id && p.name) : [];
  const savedSessions = safeReadStorage(CHAT_SESSIONS_STORAGE_KEY);
  chatState.sessions = Array.isArray(savedSessions) && savedSessions.length > 0
    ? savedSessions.map((s) => Object.assign(createBlankChatSession(s.title, s.presetId, s.systemPrompt), s, {
        params: Object.assign(createDefaultChatParams(), s.params || {}),
        messages: Array.isArray(s.messages) ? s.messages : []
      }))
    : [createBlankChatSession("New Chat", "server-default", "")];
  const uiPrefs = safeReadStorage(CHAT_UI_STATE_STORAGE_KEY) || {};
  chatState.sidebarCollapsed = uiPrefs.sidebarCollapsed === true;
  const matched = chatState.sessions.find((s) => s.id === uiPrefs.activeSessionId);
  chatState.activeSessionId = matched ? matched.id : chatState.sessions[0].id;
}
function persistChatSessions() {
  safeWriteStorage(CHAT_SESSIONS_STORAGE_KEY, chatState.sessions);
  safeWriteStorage(CHAT_UI_STATE_STORAGE_KEY, { activeSessionId: chatState.activeSessionId, sidebarCollapsed: chatState.sidebarCollapsed });
}
function persistCustomPresets() { safeWriteStorage(CHAT_PRESETS_STORAGE_KEY, chatState.customPresets); }
function getAllSystemPresets() { return BUILT_IN_SYSTEM_PRESETS.concat(chatState.customPresets || []); }
function getActiveChatSession() {
  if (!chatState.sessions.length) { const f = createBlankChatSession("New Chat", "server-default", ""); chatState.sessions.push(f); chatState.activeSessionId = f.id; }
  let s = chatState.sessions.find((x) => x.id === chatState.activeSessionId);
  if (!s) { s = chatState.sessions[0]; chatState.activeSessionId = s.id; }
  return s;
}
function commitChatWorkspace(session) {
  if (session) session.updatedAt = Date.now();
  persistChatSessions();
  renderChatWorkspace();
}
function initChatWorkspace() { if (!chatState.initialized) { loadChatPersistence(); chatState.initialized = true; } bindChatTranscriptListeners(); renderChatWorkspace(); }
function onChatTabActivated() {
  initChatWorkspace();
  if (typeof loadedSettingsObj !== "undefined" && !loadedSettingsObj && typeof loadLiveSettings === "function") loadLiveSettings();
  const input = chatEl("chat-input");
  if (input && typeof input.focus === "function") setTimeout(() => input.focus(), 40);
}
function renderChatWorkspace() {
  if (typeof document === "undefined") return;
  applyChatSidebarState();
  renderChatSessionList();
  renderActiveChatHeader();
  renderChatSystemDrawer();
  renderChatParamsDrawer();
  renderChatTranscript();
  if (typeof renderChatAttachments === "function") renderChatAttachments();
  renderChatComposerState();
}
function applyChatSidebarState() {
  if (chatEl("chat-layout")) chatEl("chat-layout").classList.toggle("sidebar-collapsed", chatState.sidebarCollapsed);
  if (chatEl("chat-expand-sidebar-btn")) chatEl("chat-expand-sidebar-btn").hidden = !chatState.sidebarCollapsed;
}
function toggleChatSidebar() { chatState.sidebarCollapsed = !chatState.sidebarCollapsed; persistChatSessions(); applyChatSidebarState(); }
function toggleChatDrawer(drawerName) { chatState.openDrawer = chatState.openDrawer === drawerName ? null : drawerName; syncChatDrawerVisibility(); }
function syncChatDrawerVisibility() {
  const showSys = chatState.openDrawer === "system";
  const showParams = chatState.openDrawer === "params";
  if (chatEl("chat-system-drawer")) chatEl("chat-system-drawer").hidden = !showSys;
  if (chatEl("chat-params-drawer")) chatEl("chat-params-drawer").hidden = !showParams;
  if (chatEl("chat-toggle-system-btn")) chatEl("chat-toggle-system-btn").setAttribute("aria-expanded", showSys ? "true" : "false");
  if (chatEl("chat-toggle-params-btn")) chatEl("chat-toggle-params-btn").setAttribute("aria-expanded", showParams ? "true" : "false");
}
function renderChatSessionList() {
  const listEl = chatEl("chat-session-list");
  if (!listEl) return;
  const query = chatEl("chat-session-search") ? chatEl("chat-session-search").value.trim().toLowerCase() : "";
  const sorted = chatState.sessions.slice().sort((a, b) => (b.updatedAt || -1) - (a.updatedAt || -1));
  const filtered = query ? sorted.filter((s) => (s.title || "").toLowerCase().includes(query)
    || s.messages.some((m) => typeof m.content === "string" && m.content.toLowerCase().includes(query))) : sorted;
  if (!filtered.length) { listEl.innerHTML = `<div class="panel-sub" style="padding:10px 4px;">${query ? "No matching sessions." : "No saved sessions."}</div>`; return; }
  listEl.innerHTML = filtered.map((s) => {
    const turns = s.messages.length;
    const stamp = s.updatedAt ? new Date(s.updatedAt).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }) : "—";
    return `<div class="chat-session-item ${s.id === chatState.activeSessionId ? "is-active" : ""}" role="listitem">
      <button type="button" class="chat-session-select-btn" data-chat-action="select-session" data-session-id="${escapeHtml(s.id)}">
        <div class="chat-session-title">${escapeHtml(s.title || "Untitled Chat")}</div>
        <div class="chat-session-meta mono">${turns} ${turns === 1 ? "turn" : "turns"} · ${escapeHtml(stamp)}</div>
      </button>
      <button type="button" class="chat-session-del" data-chat-action="delete-session" data-session-id="${escapeHtml(s.id)}" aria-label="Delete session ${escapeHtml(s.title || "Untitled")}">×</button>
    </div>`;
  }).join("");
}
function activateFreshSessionState(session) {
  chatState.activeSessionId = session.id; chatState.pendingImages = []; chatState.pendingVideos = []; chatState.editingMsgId = null;
  chatState.autoFollowScroll = true; chatState.lastTranscriptScrollTop = 0; commitChatWorkspace(null);
}
function createNewChatSession() {
  const a = getActiveChatSession();
  const f = createBlankChatSession("New Chat", a ? a.presetId : "server-default", a ? a.systemPrompt : "", a ? a.workspacePath : "");
  if (a && typeof a.harnessEnabled === "boolean") f.harnessEnabled = a.harnessEnabled;
  if (a && typeof a.useMemoryPackHarness === "boolean") f.useMemoryPackHarness = a.useMemoryPackHarness;
  if (typeof applyPresetHarnessToSession === "function") applyPresetHarnessToSession(f, f.presetId);
  chatState.sessions.unshift(f);
  activateFreshSessionState(f);
}
function selectChatSession(sessionId) {
  if (chatState.isGenerating) { notifyChat("Cancel active generation before switching sessions.", true); return; }
  const found = chatState.sessions.find((s) => s.id === sessionId);
  if (found) activateFreshSessionState(found);
}
function deleteChatSession(sessionId) {
  if (chatState.isGenerating && chatState.activeSessionId === sessionId) stopActiveChatGeneration();
  chatState.sessions = chatState.sessions.filter((s) => s.id !== sessionId);
  if (!chatState.sessions.length) chatState.sessions = [createBlankChatSession("New Chat", "server-default", "")];
  if (!chatState.sessions.some((s) => s.id === chatState.activeSessionId)) chatState.activeSessionId = chatState.sessions[0].id;
  commitChatWorkspace(null);
}
function clearAllChatSessions() {
  const now = Date.now();
  const btn = document.querySelector(".chat-sidebar-foot .btn-danger");
  if (now > chatState.clearAllArmedUntil) {
    chatState.clearAllArmedUntil = now + 4000;
    if (btn) btn.textContent = "Confirm Clear?";
    setTimeout(() => { if (Date.now() >= chatState.clearAllArmedUntil && btn) btn.textContent = "Clear All"; }, 4100);
    return;
  }
  chatState.clearAllArmedUntil = 0;
  if (btn) btn.textContent = "Clear All";
  if (chatState.isGenerating) stopActiveChatGeneration();
  const f = createBlankChatSession("New Chat", "server-default", "");
  chatState.sessions = [f];
  activateFreshSessionState(f);
  notifyChat("Cleared all saved chat sessions.");
}
function renameActiveChatSession(nextTitle) {
  const s = getActiveChatSession(); s.title = (nextTitle || "").trim() || "Untitled Chat"; s.updatedAt = Date.now();
  persistChatSessions(); renderChatSessionList();
}
function resetActiveChatMessages() {
  if (chatState.isGenerating) stopActiveChatGeneration();
  const s = getActiveChatSession(); s.messages = []; chatState.editingMsgId = null;
  commitChatWorkspace(s); notifyChat("Cleared conversation turns.");
}
function renderActiveChatHeader() {
  const s = getActiveChatSession(); const el = chatEl("chat-title-input");
  if (el && el.value !== s.title) el.value = s.title;
  syncChatHeaderWithDashboard();
}
function syncChatHeaderWithDashboard() {
  if (typeof document === "undefined") return;
  const connected = typeof isConnected !== "undefined" && isConnected;
  const modelObj = typeof publishedModel !== "undefined" ? publishedModel : null;
  if (chatEl("chat-model-badge")) chatEl("chat-model-badge").textContent = connected && modelObj && modelObj.modelId ? modelObj.modelId : "—";
  if (chatEl("chat-vision-badge")) {
    const caps = modelObj && Array.isArray(modelObj.capabilities) ? modelObj.capabilities : [];
    chatEl("chat-vision-badge").textContent = (!connected || !modelObj) ? "Vision: —" : (caps.includes("vision") ? "Vision: Ready" : "Text-Only");
  }
  const sidecarEl = chatEl("chat-sidecar-badge");
  if (sidecarEl) {
    const facts = modelObj && modelObj.facts ? modelObj.facts : null;
    if (connected && facts && facts.decisionSidecarLoaded) {
      const lat = typeof facts.decisionSidecarMeanLatencyMs === "number" ? ` · ${facts.decisionSidecarMeanLatencyMs.toFixed(1)}ms` : "";
      sidecarEl.hidden = false;
      sidecarEl.textContent = `Sidecar: ${facts.decisionSidecarModel || "Ready"}${lat}`;
    } else {
      sidecarEl.hidden = true;
    }
  }
  const session = getActiveChatSession();
  if (chatEl("chat-preset-badge")) {
    const matched = getAllSystemPresets().find((p) => p.id === session.presetId);
    chatEl("chat-preset-badge").textContent = matched ? matched.name : "Custom";
  }
  if (typeof renderChatMemoryPackAndHarnessUI === "function") renderChatMemoryPackAndHarnessUI(session);
  if (typeof renderChatAttachments === "function") renderChatAttachments();
}
function syncChatWithServerSettings(_) { if (typeof document !== "undefined") renderChatSystemDrawer(); }
function renderChatSystemDrawer() {
  const session = getActiveChatSession();
  const selectEl = chatEl("chat-preset-select");
  const promptEl = chatEl("chat-system-prompt");
  const hintEl = chatEl("chat-system-hint");
  const presets = getAllSystemPresets();
  if (selectEl) {
    const known = presets.some((p) => p.id === session.presetId);
    const options = presets.map((p) => `<option value="${escapeHtml(p.id)}">${escapeHtml(p.name + (p.isCustom ? " [Custom]" : ""))}</option>`);
    if (!known) options.push(`<option value="custom-inline">Custom (Unsaved)</option>`);
    selectEl.innerHTML = options.join("");
    selectEl.value = known ? session.presetId : "custom-inline";
  }
  const activePreset = presets.find((p) => p.id === session.presetId);
  if (chatEl("chat-delete-preset-btn")) chatEl("chat-delete-preset-btn").hidden = !(activePreset && activePreset.isCustom);
  if (promptEl && promptEl.value !== (session.systemPrompt || "")) promptEl.value = session.systemPrompt || "";
  if (hintEl) {
    const srv = typeof loadedSettingsObj !== "undefined" && loadedSettingsObj ? loadedSettingsObj.systemPrompt : "";
    hintEl.textContent = !(session.systemPrompt || "").trim() && srv
      ? `Server default active (${srv.slice(0, 60)}${srv.length > 60 ? "…" : ""}).`
      : "Prepended as the leading system message for Warm RAM prefix-cache reuse.";
  }
  if (chatEl("chat-preset-badge")) chatEl("chat-preset-badge").textContent = activePreset ? activePreset.name : "Custom";
  if (typeof renderChatMemoryPackAndHarnessUI === "function") renderChatMemoryPackAndHarnessUI(session);
  syncChatDrawerVisibility();
}
function selectChatSystemPreset(presetId) {
  const found = getAllSystemPresets().find((p) => p.id === presetId);
  if (!found) return;
  const s = getActiveChatSession(); s.presetId = found.id; s.systemPrompt = found.prompt || "";
  if (typeof applyPresetHarnessToSession === "function") applyPresetHarnessToSession(s, found.id);
  s.updatedAt = Date.now();
  persistChatSessions(); renderChatSystemDrawer(); renderChatParamsDrawer();
}
function updateActiveChatSystemPrompt(value) {
  const s = getActiveChatSession(); s.systemPrompt = value;
  const matched = getAllSystemPresets().find((p) => p.prompt === value);
  s.presetId = matched ? matched.id : "custom-inline"; s.updatedAt = Date.now();
  persistChatSessions(); renderChatSystemDrawer();
}
function saveCustomSystemPreset() {
  const drawer = chatEl("chat-system-drawer");
  if (!drawer) return;
  if (chatEl("chat-preset-inline-save")) { if (chatEl("chat-preset-name-input")) chatEl("chat-preset-name-input").focus(); return; }
  const box = document.createElement("div");
  box.id = "chat-preset-inline-save"; box.className = "chat-preset-inline-save";
  box.innerHTML = `<label for="chat-preset-name-input" class="chat-field-label">Preset Name</label>
    <input id="chat-preset-name-input" name="chat-preset-name-input" autocomplete="off" class="input select-sm" style="max-width:220px;" placeholder="e.g. Swift Reviewer…" />
    <button type="button" class="btn btn-primary btn-sm" onclick="confirmSaveCustomSystemPreset()">Save</button>
    <button type="button" class="btn btn-sm" onclick="cancelSaveCustomSystemPreset()">Cancel</button>`;
  drawer.insertBefore(box, chatEl("chat-system-prompt"));
  if (chatEl("chat-preset-name-input")) chatEl("chat-preset-name-input").focus();
}
function confirmSaveCustomSystemPreset() {
  const nameInput = chatEl("chat-preset-name-input");
  const rawName = nameInput ? nameInput.value.trim() : "";
  if (!rawName) { notifyChat("Enter a preset name.", true); if (nameInput) nameInput.focus(); return; }
  const session = getActiveChatSession();
  let saved = chatState.customPresets.find((p) => p.name.toLowerCase() === rawName.toLowerCase());
  if (saved) { saved.name = rawName; saved.prompt = session.systemPrompt || ""; }
  else { saved = { id: generateChatId("preset"), name: rawName, prompt: session.systemPrompt || "", isCustom: true }; chatState.customPresets.push(saved); }
  session.presetId = saved.id; session.updatedAt = Date.now();
  persistCustomPresets(); persistChatSessions(); cancelSaveCustomSystemPreset(); renderChatSystemDrawer(); notifyChat(`Saved preset "${rawName}".`);
}
function cancelSaveCustomSystemPreset() { const b = chatEl("chat-preset-inline-save"); if (b && b.parentNode) b.parentNode.removeChild(b); }
function deleteCustomSystemPreset() {
  const session = getActiveChatSession();
  const target = chatState.customPresets.find((p) => p.id === session.presetId);
  if (!target) return;
  chatState.customPresets = chatState.customPresets.filter((p) => p.id !== target.id);
  session.presetId = "custom-inline"; persistCustomPresets(); persistChatSessions(); renderChatSystemDrawer(); notifyChat(`Deleted preset "${target.name}".`);
}
async function saveSystemPromptToServerDefault() {
  const promptText = getActiveChatSession().systemPrompt || "";
  try {
    let base = typeof loadedSettingsObj !== "undefined" && loadedSettingsObj ? Object.assign({}, loadedSettingsObj) : null;
    if (!base) { const r = await fetch(`${API_BASE}/v1/settings`); if (!r.ok) throw new Error(`HTTP ${r.status}`); base = await r.json(); }
    base.systemPrompt = promptText;
    const postRes = await fetch(`${API_BASE}/v1/settings`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(base) });
    const data = await postRes.json().catch(() => ({}));
    if (!postRes.ok) throw new Error(data.message || `HTTP ${postRes.status}`);
    if (typeof loadedSettingsObj !== "undefined") loadedSettingsObj = base;
    if (typeof renderSettingsForm === "function") renderSettingsForm(base);
    if (typeof updateSettingsDirtyState === "function") updateSettingsDirtyState();
    renderChatSystemDrawer(); notifyChat("Updated server default systemPrompt in ~/.qwg/settings.json.");
  } catch (err) { notifyChat("Failed to save server default: " + err.message, true); }
}
function renderChatParamsDrawer() {
  const p = getActiveChatSession().params;
  const setVal = (id, val) => { const el = chatEl(id); if (el) { const s = val === null || val === undefined ? "" : String(val); if (el.value !== s) el.value = s; } };
  CHAT_PARAM_INPUT_BINDINGS.forEach(([id, key]) => setVal(id, p[key]));
  setVal("chat-param-thinking", p.enableThinking || "inherit"); setVal("chat-param-effort", p.reasoningEffort || "inherit");
  setVal("chat-param-stream", p.stream === false ? "false" : "true"); setVal("chat-param-format", p.responseFormat || "text");
  setVal("chat-param-stop", p.stop || ""); setVal("chat-json-schema", p.jsonSchema || ""); setVal("chat-tools-json", p.toolsJson || "");
  setVal("chat-param-cache-prompt", p.cachePrompt === false ? "false" : "true");
  setVal("chat-param-cache-key", p.promptCacheKey || "");
  setVal("chat-param-store", p.store === false ? "false" : "true");
  setVal("chat-param-exact-spec", p.exactSpeculation || "inherit");
  setVal("chat-param-metadata", p.customMetadata || "");
  if (chatEl("chat-json-schema-wrap")) chatEl("chat-json-schema-wrap").hidden = (p.responseFormat || "text") !== "json_schema";
  validateChatToolsJson(p.toolsJson || "");
}
function validateChatToolsJson(raw) {
  const errEl = chatEl("chat-tools-error");
  const trimmed = (raw || "").trim();
  if (!trimmed) { if (errEl) { errEl.hidden = true; errEl.textContent = ""; } return { valid: true, tools: null }; }
  try {
    const parsed = JSON.parse(trimmed);
    if (!Array.isArray(parsed)) throw new Error("Tools definition must be a JSON array.");
    if (errEl) { errEl.hidden = true; errEl.textContent = ""; }
    return { valid: true, tools: parsed };
  } catch (err) {
    if (errEl) { errEl.hidden = false; errEl.textContent = "Invalid tools JSON: " + err.message; }
    return { valid: false, error: err.message };
  }
}
function updateChatParam(key, rawValue, kind) {
  const session = getActiveChatSession();
  let parsed = rawValue;
  if (kind === "float") { parsed = rawValue === "" ? null : parseFloat(rawValue); if (parsed !== null && !Number.isFinite(parsed)) return; }
  else if (kind === "int") { parsed = rawValue === "" ? null : parseInt(rawValue, 10); if (parsed !== null && !Number.isFinite(parsed)) return; }
  else if (kind === "bool") { parsed = Boolean(rawValue); }
  session.params[key] = parsed; session.updatedAt = Date.now(); persistChatSessions();
  if (key === "responseFormat" && chatEl("chat-json-schema-wrap")) chatEl("chat-json-schema-wrap").hidden = parsed !== "json_schema";
  if (key === "toolsJson") validateChatToolsJson(parsed);
}
function resetChatParamsToDefault() {
  const s = getActiveChatSession(); s.params = createDefaultChatParams(); s.updatedAt = Date.now();
  persistChatSessions(); renderChatParamsDrawer(); notifyChat("Reset per-session parameter overrides.");
}
function loadSampleChatTool() { const s = getActiveChatSession(); s.params.toolsJson = SAMPLE_CHAT_TOOLS_JSON; s.updatedAt = Date.now(); persistChatSessions(); renderChatParamsDrawer(); }
function clearChatToolsJson() { const s = getActiveChatSession(); s.params.toolsJson = ""; s.updatedAt = Date.now(); persistChatSessions(); renderChatParamsDrawer(); }
function isChatMsgActivelyReasoning(msg) {
  return Boolean(msg && msg.isStreaming && msg.reasoning && !msg.content && !(Array.isArray(msg.toolCalls) && msg.toolCalls.length));
}
function isChatReasoningOpen(msgId) { return Boolean(msgId && chatState.reasoningOpenById[msgId] === true); }
function handleChatThinkingToggle(e) {
  const el = e && e.target;
  if (!el || el.tagName !== "DETAILS") return;
  const msgId = typeof el.getAttribute === "function" ? el.getAttribute("data-thinking-msg-id") : "";
  if (msgId) chatState.reasoningOpenById[msgId] = Boolean(el.open);
}
function stopChatReasoningQuip() {
  if (chatState.reasoningQuip.timerId && typeof clearInterval === "function") clearInterval(chatState.reasoningQuip.timerId);
  chatState.reasoningQuip.msgId = null; chatState.reasoningQuip.text = ""; chatState.reasoningQuip.timerId = null;
}
function ensureChatReasoningQuip(msgId) {
  const q = chatState.reasoningQuip;
  if (q.msgId !== msgId || !q.text) {
    const pick = pickRandomReasoningQuip(q.index);
    q.msgId = msgId; q.index = pick.index; q.text = pick.text;
  }
  if (!q.timerId && typeof setInterval === "function") {
    q.timerId = setInterval(() => {
      const art = chatEl(`chat-turn-${q.msgId}`);
      const det = art && typeof art.querySelector === "function" ? art.querySelector("details.md-thinking") : null;
      if (!det || !det.classList || !det.classList.contains("is-reasoning")) { stopChatReasoningQuip(); return; }
      const pick = pickRandomReasoningQuip(q.index);
      q.index = pick.index; q.text = pick.text;
      const quipEl = det.querySelector(".md-thinking-quip");
      if (quipEl) quipEl.textContent = ` · ${q.text}`;
    }, CHAT_REASONING_QUIP_INTERVAL_MS);
  }
  return q.text;
}
function buildAnsweredToolCallIdSet(msgs) {
  const answeredIds = new Set();
  (msgs || []).forEach((m) => { if (m.role === "tool" && m.toolCallId) answeredIds.add(m.toolCallId); });
  return answeredIds;
}
function buildChatThinkingHtml(msg) {
  const active = isChatMsgActivelyReasoning(msg);
  const openAttr = isChatReasoningOpen(msg.id) ? " open" : "";
  const quipText = active ? ensureChatReasoningQuip(msg.id) : "";
  const quipHtml = quipText ? `<span class="md-thinking-quip"> · ${escapeHtml(quipText)}</span>` : "";
  return `<details class="md-thinking${active ? " is-reasoning" : ""}" data-thinking-msg-id="${escapeHtml(msg.id)}"${openAttr}><summary class="md-thinking-summary"><span class="md-thinking-title">${active ? "Reasoning" : "Reasoning trace"}</span> <span class="md-thinking-meta mono">(${msg.reasoning.length} chars)</span>${quipHtml}</summary><pre class="md-thinking-pre">${escapeHtml(msg.reasoning)}</pre></details>`;
}
function buildAssistantAnswerHtml(msg) {
  if (msg.isRaw) return `<pre class="probe-raw mono">${escapeHtml(msg.content || "")}</pre>`;
  if (msg.content) return `<div class="probe-markdown">${renderSmokeProbeMarkdown(msg.content)}</div>`;
  if (!(msg.toolCalls && msg.toolCalls.length)) {
    if (msg.isStreaming && !msg.reasoning) return `<div class="panel-sub mono">Streaming…</div>`;
    if (!msg.isStreaming) return `<div class="panel-sub mono">—</div>`;
  }
  return "";
}
function renderChatTranscript() {
  const host = chatEl("chat-transcript");
  if (!host) return;
  const keepTop = !chatState.autoFollowScroll ? (host.scrollTop || 0) : 0;
  const msgs = getActiveChatSession().messages;
  if (!msgs.length) {
    host.innerHTML = `<div class="chat-empty-state"><h3>Local Agent &amp; Chat Workbench</h3>
      <p>Multi-turn sessions stay in your browser. System instructions condition Warm RAM prefix reuse cleanly across turns.</p>
      <div class="chat-starter-row">
        <button type="button" class="btn btn-sm" onclick="applyChatStarterPrompt('Explain the trade-offs of KV cache quantization in local unified-memory inference.')">Explain KV cache quant</button>
        <button type="button" class="btn btn-sm" onclick="applyChatStarterPrompt('Write a clean Swift function with unit tests to parse NDJSON safely.')">Swift NDJSON helper</button>
        <button type="button" class="btn btn-sm" onclick="toggleChatDrawer('system')">Choose System Prompt</button>
      </div></div>`;
    updateChatScrollToBottomButton();
    return;
  }
  const answeredIds = buildAnsweredToolCallIdSet(msgs);
  host.innerHTML = msgs.map((m, idx) => renderSingleChatTurnHtml(m, idx, msgs.length, answeredIds)).join("");
  if (!chatState.autoFollowScroll && keepTop > 0) {
    chatState.ignoreScrollUntil = Date.now() + 80;
    host.scrollTop = keepTop;
    chatState.lastTranscriptScrollTop = keepTop;
  }
  updateChatScrollToBottomButton();
}
function renderSingleChatTurnHtml(msg, idx, totalCount, answeredToolCallIds) {
  const session = getActiveChatSession();
  const isAssistant = msg.role === "assistant"; const isUser = msg.role === "user"; const isTool = msg.role === "tool";
  const isJsonMode = session.presetId === "json-extractor" || (session.params && session.params.responseFormat !== "text");
  const roleLabel = isUser ? "You" : (isTool ? `Tool (${msg.toolName || msg.toolCallId || "output"})` : "Assistant");
  const btns = [];
  if (isUser) btns.push(`<button type="button" class="chat-turn-btn" data-chat-action="edit-turn" data-msg-id="${escapeHtml(msg.id)}">Edit &amp; Fork</button>`);
  if (isAssistant) {
    btns.push(`<button type="button" class="chat-turn-btn" data-chat-action="regen-turn" data-msg-id="${escapeHtml(msg.id)}">Regenerate</button>`);
    if (idx === totalCount - 1 && !chatState.isGenerating) btns.push(`<button type="button" class="chat-turn-btn" data-chat-action="continue-turn" data-msg-id="${escapeHtml(msg.id)}">Continue</button>`);
    if (!msg.isStreaming && msg.content && isJsonMode) btns.push(`<button type="button" class="chat-turn-btn" data-chat-action="format-json-turn" data-msg-id="${escapeHtml(msg.id)}">Format JSON</button>`);
    btns.push(`<button type="button" class="chat-turn-btn" data-chat-action="toggle-raw-turn" data-msg-id="${escapeHtml(msg.id)}">${msg.isRaw ? "Rendered" : "Raw"}</button>`);
  }
  if (!msg.isStreaming && msg.content) btns.push(`<button type="button" class="chat-turn-btn" data-chat-action="send-to-decisions" data-msg-id="${escapeHtml(msg.id)}">Decisions →</button>`);
  btns.push(`<button type="button" class="chat-turn-btn" data-chat-action="copy-turn" data-msg-id="${escapeHtml(msg.id)}">Copy</button>`);
  btns.push(`<button type="button" class="chat-turn-btn" data-chat-action="delete-turn" data-msg-id="${escapeHtml(msg.id)}" aria-label="Delete turn">×</button>`);

  let jsonBadgeHtml = "";
  if (isAssistant && !msg.isStreaming && msg.content && isJsonMode && typeof validateJsonTurnContent === "function") {
    const jv = validateJsonTurnContent(msg.content);
    jsonBadgeHtml = `<span class="knob-pill chat-json-badge ${jv.valid ? "is-valid" : "is-invalid"}" title="${escapeHtml(jv.valid ? "Strict JSON verified" : (jv.reason || "Invalid JSON"))}">${jv.valid ? "JSON Valid" : "JSON Invalid"}</span>`;
  }
  const routeBadgeHtml = isAssistant && msg.decisionRoute
    ? `<span class="knob-pill mono chat-route-badge" title="Auto reasoning effort tier routed by Decision Sidecar">${escapeHtml(msg.decisionRoute)}</span>`
    : "";

  let bodyHtml = "";
  if (chatState.editingMsgId === msg.id) {
    bodyHtml = `<div class="chat-edit-wrap"><textarea id="chat-edit-textarea-${escapeHtml(msg.id)}" rows="3" class="input chat-textarea mono">${escapeHtml(msg.content || "")}</textarea>
      <div class="chat-edit-actions"><button type="button" class="btn btn-primary btn-sm" data-chat-action="save-edit-turn" data-msg-id="${escapeHtml(msg.id)}">Save &amp; Resend</button>
      <button type="button" class="btn btn-sm" data-chat-action="cancel-edit-turn">Cancel</button></div></div>`;
  } else {
    if (isAssistant && msg.reasoning) bodyHtml += buildChatThinkingHtml(msg);
    if (isAssistant) bodyHtml += `<div class="chat-turn-answer">${buildAssistantAnswerHtml(msg)}</div>`;
    else bodyHtml += `<div class="chat-user-text">${escapeHtml(msg.content || "")}</div>`;
    if (Array.isArray(msg.images) && msg.images.length > 0 && typeof safeChatImageDataUrl === "function") {
      bodyHtml += `<div class="chat-turn-images">${msg.images.map((img) => `<img class="chat-turn-thumb" src="${escapeHtml(safeChatImageDataUrl(img.dataUrl))}" alt="${escapeHtml(img.name || "Attached image")}" />`).join("")}</div>`;
    }
    if (Array.isArray(msg.videos) && msg.videos.length > 0 && typeof safeChatVideoDataUrl === "function") {
      bodyHtml += `<div class="chat-turn-images">${msg.videos.map((vid) => `<video class="chat-turn-thumb" controls muted playsinline preload="metadata" src="${escapeHtml(safeChatVideoDataUrl(vid.dataUrl))}"></video>`).join("")}</div>`;
    }
    if (Array.isArray(msg.toolCalls) && msg.toolCalls.length > 0) {
      bodyHtml += `<div class="chat-turn-tools">${msg.toolCalls.map((tc) => renderToolCallCardHtml(msg.id, tc, answeredToolCallIds)).join("")}</div>`;
    }
    if (msg.error) bodyHtml += `<div class="chat-turn-error">${escapeHtml(msg.error)}</div>`;
  }
  const roleClass = isUser ? "chat-turn-user" : (isTool ? "chat-turn-tool" : "chat-turn-assistant");
  return `<article id="chat-turn-${escapeHtml(msg.id)}" class="chat-turn ${roleClass}">
    <div class="chat-turn-head"><div class="chat-turn-role"><span>${escapeHtml(roleLabel)}</span>${jsonBadgeHtml}${routeBadgeHtml}${formatChatTurnMetrics(msg)}</div><div class="chat-turn-actions">${btns.join("")}</div></div>
    <div class="chat-turn-body">${bodyHtml}</div></article>`;
}
function updateStreamingChatTurn(msg, answeredToolCallIds) {
  const article = msg ? chatEl(`chat-turn-${msg.id}`) : null;
  if (!article || typeof article.querySelector !== "function" || typeof document === "undefined" || typeof document.createElement !== "function") {
    renderChatTranscript();
    return;
  }
  const roleRow = article.querySelector(".chat-turn-role");
  if (roleRow) {
    const metricsEl = roleRow.querySelector(".chat-turn-metrics");
    const nextMetricsHtml = formatChatTurnMetrics(msg);
    if (nextMetricsHtml) {
      const wrap = document.createElement("div"); wrap.innerHTML = nextMetricsHtml;
      const fresh = wrap.firstElementChild;
      if (metricsEl && fresh) metricsEl.textContent = fresh.textContent;
      else if (fresh) roleRow.appendChild(fresh);
    } else if (metricsEl) {
      metricsEl.remove();
    }
  }
  const body = article.querySelector(".chat-turn-body");
  if (!body) { renderChatTranscript(); return; }
  const active = isChatMsgActivelyReasoning(msg);
  let det = body.querySelector("details.md-thinking");
  if (msg.reasoning) {
    if (!det) {
      const wrap = document.createElement("div"); wrap.innerHTML = buildChatThinkingHtml(msg);
      det = wrap.firstElementChild; if (det) body.insertBefore(det, body.firstChild);
    } else {
      det.classList.toggle("is-reasoning", active);
      const titleEl = det.querySelector(".md-thinking-title"); if (titleEl) titleEl.textContent = active ? "Reasoning" : "Reasoning trace";
      const metaEl = det.querySelector(".md-thinking-meta"); if (metaEl) metaEl.textContent = `(${msg.reasoning.length} chars)`;
      const summaryEl = det.querySelector(".md-thinking-summary");
      let quipEl = det.querySelector(".md-thinking-quip");
      if (active) {
        const quipText = ` · ${ensureChatReasoningQuip(msg.id)}`;
        if (!quipEl && summaryEl) { quipEl = document.createElement("span"); quipEl.className = "md-thinking-quip"; summaryEl.appendChild(quipEl); }
        if (quipEl) { quipEl.hidden = false; quipEl.textContent = quipText; }
      } else {
        stopChatReasoningQuip();
        if (quipEl) quipEl.remove();
      }
      const preEl = det.querySelector(".md-thinking-pre"); if (preEl) preEl.textContent = msg.reasoning;
    }
  } else if (!active && chatState.reasoningQuip.msgId === msg.id) {
    stopChatReasoningQuip();
  }
  const answerEl = body.querySelector(".chat-turn-answer");
  const nextAnswerHtml = buildAssistantAnswerHtml(msg);
  if (answerEl) { if (answerEl.innerHTML !== nextAnswerHtml) answerEl.innerHTML = nextAnswerHtml; }
  else { renderChatTranscript(); return; }
  const answeredIds = answeredToolCallIds || buildAnsweredToolCallIdSet(getActiveChatSession().messages);
  let toolsEl = body.querySelector(".chat-turn-tools");
  if (Array.isArray(msg.toolCalls) && msg.toolCalls.length > 0) {
    const nextToolsHtml = msg.toolCalls.map((tc) => renderToolCallCardHtml(msg.id, tc, answeredIds)).join("");
    if (!toolsEl) { toolsEl = document.createElement("div"); toolsEl.className = "chat-turn-tools"; body.insertBefore(toolsEl, body.querySelector(".chat-turn-error") || null); }
    toolsEl.innerHTML = nextToolsHtml;
  } else if (toolsEl) {
    toolsEl.remove();
  }
}
function isChatTranscriptNearBottom(el, thresholdPx) { return isTranscriptNearBottom(el, thresholdPx); }
function updateChatScrollToBottomButton() {
  const btn = chatEl("chat-scroll-bottom-btn");
  const h = chatEl("chat-transcript");
  if (!btn) return;
  const canScroll = Boolean(h && h.scrollHeight > (h.clientHeight || 0) + 4);
  const nearBottom = h ? isChatTranscriptNearBottom(h, 48) : true;
  btn.hidden = !canScroll || chatState.autoFollowScroll || nearBottom;
}
function handleChatTranscriptWheel(e) {
  if (e && typeof e.deltaY === "number" && e.deltaY < 0) {
    chatState.autoFollowScroll = false;
    updateChatScrollToBottomButton();
  }
}
function handleChatTranscriptScroll() {
  const h = chatEl("chat-transcript");
  if (!h) return;
  const curTop = h.scrollTop;
  const nearBottom = isChatTranscriptNearBottom(h, 48);
  if (!nearBottom && (curTop < chatState.lastTranscriptScrollTop - 0.5 || Date.now() >= chatState.ignoreScrollUntil)) {
    chatState.autoFollowScroll = false;
  } else if (nearBottom && Date.now() >= chatState.ignoreScrollUntil) {
    chatState.autoFollowScroll = true;
  }
  chatState.lastTranscriptScrollTop = curTop;
  updateChatScrollToBottomButton();
}
function scrollChatTranscriptToBottom(force) {
  if (force === true) chatState.autoFollowScroll = true;
  const h = chatEl("chat-transcript");
  if (!chatState.autoFollowScroll || !h) { updateChatScrollToBottomButton(); return; }
  chatState.ignoreScrollUntil = Date.now() + 80;
  h.scrollTop = h.scrollHeight;
  chatState.lastTranscriptScrollTop = h.scrollTop;
  updateChatScrollToBottomButton();
}
function bindChatTranscriptListeners() {
  if (chatState.transcriptListenersBound || typeof document === "undefined") return;
  const host = chatEl("chat-transcript");
  if (!host) return;
  chatState.transcriptListenersBound = true;
  if (typeof document.addEventListener === "function") document.addEventListener("toggle", handleChatThinkingToggle, true);
  if (typeof host.addEventListener === "function") {
    host.addEventListener("scroll", handleChatTranscriptScroll, { passive: true });
    host.addEventListener("wheel", handleChatTranscriptWheel, { passive: true });
  }
}
function applyChatStarterPrompt(text) { const i = chatEl("chat-input"); if (i) { i.value = text; autoResizeChatInput(i); i.focus(); } }
function autoResizeChatInput(el) { if (el) { el.style.height = "auto"; el.style.height = `${Math.min(Math.max(el.scrollHeight, 68), 220)}px`; } }
function handleChatInputKeyDown(e) { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); if (!chatState.isGenerating) sendChatTurn(); } }
function renderChatComposerState() {
  if (chatEl("chat-send-btn")) chatEl("chat-send-btn").disabled = chatState.isGenerating;
  if (chatEl("chat-stop-btn")) chatEl("chat-stop-btn").hidden = !chatState.isGenerating;
  if (chatEl("chat-composer-status")) chatEl("chat-composer-status").textContent = chatState.activeStatusLabel || "Ready";
}
function stopActiveChatGeneration() {
  stopChatReasoningQuip();
  if (chatState.activeAbortController) { chatState.activeAbortController.abort(); chatState.activeAbortController = null; }
}
async function sendChatTurn() {
  if (chatState.isGenerating) return;
  const inputEl = chatEl("chat-input"); const text = inputEl ? inputEl.value.trim() : "";
  const images = (chatState.pendingImages || []).slice(); const videos = (chatState.pendingVideos || []).slice();
  if (!text && images.length === 0 && videos.length === 0) return;
  const session = getActiveChatSession();
  if (session.messages.length === 0 && (session.title === "New Chat" || session.title === "Untitled Chat") && text) session.title = text.replace(/\s+/g, " ").slice(0, 42);
  session.messages.push({ id: generateChatId("msg"), role: "user", content: text, images, videos, createdAt: Date.now() });
  if (inputEl) { inputEl.value = ""; autoResizeChatInput(inputEl); }
  chatState.pendingImages = []; chatState.pendingVideos = []; commitChatWorkspace(session); scrollChatTranscriptToBottom(true); await executeChatCompletion(session, {});
}
function beginEditChatTurn(msgId) {
  if (chatState.isGenerating) return;
  chatState.editingMsgId = msgId; renderChatTranscript();
  if (chatEl(`chat-edit-textarea-${msgId}`)) chatEl(`chat-edit-textarea-${msgId}`).focus();
}
function cancelEditChatTurn() { chatState.editingMsgId = null; renderChatTranscript(); }
async function saveAndResendChatTurn(msgId) {
  const session = getActiveChatSession(); const idx = session.messages.findIndex((m) => m.id === msgId);
  if (idx === -1) return;
  const area = chatEl(`chat-edit-textarea-${msgId}`); const nextText = area ? area.value.trim() : "";
  const hasTurnMedia = (session.messages[idx].images && session.messages[idx].images.length) || (session.messages[idx].videos && session.messages[idx].videos.length);
  if (!nextText && !hasTurnMedia) { notifyChat("Message text cannot be empty.", true); return; }
  session.messages[idx].content = nextText; session.messages = session.messages.slice(0, idx + 1); chatState.editingMsgId = null;
  commitChatWorkspace(session); scrollChatTranscriptToBottom(true); await executeChatCompletion(session, {});
}
async function regenerateChatTurn(msgId) {
  if (chatState.isGenerating) return;
  const session = getActiveChatSession(); const idx = session.messages.findIndex((m) => m.id === msgId);
  if (idx <= 0) return;
  session.messages = session.messages.slice(0, idx); commitChatWorkspace(session); await executeChatCompletion(session, {});
}
async function continueChatAssistantTurn(msgId) {
  if (chatState.isGenerating) return;
  const session = getActiveChatSession(); const idx = session.messages.findIndex((m) => m.id === msgId);
  if (idx === -1 || idx !== session.messages.length - 1) return;
  await executeChatCompletion(session, { continueLastAssistant: true, targetAssistantMsg: session.messages[idx] });
}
function toggleChatTurnRaw(msgId) {
  const msg = getActiveChatSession().messages.find((m) => m.id === msgId);
  if (msg) { msg.isRaw = !msg.isRaw; persistChatSessions(); renderChatTranscript(); }
}
function copyChatTurnContent(msgId) {
  const msg = getActiveChatSession().messages.find((m) => m.id === msgId);
  if (!msg) return;
  const text = msg.content || (msg.toolCalls ? JSON.stringify(msg.toolCalls, null, 2) : "");
  if (typeof navigator !== "undefined" && navigator.clipboard && navigator.clipboard.writeText) {
    navigator.clipboard.writeText(text).then(() => notifyChat("Turn copied.")).catch(() => notifyChat("Clipboard unavailable.", true));
  }
}
function deleteChatTurn(msgId) {
  if (chatState.isGenerating) return;
  const s = getActiveChatSession(); s.messages = s.messages.filter((m) => m.id !== msgId); commitChatWorkspace(s);
}
async function submitToolResultForCall(msgId, toolCallId, toolName, runContinue) {
  if (chatState.isGenerating) return;
  const input = chatEl(`tool-out-${toolCallId}`); const session = getActiveChatSession();
  const idx = session.messages.findIndex((m) => m.id === msgId);
  const toolMsg = { id: generateChatId("msg"), role: "tool", toolCallId, toolName, content: (input ? input.value : "") || "{}", createdAt: Date.now() };
  if (idx !== -1) session.messages.splice(idx + 1, 0, toolMsg); else session.messages.push(toolMsg);
  commitChatWorkspace(session); scrollChatTranscriptToBottom(true); if (runContinue) await executeChatCompletion(session, {});
}
async function readChatSSEStream(res, assistantMsg, startedAt, acc) {
  const reader = res.body.getReader(); const decoder = new TextDecoder(); let buffer = ""; let sawDone = false;
  while (!sawDone) {
    const chunk = await reader.read(); if (chunk.done) break;
    buffer += decoder.decode(chunk.value, { stream: true });
    const lines = buffer.split("\n"); buffer = lines.pop() || "";
    for (let i = 0; i < lines.length; i += 1) {
      const parsed = parseChatSSELine(lines[i]); if (!parsed) continue;
      if (parsed.done) { sawDone = true; break; } if (parsed.error) throw new Error(parsed.error);
      if (parsed.data && parsed.data.usage) assistantMsg.usage = parsed.data.usage;
      const delta = parsed.data && Array.isArray(parsed.data.choices) && parsed.data.choices[0] ? parsed.data.choices[0].delta : null;
      if (delta) {
        if (typeof delta.reasoning_content === "string") acc.reasoning += delta.reasoning_content;
        if (typeof delta.content === "string") acc.content += delta.content;
        if (Array.isArray(delta.tool_calls)) acc.toolCalls = mergeToolCallChunks(acc.toolCalls, delta.tool_calls);
      }
    }
    const split = splitInlineThinkTags(acc.content, acc.reasoning, true);
    assistantMsg.content = split.answer; assistantMsg.reasoning = split.reasoning; assistantMsg.toolCalls = acc.toolCalls;
    assistantMsg.elapsedMs = Math.round(performance.now() - startedAt);
    updateStreamingChatTurn(assistantMsg); scrollChatTranscriptToBottom(false);
  }
  const finalSplit = splitInlineThinkTags(acc.content, acc.reasoning, false);
  assistantMsg.content = finalSplit.answer; assistantMsg.reasoning = finalSplit.reasoning;
}
async function executeChatCompletion(session, opts) {
  let payload;
  try { payload = buildChatCompletionPayload(session, { continueLastAssistant: Boolean(opts.continueLastAssistant) }); }
  catch (err) { notifyChat("Invalid request parameters: " + err.message, true); return; }
  const controller = typeof AbortController !== "undefined" ? new AbortController() : null;
  chatState.isGenerating = true; chatState.activeAbortController = controller; chatState.activeStatusLabel = payload.stream ? "Streaming…" : "Generating…";
  const isContinue = Boolean(opts.continueLastAssistant && opts.targetAssistantMsg);
  const basePrefix = isContinue ? (opts.targetAssistantMsg.content || "") : "";
  const assistantMsg = isContinue ? opts.targetAssistantMsg : {
    id: generateChatId("msg"), role: "assistant", content: "", reasoning: "", toolCalls: [], usage: null, isStreaming: true, isRaw: false, createdAt: Date.now()
  };
  if (!isContinue) session.messages.push(assistantMsg); else { assistantMsg.isStreaming = true; assistantMsg.error = ""; }
  renderChatTranscript(); renderChatComposerState(); scrollChatTranscriptToBottom(true);
  const startedAt = performance.now();
  const acc = { content: basePrefix, reasoning: isContinue ? (assistantMsg.reasoning || "") : "", toolCalls: isContinue ? (assistantMsg.toolCalls || []) : [] };
  const reqId = generateChatId("req");
  const reqHeaders = { "Content-Type": "application/json", "X-Request-ID": reqId, "X-Qwegn-Client": "webui" };
  if (session && session.id) reqHeaders["X-Session-ID"] = session.id;
  try {
    const res = await fetch(`${API_BASE}/v1/chat/completions`, {
      method: "POST", headers: reqHeaders, body: JSON.stringify(payload), signal: controller ? controller.signal : undefined
    });
    const echoedReqId = res.headers && typeof res.headers.get === "function" ? res.headers.get("x-request-id") : "";
    const routeHdr = res.headers && typeof res.headers.get === "function" ? res.headers.get("x-qwegn-decision-route") : "";
    assistantMsg.requestId = echoedReqId || reqId;
    if (routeHdr) assistantMsg.decisionRoute = routeHdr;
    if (!res.ok) { const errBody = await res.json().catch(() => ({})); throw new Error((errBody.error && errBody.error.message) || errBody.message || `HTTP ${res.status}`); }
    if (payload.stream && res.body && typeof res.body.getReader === "function") await readChatSSEStream(res, assistantMsg, startedAt, acc);
    else {
      const json = await res.json(); if (json.usage) assistantMsg.usage = json.usage;
      const msg = Array.isArray(json.choices) && json.choices[0] ? json.choices[0].message : null;
      const split = splitInlineThinkTags(basePrefix + (msg && typeof msg.content === "string" ? msg.content : ""), acc.reasoning + (msg && typeof msg.reasoning_content === "string" ? msg.reasoning_content : ""), false);
      assistantMsg.content = split.answer; assistantMsg.reasoning = split.reasoning;
      if (msg && Array.isArray(msg.tool_calls)) assistantMsg.toolCalls = msg.tool_calls;
    }
    assistantMsg.elapsedMs = Math.round(performance.now() - startedAt);
    chatState.activeStatusLabel = `browser round trip: ${assistantMsg.elapsedMs} ms`;
    if (typeof fetchDashboard === "function") fetchDashboard();
  } catch (err) {
    assistantMsg.elapsedMs = Math.round(performance.now() - startedAt);
    if (err && err.name === "AbortError") chatState.activeStatusLabel = "Cancelled by user";
    else { assistantMsg.error = err ? err.message : "Request failed"; chatState.activeStatusLabel = "browser round trip: failed"; notifyChat("Chat request failed: " + assistantMsg.error, true); }
  } finally {
    stopChatReasoningQuip();
    assistantMsg.isStreaming = false; chatState.isGenerating = false; chatState.activeAbortController = null;
    commitChatWorkspace(session); scrollChatTranscriptToBottom(false);
  }
}

function handleChatActionClick(e) {
  const el = e.target && e.target.closest ? e.target.closest("[data-chat-action]") : null;
  if (!el) return;
  const action = el.getAttribute("data-chat-action");
  const id = el.getAttribute("data-msg-id") || el.getAttribute("data-session-id") || el.getAttribute("data-img-id") || el.getAttribute("data-vid-id") || "";
  const map = {
    "select-session": () => selectChatSession(id),
    "delete-session": () => deleteChatSession(id),
    "edit-turn": () => beginEditChatTurn(id),
    "regen-turn": () => regenerateChatTurn(id),
    "continue-turn": () => continueChatAssistantTurn(id),
    "format-json-turn": () => formatJsonChatTurn(id),
    "toggle-raw-turn": () => toggleChatTurnRaw(id),
    "copy-turn": () => copyChatTurnContent(id),
    "send-to-decisions": () => {
      const m = getActiveChatSession().messages.find((x) => x.id === id);
      if (m && typeof openDecisionsWithPrefill === "function") {
        openDecisionsWithPrefill(m.content || "", (m.images || []).map((i) => i.dataUrl).filter(Boolean));
      }
    },
    "delete-turn": () => deleteChatTurn(id),
    "save-edit-turn": () => saveAndResendChatTurn(id),
    "cancel-edit-turn": () => cancelEditChatTurn(),
    "remove-image": () => { if (typeof removePendingChatImage === "function") removePendingChatImage(id); },
    "remove-video": () => { if (typeof removePendingChatVideo === "function") removePendingChatVideo(id); },
    "submit-tool-result": () => submitToolResultForCall(id, el.getAttribute("data-call-id") || "", el.getAttribute("data-tool-name") || "tool", el.getAttribute("data-run-continue") === "true")
  };
  if (map[action]) map[action]();
}
if (typeof document !== "undefined") {
  document.addEventListener("DOMContentLoaded", () => {
    document.addEventListener("click", handleChatActionClick);
    initChatWorkspace();
  });
}
