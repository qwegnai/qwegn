async function loadDiscoveredModels() {
  try {
    const res = await fetch(`${API_BASE}/v1/models/discovered`);
    if (!res.ok) throw new Error("HTTP " + res.status);
    const data = await res.json();
    discoveredModelsList = Array.isArray(data.models) ? data.models : [];
    discoveredDraftersList = Array.isArray(data.drafters) ? data.drafters : [];
    renderModelsCards();
  } catch (err) {
    showToast("Failed to list models: " + err.message, true);
  }
}

function renderModelsCards() {
  const container = document.getElementById("models-cards-grid");
  if (!container) return;
  const supported = discoveredModelsList.filter((m) => m.supported !== false);
  if (supported.length === 0) {
    container.innerHTML = `<div class="panel-sub" style="padding:24px;text-align:center;">No supported models found in <code>~/.qwg/models</code>.</div>`;
    return;
  }
  const activeName = publishedModel ? publishedModel.modelId : null;
  container.innerHTML = supported.map((m) => {
    const isActive = activeName && (m.name === activeName);
    const rawCaps = Array.isArray(m.capabilities) ? m.capabilities.slice() : [];
    const badges = rawCaps.slice();
    if (m.isDecisionCapable && !badges.some((b) => String(b).toLowerCase() === "decision")) badges.push("Decision");
    if (m.hasVision && !badges.some((b) => String(b).toLowerCase() === "vision")) badges.push("Vision");
    if (m.isCoreAIBundle && !badges.some((b) => String(b).toLowerCase() === "coreai")) badges.push("CoreAI");
    const caps = badges.map((c) => `<span class="knob-pill mono">${escapeHtml(c)}</span>`).join("");
    const modelName = String(m.name || "");
    const fullPath = String(m.packPath || "");
    const loadSidecarBtn = m.isDecisionCapable
      ? `<button type="button" class="btn btn-sm" style="white-space:nowrap;" data-load-decision-path="${escapeHtml(fullPath)}" data-load-decision-vision="${m.hasVision ? "true" : "false"}">Load Sidecar</button>`
      : "";
    return `
      <div class="model-card">
        <div style="display:flex;flex-direction:column;gap:10px;min-width:0;">
          <div class="model-card-head">
            <div style="min-width:0;flex:1;">
              <div class="model-card-title">${escapeHtml(m.name)}</div>
              <div class="model-card-path mono" title="${escapeHtml(fullPath)}">${escapeHtml(middleTruncate(fullPath, 46))}</div>
            </div>
            <span class="knob-pill mono ${isActive ? "ink-secondary" : ""}" style="width:auto;flex-shrink:0;border-radius:999px;">${isActive ? "Resident" : "Installed"}</span>
          </div>
          ${caps ? `<div class="model-card-caps">${caps}</div>` : ""}
        </div>
        <div class="model-card-foot" style="gap:6px;flex-wrap:wrap;">
          ${loadSidecarBtn}
          <button type="button" class="btn ${isActive ? "" : "btn-primary"} btn-sm" style="white-space:nowrap;" data-select-model-name="${escapeHtml(modelName)}" data-select-model-path="${escapeHtml(fullPath)}">
            ${isActive ? "Reload Model" : "Activate"}
          </button>
        </div>
      </div>`;
  }).join("");
}

async function selectAndApplyModel(name, packPath) {
  try {
    const res = await fetch(`${API_BASE}/v1/settings`);
    if (!res.ok) throw new Error(`settings read HTTP ${res.status}`);
    const currentSettings = await res.json();
    currentSettings.modelName = name;
    currentSettings.modelPackPath = packPath;
    const saveRes = await fetch(`${API_BASE}/v1/settings`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(currentSettings)
    });
    if (!saveRes.ok) {
      const errData = await saveRes.json().catch(() => ({}));
      throw new Error(errData.message || `HTTP ${saveRes.status}`);
    }
    const loadRes = await fetch(`${API_BASE}/v1/models/load`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model: packPath || name })
    });
    if (!loadRes.ok) {
      const errData = await loadRes.json().catch(() => ({}));
      throw new Error(errData.message || `load HTTP ${loadRes.status}`);
    }
    showToast(`Switched active model to ${name}.`);
    fetchDashboard();
    fetchEngineInfo();
    if (document.getElementById("settings-form-host")) loadLiveSettings();
  } catch (e) {
    showToast("Failed to switch model: " + e.message, true);
  }
}

function openSwitchModelModal() {
  const dialog = document.getElementById("modal-switch-model");
  const list = document.getElementById("modal-model-options");
  if (!dialog || !list) return;
  loadDiscoveredModels().then(() => {
    const supported = discoveredModelsList.filter((m) => m.supported !== false);
    list.innerHTML = supported.map((m) => {
      const modelName = String(m.name || "");
      const packPath = String(m.packPath || "");
      return `
        <button type="button" class="btn" style="justify-content:space-between;width:100%;" data-select-model-name="${escapeHtml(modelName)}" data-select-model-path="${escapeHtml(packPath)}" data-close-switch-modal="true">
          <span>${escapeHtml(m.name)}</span>
          <span class="ink-primary mono">Select</span>
        </button>`;
    }).join("");
    if (typeof dialog.showModal === "function") dialog.showModal();
  });
}

function closeSwitchModelModal() {
  const dialog = document.getElementById("modal-switch-model");
  if (dialog && dialog.open && typeof dialog.close === "function") dialog.close();
}

async function unloadModel() {
  if (!confirm("Unload active model weights from unified memory? The HTTP server will remain available.")) return;
  try {
    const res = await fetch(`${API_BASE}/v1/models/unload`, {
      method: "POST",
      headers: { "Content-Type": "application/json" }
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    const reclaimed = typeof data.memoryReclaimedMB === "number"
      ? ` Unified memory reclaimed: ${data.memoryReclaimedMB} MB.` : "";
    showToast(`Model unloaded.${reclaimed}`);
  } catch (e) {
    showToast("Unload failed: " + e.message, true);
  }
}

async function reloadOrLoadModel() {
  try {
    const res = await fetch(`${API_BASE}/v1/models/load`, {
      method: "POST",
      headers: { "Content-Type": "application/json" }
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    showToast("Model load triggered.");
  } catch (e) {
    showToast("Load failed: " + e.message, true);
  }
}

async function restartEngine() {
  if (!confirm("Restart the resident model? Active requests will be interrupted.")) return;
  try {
    const res = await fetch(`${API_BASE}/v1/engine/restart`, {
      method: "POST",
      headers: { "Content-Type": "application/json" }
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    showToast("Model restart requested.");
  } catch (e) {
    showToast("Restart failed: " + e.message, true);
  }
}

async function enablePrefixCacheAction() {
  try {
    const res = await fetch(`${API_BASE}/v1/settings`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const s = await res.json();
    s.warmRAMLimitGB = s.warmRAMLimitGB > 0 ? s.warmRAMLimitGB : 4.0;
    const saveRes = await fetch(`${API_BASE}/v1/settings`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(s)
    });
    if (!saveRes.ok) throw new Error(`HTTP ${saveRes.status}`);
    showToast("Saved prefix cache configuration.");
  } catch (e) {
    showToast("Enable prefix cache failed: " + e.message, true);
  }
}

// --- LIVE LOGS TAB (/v1/logs/stream) ---
function initLogStream() {
  if (logsEventSource) return;
  const badge = document.getElementById("logs-stream-status");
  if (badge) badge.textContent = "Connecting…";
  logsEventSource = new EventSource(`${API_BASE}/v1/logs/stream`);
  logsEventSource.onopen = () => {
    if (badge) badge.textContent = "SSE Connected";
  };
  logsEventSource.onmessage = (event) => {
    try {
      const payload = JSON.parse(event.data);
      const line = String(payload.line || "").trimEnd();
      if (!line) return;
      if (logBuffer.slice(-16).includes(line)) return;
      logBuffer.push(line);
      if (logBuffer.length > 2000) logBuffer.shift();
      renderLogsView();
    } catch (err) {
      console.warn("log parse error:", err);
    }
  };
  logsEventSource.onerror = () => {
    if (badge) badge.textContent = "Reconnecting…";
  };
}

function closeLogStream() {
  if (!logsEventSource) return;
  logsEventSource.close();
  logsEventSource = null;
  const badge = document.getElementById("logs-stream-status");
  if (badge) badge.textContent = "Paused";
}

function setLogFilter(filterName) {
  activeLogFilter = filterName;
  document.querySelectorAll("[data-log-filter]").forEach((btn) => {
    btn.classList.toggle("is-active", btn.getAttribute("data-log-filter") === filterName);
  });
  renderLogsView();
}

function toggleLogWrap() {
  logWrap = !logWrap;
  const btn = document.getElementById("logs-wrap-btn");
  if (btn) btn.textContent = `Wrap: ${logWrap ? "On" : "Off"}`;
  renderLogsView();
}

function toggleLogFollow() {
  logFollow = !logFollow;
  const btn = document.getElementById("logs-follow-btn");
  if (btn) btn.textContent = `Follow: ${logFollow ? "On" : "Off"}`;
  renderLogsView();
}

function filteredLogLines() {
  const q = (document.getElementById("logs-search")?.value || "").trim().toLowerCase();
  return logBuffer.filter((line) => {
    const isHttp = line.includes(" -> ") || /\b(GET|POST|PUT|DELETE|PATCH)\s+\//.test(line);
    let levelMatch = true;
    if (activeLogFilter === "HTTP") levelMatch = isHttp;
    else if (activeLogFilter !== "ALL") levelMatch = line.includes(` ${activeLogFilter} `) || line.includes(`[${activeLogFilter}]`);
    if (!levelMatch) return false;
    return !q || line.toLowerCase().includes(q);
  });
}

function formatLogEntryHtml(line) {
  const m = line.match(/^\[?(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?Z?)\]?\s+(INFO|WARN|ERROR|ERR|DEBUG)\s*(?:\|\s*)?([\s\S]*)$/);
  const isHttp = line.includes(" -> ") || /\b(GET|POST|PUT|DELETE|PATCH)\s+\//.test(line);
  if (m) {
    const [, ts, rawLvl, rest] = m;
    const lvl = rawLvl === "ERR" ? "ERROR" : rawLvl;
    const badgeCls = lvl === "ERROR" ? "log-lvl-error"
      : lvl === "WARN" ? "log-lvl-warn"
      : lvl === "DEBUG" ? "log-lvl-debug"
      : isHttp ? "log-lvl-http" : "log-lvl-info";
    const bodyCls = lvl === "ERROR" ? "ink-error"
      : lvl === "WARN" ? "ink-tertiary"
      : lvl === "DEBUG" ? "ink-muted" : "";
    return `<span class="log-ts">${escapeHtml(ts)}</span><span class="log-badge ${badgeCls}">${escapeHtml(isHttp ? "HTTP" : lvl)}</span><span class="${bodyCls}">${escapeHtml(rest)}</span>`;
  }
  const httpMatch = line.match(/^\[?(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?Z?)\]?\s+((?:GET|POST|PUT|DELETE|PATCH)\s+\/[\s\S]*)$/);
  if (httpMatch) {
    const [, ts, rest] = httpMatch;
    const isErrStatus = /\b[45]\d{2}\b/.test(rest);
    const badgeCls = isErrStatus ? "log-lvl-error" : "log-lvl-http";
    const bodyCls = isErrStatus ? "ink-error" : "";
    return `<span class="log-ts">${escapeHtml(ts)}</span><span class="log-badge ${badgeCls}">HTTP</span><span class="${bodyCls}">${escapeHtml(rest)}</span>`;
  }
  const fallbackCls = (line.includes(" ERROR ") || line.includes(" ERR |")) ? "ink-error"
    : line.includes(" WARN ") ? "ink-tertiary"
    : line.includes(" DEBUG ") ? "ink-muted"
    : line.includes(" -> 200 OK") ? "ink-secondary" : "";
  return `<span class="${fallbackCls}">${escapeHtml(line)}</span>`;
}

function renderLogsView() {
  const box = document.getElementById("logs-scroll");
  if (!box) return;
  const lines = filteredLogLines();
  if (lines.length === 0) {
    box.innerHTML = `<div class="log-empty">No log entries match the active filter.</div>`;
    return;
  }
  box.innerHTML = lines.map((line) => {
    return `<div class="log-entry ${logWrap ? "" : "nowrap"}">${formatLogEntryHtml(line)}</div>`;
  }).join("");
  if (logFollow) box.scrollTop = box.scrollHeight;
}

function copyVisibleLogs() {
  const text = filteredLogLines().join("\n");
  if (navigator.clipboard && navigator.clipboard.writeText) {
    navigator.clipboard.writeText(text).then(() => showToast("Copied visible logs."));
  }
}

function clearVisibleLogs() {
  logBuffer = [];
  renderLogsView();
}

async function revealLogsInFinder() {
  try {
    const res = await fetch(`${API_BASE}/v1/logs/reveal`, {
      method: "POST",
      headers: { "Content-Type": "application/json" }
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
  } catch (err) {
    showToast("Reveal logs failed: " + err.message, true);
  }
}

function handleDelegatedActionClick(event) {
  const unmountVaultBtn = event.target && event.target.closest ? event.target.closest("[data-unmount-vault-pack]") : null;
  if (unmountVaultBtn) {
    unmountVaultPackFromModal(unmountVaultBtn.getAttribute("data-unmount-vault-pack") || "");
    return;
  }
  const loadDecBtn = event.target && event.target.closest ? event.target.closest("[data-load-decision-path]") : null;
  if (loadDecBtn && typeof loadDecisionSidecarFromWeb === "function") {
    const p = loadDecBtn.getAttribute("data-load-decision-path") || "";
    const vis = loadDecBtn.getAttribute("data-load-decision-vision") !== "false";
    loadDecisionSidecarFromWeb(p, "auto", vis);
    return;
  }
  const btn = event.target && event.target.closest ? event.target.closest("[data-select-model-name]") : null;
  if (!btn) return;
  const name = btn.getAttribute("data-select-model-name") || "";
  const packPath = btn.getAttribute("data-select-model-path") || "";
  selectAndApplyModel(name, packPath);
  if (btn.getAttribute("data-close-switch-modal") === "true") closeSwitchModelModal();
}

async function refreshMountedVaultPacksInModal() {
  if (typeof document === "undefined") return;
  const sec = document.getElementById("mempack-mounted-section");
  const list = document.getElementById("mempack-mounted-list");
  if (!sec || !list) return;
  try {
    const res = await fetch(`${API_BASE}/v1/memory/packs`);
    if (!res.ok) return;
    const data = await res.json();
    const packs = Array.isArray(data.packs) ? data.packs : [];
    sec.hidden = packs.length === 0;
    list.innerHTML = packs.map((p) => {
      const name = String(p.pack_name || "pack");
      const root = String(p.root_path || "");
      const role = p.is_primary_cortex ? "Primary Cortex" : "Vault Pack";
      const pdfInfo = p.pdf_file_count ? ` · ${p.pdf_file_count} PDF` : "";
      const astInfo = p.ast_declaration_count ? ` · ${p.ast_declaration_count} AST` : "";
      return `<div class="mempack-artifact-row">
        <div class="mempack-artifact-meta">
          <strong class="mono" style="font-size:12px;">${escapeHtml(name)} (${role})</strong>
          <span class="panel-sub mono">${p.file_count || 0} files${pdfInfo}${astInfo} · ${escapeHtml(middleTruncate(root, 44))}</span>
        </div>
        <div><button type="button" class="btn btn-sm" data-unmount-vault-pack="${escapeHtml(name)}">Unmount</button></div>
      </div>`;
    }).join("");
  } catch (_) {}
}

async function unmountVaultPackFromModal(packId) {
  if (!packId) return;
  try {
    const res = await fetch(`${API_BASE}/v1/memory/unmount`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ pack_id: packId })
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    showToast(`Unmounted '${packId}'.`);
    await refreshMountedVaultPacksInModal();
    if (typeof fetchEngineInfo === "function") fetchEngineInfo();
    if (typeof fetchDashboard === "function") fetchDashboard();
  } catch (err) {
    showToast("Unmount failed: " + err.message, true);
  }
}

function wireModernWebPlatformControls() {
  if (typeof document === "undefined") return;
  const tablist = document.querySelector("[role='tablist']");
  if (tablist) {
    tablist.addEventListener("keydown", (e) => {
      const tabs = Array.from(tablist.querySelectorAll("[role='tab']"));
      const idx = tabs.indexOf(document.activeElement);
      if (idx < 0 || !tabs.length) return;
      let next = -1;
      if (e.key === "ArrowRight") next = (idx + 1) % tabs.length;
      else if (e.key === "ArrowLeft") next = (idx - 1 + tabs.length) % tabs.length;
      else if (e.key === "Home") next = 0;
      else if (e.key === "End") next = tabs.length - 1;
      if (next >= 0) {
        e.preventDefault();
        tabs[next].focus();
        const tabId = String(tabs[next].id || "").replace(/^nav-tab-/, "");
        if (tabId && typeof switchTab === "function") switchTab(tabId);
      }
    });
  }
  const switchDlg = document.getElementById("modal-switch-model");
  if (switchDlg) {
    let downOut = false;
    const isOutside = (ev, d) => {
      const r = d.getBoundingClientRect();
      return ev.clientX < r.left || ev.clientX > r.right || ev.clientY < r.top || ev.clientY > r.bottom;
    };
    switchDlg.addEventListener("pointerdown", (e) => { downOut = isOutside(e, switchDlg); });
    switchDlg.addEventListener("pointerup", (e) => { if (downOut && isOutside(e, switchDlg)) switchDlg.close(); downOut = false; });
  }
  const memDlg = document.getElementById("modal-memory-pack");
  if (memDlg) {
    memDlg.addEventListener("cancel", (e) => {
      if (typeof mempackModalState !== "undefined" && mempackModalState && mempackModalState.indexing) e.preventDefault();
    });
  }
  const updDlg = document.getElementById("modal-update");
  if (updDlg) {
    updDlg.addEventListener("cancel", (e) => {
      const p = typeof webUpdateState !== "undefined" && webUpdateState && webUpdateState.snapshot && webUpdateState.snapshot.phase;
      if (p === "downloading" || p === "verifying" || p === "installing") e.preventDefault();
    });
  }
  if (typeof HTMLButtonElement === "undefined" || !("command" in HTMLButtonElement.prototype)) {
    document.addEventListener("click", (e) => {
      const inv = e.target && e.target.closest ? e.target.closest("[command][commandfor]") : null;
      if (!inv) return;
      const target = document.getElementById(inv.getAttribute("commandfor") || "");
      const cmd = inv.getAttribute("command");
      if (target && cmd === "close" && typeof target.close === "function") target.close();
      else if (target && cmd === "show-modal" && typeof target.showModal === "function") target.showModal();
    });
  }
}

if (typeof window !== "undefined") {
  window.addEventListener("DOMContentLoaded", () => {
    wireModernWebPlatformControls();
    const port = new URL(API_BASE, window.location.href).port;
    const portEl = document.getElementById("header-port");
    if (portEl) portEl.textContent = port ? `Port ${port}` : "Port —";

    const tabParam = new URLSearchParams(window.location.search).get("tab");
    if (tabParam) switchTab(tabParam, false);

    if (typeof document !== "undefined") {
      document.addEventListener("click", handleDelegatedActionClick);
    }

    wireSparklineHover();
    window.addEventListener("resize", syncHeroTitleMarquee);
    fetchDashboard();
    initMetricsStream();
    fetchEngineInfo();
    if (typeof initWebUpdates === "function") initWebUpdates();
    setInterval(fetchEngineInfo, 3000);
    window.addEventListener("beforeunload", closeLogStream);
  });
}
