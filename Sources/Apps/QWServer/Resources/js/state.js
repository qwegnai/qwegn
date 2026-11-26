const API_BASE = (typeof window !== "undefined" && window.location && window.location.origin) ? window.location.origin : "http://127.0.0.1:8080";
let isConnected = false;
let publishedModel = null;
let modelGeneration = 0;
let discoveredModelsList = [];
let discoveredDraftersList = [];
let metricsEventSource = null;
let logsEventSource = null;
let expandedRequestId = null;
let logBuffer = [];
let activeLogFilter = "ALL";
let logWrap = true;
let logFollow = true;
let loadedSettingsObj = null;
let wasPrefilling = false;
let prefillHoldTimer = null;
let lastPrefillTargetValue = null;
let lastPrefillTargetText = "";
let smokeProbeState = {
  status: "empty",
  content: "",
  reasoning: "",
  rawText: "",
  isJsonFallback: false,
  mode: "rendered"
};

function formatTokenCount(raw) {
  if (typeof raw !== "number" || !Number.isFinite(raw)) return "—";
  const n = Math.round(raw);
  const sign = n < 0 ? "-" : "";
  const absVal = Math.abs(n);
  if (absVal >= 99995000) return `${sign}${(absVal / 1000000).toFixed(1)}M`;
  if (absVal >= 1000000) return `${sign}${(absVal / 1000000).toFixed(2)}M`;
  return `${sign}${absVal.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",")}`;
}

function showToast(message, isError = false) {
  const host = document.getElementById("toast-region");
  if (!host) return;
  const node = document.createElement("div");
  node.className = `toast ${isError ? "toast-error" : ""}`;
  node.textContent = message;
  host.appendChild(node);
  setTimeout(() => { node.remove(); }, 3800);
}

function escapeHtml(raw) {
  return String(raw ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function switchTab(tabId, updateUrl = true) {
  const validTabs = ["monitor", "chat", "decisions", "models", "logs", "settings"];
  const resolved = validTabs.includes(tabId) ? tabId : "monitor";
  document.querySelectorAll(".tab-pane").forEach((el) => el.classList.add("hidden"));
  const target = document.getElementById(`view-${resolved}`);
  if (target) target.classList.remove("hidden");

  validTabs.forEach((t) => {
    const btn = document.getElementById(`nav-tab-${t}`);
    if (!btn) return;
    const active = t === resolved;
    btn.classList.toggle("is-active", active);
    if (typeof btn.setAttribute === "function") {
      btn.setAttribute("aria-selected", active ? "true" : "false");
      btn.setAttribute("tabindex", active ? "0" : "-1");
    }
    if (active && typeof btn.scrollIntoView === "function") {
      btn.scrollIntoView({ block: "nearest", inline: "nearest" });
    }
  });

  if (updateUrl && window.history && typeof window.history.replaceState === "function") {
    const nextUrl = new URL(window.location.href);
    nextUrl.searchParams.set("tab", resolved);
    window.history.replaceState(null, "", nextUrl.toString());
  }

  if (resolved === "models") loadDiscoveredModels();
  if (resolved === "logs") initLogStream();
  else closeLogStream();
  if (resolved === "settings") loadLiveSettings();
  if (resolved === "chat" && typeof onChatTabActivated === "function") onChatTabActivated();
  if (resolved === "decisions" && typeof onDecisionsTabActivated === "function") onDecisionsTabActivated();
}

function updateSparkline(firstSeries, secondSeries) {
  const first = Array.isArray(firstSeries) ? firstSeries : [];
  const second = Array.isArray(secondSeries) ? secondSeries : [];
  const totalSlots = Math.max(first.length, second.length);
  const validPositive = (arr) => arr.filter((v) => v !== null && v !== undefined && v > 0);
  const posA = validPositive(first);
  const posB = validPositive(second);
  const peakA = posA.length ? Math.max(...posA) : 0;
  const peakB = posB.length ? Math.max(...posB) : 0;

  let ceilA = 0.80;
  let ceilB = 0.80;
  if (peakA > 0 && peakB > 0) {
    if (peakA > peakB * 1.5) {
      ceilA = 0.84;
      ceilB = 0.60;
    } else if (peakB > peakA * 1.5) {
      ceilA = 0.60;
      ceilB = 0.84;
    } else {
      ceilA = 0.84;
      ceilB = 0.70;
    }
  }

  const makePathGeo = (history, companion, peak, companionPeak, ceiling) => {
    const runs = [];
    let current = [];
    const maxRef = Math.max(1, peak > 0 ? peak / ceiling : 1);
    history.forEach((val, idx) => {
      if (val === null || val === undefined) {
        const comp = companion[idx];
        if (comp !== null && comp !== undefined && Number(comp) > 0) return;
        if (current.length > 0) runs.push(current);
        current = [];
        return;
      }
      const clamped = Math.max(0, Number(val));
      const x = totalSlots <= 1 ? 588 : 12 + (idx / (totalSlots - 1)) * 576;
      const norm = peak > 0 ? Math.min(1, Math.max(0, clamped / maxRef)) : 0;
      const y = 94 - norm * 84;
      current.push({ x, y, raw: clamped, idx });
    });
    if (current.length > 0) runs.push(current);

    const lineRuns = [];
    const areaRuns = [];
    const markerRuns = [];
    let guideD = "";
    let lastPoint = null;

    runs.forEach((pts) => {
      if (pts.length === 0) return;
      lastPoint = pts[pts.length - 1];
      if (pts.length >= 2) {
        let d = `M ${pts[0].x.toFixed(1)},${pts[0].y.toFixed(1)}`;
        for (let i = 1; i < pts.length; i++) {
          const prev = pts[i - 1];
          const curr = pts[i];
          const cpX = (prev.x + curr.x) / 2;
          d += ` C ${cpX.toFixed(1)},${prev.y.toFixed(1)} ${cpX.toFixed(1)},${curr.y.toFixed(1)} ${curr.x.toFixed(1)},${curr.y.toFixed(1)}`;
        }
        lineRuns.push(d);
        const firstPt = pts[0];
        const lastPt = pts[pts.length - 1];
        areaRuns.push(`${d} L ${lastPt.x.toFixed(1)},96 L ${firstPt.x.toFixed(1)},96 Z`);
      } else {
        const pt = pts[0];
        markerRuns.push(`M ${(pt.x - 0.1).toFixed(1)},${pt.y.toFixed(1)} L ${(pt.x + 0.1).toFixed(1)},${pt.y.toFixed(1)}`);
      }
    });

    if (runs.length === 1 && runs[0].length === 1 && (totalSlots === 1 || companionPeak > 0)) {
      const pt = runs[0][0];
      const yStr = pt.y.toFixed(1);
      guideD = `M 12,${yStr} L 588,${yStr}`;
      areaRuns.push(`M 12,${yStr} L 588,${yStr} L 588,96 L 12,96 Z`);
    } else if (lastPoint) {
      const marker = `M ${(lastPoint.x - 0.1).toFixed(1)},${lastPoint.y.toFixed(1)} L ${(lastPoint.x + 0.1).toFixed(1)},${lastPoint.y.toFixed(1)}`;
      if (!markerRuns.includes(marker)) markerRuns.push(marker);
    }

    return {
      d: lineRuns.join(" "),
      areaD: areaRuns.join(" "),
      guideD,
      markersD: markerRuns.join(" "),
      lastPt: lastPoint
    };
  };

  const drawSeries = (geometry, ids) => {
    const path = document.getElementById(ids.path);
    const area = document.getElementById(ids.area);
    const guide = document.getElementById(ids.guide);
    const markers = document.getElementById(ids.markers);
    const dot = document.getElementById(ids.dot);
    if (path) path.setAttribute("d", geometry.d);
    if (area) area.setAttribute("d", geometry.areaD);
    if (guide) guide.setAttribute("d", geometry.guideD);
    if (markers) markers.setAttribute("d", geometry.markersD);
    if (dot) {
      if (geometry.lastPt) {
        dot.setAttribute("cx", geometry.lastPt.x.toFixed(1));
        dot.setAttribute("cy", geometry.lastPt.y.toFixed(1));
        dot.classList.remove("opacity-0");
      } else {
        dot.classList.add("opacity-0");
      }
    }
  };

  drawSeries(makePathGeo(first, second, peakA, peakB, ceilA), {
    path: "sparkline-series-a-path",
    area: "sparkline-series-a-area",
    guide: "sparkline-series-a-guide",
    markers: "sparkline-series-a-markers",
    dot: "sparkline-series-a-dot"
  });
  drawSeries(makePathGeo(second, first, peakB, peakA, ceilB), {
    path: "sparkline-series-b-path",
    area: "sparkline-series-b-area",
    guide: "sparkline-series-b-guide",
    markers: "sparkline-series-b-markers",
    dot: "sparkline-series-b-dot"
  });
}

function wireSparklineHover() {
  const svg = document.getElementById("sparkline-svg");
  const tip = document.getElementById("sparkline-tooltip");
  if (!svg || !tip) return;
  svg.addEventListener("mousemove", (evt) => {
    if (!publishedModel) return;
    const slots = groupCards(publishedModel, "throughput").slice(0, 2);
    const s0 = (slots[0] && Array.isArray(slots[0].series)) ? slots[0].series : [];
    const s1 = (slots[1] && Array.isArray(slots[1].series)) ? slots[1].series : [];
    const len = Math.max(s0.length, s1.length);
    if (len === 0) return;
    const rect = svg.getBoundingClientRect();
    const frac = Math.min(1, Math.max(0, (evt.clientX - rect.left) / Math.max(1, rect.width)));
    const idx = Math.min(len - 1, Math.round(frac * Math.max(0, len - 1)));
    const v0 = s0[idx];
    const v1 = s1[idx];
    const fmt = (val, unit) => (val === null || val === undefined) ? "—" : `${Number(val).toFixed(1)}${unit ? ` ${unit}` : ""}`;
    const label0 = slots[0] ? slots[0].title : "A";
    const label1 = slots[1] ? slots[1].title : "B";
    tip.textContent = `#${idx + 1} · ${label0}: ${fmt(v0, slots[0]?.unit)} · ${label1}: ${fmt(v1, slots[1]?.unit)}`;
    tip.classList.remove("hidden");
  });
  svg.addEventListener("mouseleave", () => {
    tip.classList.add("hidden");
  });
}

function middleTruncate(str, maxLen) {
  const s = String(str || "");
  if (s.length <= maxLen || maxLen < 12) return s;
  const head = Math.ceil((maxLen - 1) * 0.45);
  const tail = Math.floor((maxLen - 1) * 0.55);
  return `${s.slice(0, head)}…${s.slice(-tail)}`;
}

// Memory Pack Inspection & Live Indexing Modal Helpers
let mempackModalState = { info: null, fromSettings: false, indexing: false, abortCtrl: null, pendingEvent: null, rafId: 0 };

function notifyMemoryPackStatus(message, isError = false) {
  if (typeof showToast === "function") showToast(message, isError);
  if (typeof notifyChat === "function") notifyChat(message, isError);
}

async function inspectMemoryPackAndMount(repoPath, opts = {}) {
  if (typeof fetch === "undefined") return;
  try {
    const res = await fetch(`${API_BASE}/v1/memory/inspect`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path: repoPath })
    });
    const info = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(info.message || `HTTP ${res.status}`);
    if (info.status === "ready" && !opts.forceModal) {
      await performDirectMemoryPackMount(info.rootPath || repoPath, Boolean(opts.fromSettings));
      return;
    }
    openMemoryPackModalWithInspection(info, Boolean(opts.fromSettings));
  } catch (err) {
    notifyMemoryPackStatus("Memory Pack inspect failed: " + err.message, true);
  }
}

async function performDirectMemoryPackMount(repoPath, fromSettings = false) {
  if (typeof fetch === "undefined") return;
  try {
    const res = await fetch(`${API_BASE}/v1/memory/mount`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path: repoPath })
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.message || `HTTP ${res.status}`);
    syncWorkspaceAfterMemoryPackAction(data.rootPath || repoPath);
    if (typeof fetchDashboard === "function") await fetchDashboard();
    if (fromSettings || (typeof loadedSettingsObj !== "undefined" && loadedSettingsObj)) {
      if (typeof loadLiveSettings === "function") await loadLiveSettings();
    }
    notifyMemoryPackStatus(`Mounted .qwcache '${data.packName || repoPath}' (${data.fileCount || 0} files, KV ${String(data.enforcedKVMode || "q4").toUpperCase()}).`);
  } catch (err) {
    notifyMemoryPackStatus("Memory Pack mount failed: " + err.message, true);
  }
}

function syncWorkspaceAfterMemoryPackAction(rootPath) {
  if (!rootPath) return;
  const session = typeof getActiveChatSession === "function" ? getActiveChatSession() : null;
  if (session) {
    session.workspacePath = rootPath;
    session.updatedAt = Date.now();
    if (typeof persistChatSessions === "function") persistChatSessions();
  }
  if (typeof document !== "undefined") {
    const chatInput = document.getElementById("chat-workspace-path");
    if (chatInput) chatInput.value = rootPath;
    const settingsInput = document.getElementById("settings-mempack-path");
    if (settingsInput) settingsInput.value = rootPath;
  }
}

function describeInspectionSummary(info) {
  const active = info.activeModelName || "active model";
  const count = info.candidateFileCount || 0;
  if (info.status === "missing") {
    return `No compatible .qwcache index exists yet for '${active}'. ${count} indexable workspace file(s) found.`;
  }
  if (info.status === "wrong_model") {
    return `Found .qwcache built for a different model (active model is '${active}'). Index for '${active}' or switch models below.`;
  }
  if (info.status === "stale") {
    const reasons = [];
    if (info.commitsBehind > 0) reasons.push(`${info.commitsBehind} commit(s) behind HEAD`);
    if (info.gitDiverged) reasons.push("branch/HEAD diverged");
    if (info.modifiedFiles && info.modifiedFiles.length) reasons.push(`${info.modifiedFiles.length} modified`);
    if (info.deletedFiles && info.deletedFiles.length) reasons.push(`${info.deletedFiles.length} deleted`);
    return `Existing .qwcache is out of date (${reasons.join(", ") || "workspace changed"}). Update before mounting or mount with live delta.`;
  }
  return `Workspace .qwcache matches active model '${active}' (${count} workspace file(s)) and is ready to mount.`;
}

function openMemoryPackModalWithInspection(info, fromSettings = false) {
  if (typeof document === "undefined") return;
  const dialog = document.getElementById("modal-memory-pack");
  if (!dialog) return;
  mempackModalState = { info, fromSettings: Boolean(fromSettings), indexing: false, abortCtrl: null, pendingEvent: null, rafId: 0 };

  const badgeEl = document.getElementById("mempack-status-badge");
  const statusLabels = { ready: "Ready", stale: "Stale Index", wrong_model: "Different Model", missing: "No .qwcache" };
  if (badgeEl) {
    badgeEl.textContent = statusLabels[info.status] || info.status;
    badgeEl.className = `mempack-badge is-${info.status || "missing"}`;
  }
  const rootEl = document.getElementById("mempack-root-path");
  if (rootEl) rootEl.textContent = info.rootPath || "—";
  const summaryEl = document.getElementById("mempack-status-summary");
  if (summaryEl) summaryEl.textContent = describeInspectionSummary(info);

  const gitEl = document.getElementById("mempack-git-summary");
  if (gitEl) {
    const parts = [];
    if (info.manifestGitHeadSHA) parts.push(`Indexed SHA: ${String(info.manifestGitHeadSHA).slice(0, 10)}`);
    if (info.gitHeadSHA) parts.push(`HEAD: ${String(info.gitHeadSHA).slice(0, 10)}`);
    const deltaPreview = [...(info.modifiedFiles || []), ...(info.deletedFiles || []).map((p) => `[deleted] ${p}`)].slice(0, 3);
    if (deltaPreview.length) parts.push(`Changed: ${deltaPreview.join(", ")}`);
    gitEl.textContent = parts.join(" · ");
    gitEl.hidden = parts.length === 0;
  }

  const artSection = document.getElementById("mempack-artifacts-section");
  const artList = document.getElementById("mempack-artifacts-list");
  const artifacts = Array.isArray(info.availableArtifacts) ? info.availableArtifacts : [];
  if (artSection && artList) {
    artSection.hidden = artifacts.length === 0;
    artList.innerHTML = artifacts.map((entry) => {
      const a = entry.artifact || {};
      const mib = a.cacheFileBytes ? `${(a.cacheFileBytes / 1048576).toFixed(1)} MiB` : "—";
      const targetModel = entry.installedModelName || a.modelName || "";
      let actionHtml = `<span class="panel-sub mono">Not installed</span>`;
      if (!entry.cacheFileExists) {
        actionHtml = `<span class="panel-sub mono">File missing</span>`;
      } else if (entry.matchesActiveModel) {
        actionHtml = `<span class="panel-sub mono">Active model</span>`;
      } else if (entry.modelInstalled && targetModel) {
        actionHtml = `<button type="button" class="btn btn-sm" data-mempack-switch-model="${escapeHtml(targetModel)}">Switch &amp; Mount</button>`;
      }
      return `<div class="mempack-artifact-row">
        <div class="mempack-artifact-meta">
          <strong class="mono" style="font-size:12px;">${escapeHtml(a.modelName || "model")}</strong>
          <span class="panel-sub mono">kv=${escapeHtml(a.kvMode || "?")} · lsa=${escapeHtml(a.lsaMode || "?")} · ${a.tokenCount || 0} tok · ${mib}</span>
        </div>
        <div>${actionHtml}</div>
      </div>`;
    }).join("");
  }

  const rec = info.recommendedOptions || {};
  const kvSel = document.getElementById("mempack-opt-kv");
  const lsaSel = document.getElementById("mempack-opt-lsa");
  const scopeSel = document.getElementById("mempack-opt-scope");
  const testsChk = document.getElementById("mempack-opt-tests");
  if (kvSel && rec.kvMode) kvSel.value = rec.kvMode;
  if (lsaSel && rec.lsaMode) lsaSel.value = rec.lsaMode;
  if (scopeSel && rec.scope) scopeSel.value = rec.scope;
  if (testsChk) testsChk.checked = Boolean(rec.includeTests);

  const progBox = document.getElementById("mempack-progress-box");
  if (progBox) progBox.hidden = true;
  renderMemoryPackProgressFrame({ phase: "discovering", progressFraction: 0, processedFiles: 0, totalFiles: info.candidateFileCount || 0, message: "Ready" });
  renderMemoryPackActionsBar(info);
  setMemoryPackControlsDisabled(false);
  if (typeof dialog.showModal === "function" && !dialog.open) dialog.showModal();
}

function renderMemoryPackActionsBar(info) {
  const bar = document.getElementById("mempack-actions-bar");
  if (!bar) return;
  const primaryLabel = info.status === "stale" ? "Update & Mount"
    : info.status === "wrong_model" ? `Index for ${escapeHtml(info.activeModelName || "Active Model")} & Mount`
    : info.status === "ready" ? "Rebuild & Mount" : "Index & Mount";
  const showMountExisting = (info.status === "stale" || info.status === "ready") && info.matchingArtifact;
  const existingLabel = info.status === "stale" ? "Mount Existing with Live Delta" : "Mount Existing";
  bar.innerHTML = `<button type="button" class="btn btn-sm" onclick="closeMemoryPackModal()">Cancel</button>
    ${showMountExisting ? `<button type="button" class="btn btn-sm" onclick="mountExistingFromMemoryPackModal()">${existingLabel}</button>` : ""}
    <button id="mempack-start-btn" type="button" class="btn btn-sm btn-primary" onclick="startMemoryPackIndexing()">${primaryLabel}</button>`;
}

function setMemoryPackControlsDisabled(disabled) {
  ["mempack-opt-kv", "mempack-opt-lsa", "mempack-opt-scope", "mempack-opt-tests"].forEach((id) => {
    const el = document.getElementById(id);
    if (el) el.disabled = Boolean(disabled);
  });
  const bar = document.getElementById("mempack-actions-bar");
  if (bar) bar.querySelectorAll("button").forEach((btn) => { if (btn.textContent !== "Cancel") btn.disabled = Boolean(disabled); });
  const list = document.getElementById("mempack-artifacts-list");
  if (list) list.querySelectorAll("button").forEach((btn) => { btn.disabled = Boolean(disabled); });
}

function closeMemoryPackModal() {
  if (mempackModalState.abortCtrl) { mempackModalState.abortCtrl.abort(); mempackModalState.abortCtrl = null; }
  mempackModalState.indexing = false;
  if (mempackModalState.rafId && typeof cancelAnimationFrame === "function") {
    cancelAnimationFrame(mempackModalState.rafId);
    mempackModalState.rafId = 0;
  }
  if (typeof document !== "undefined") {
    const dialog = document.getElementById("modal-memory-pack");
    if (dialog && dialog.open && typeof dialog.close === "function") dialog.close();
  }
}

async function mountExistingFromMemoryPackModal() {
  const info = mempackModalState.info;
  if (!info || !info.rootPath) return;
  const fromSettings = mempackModalState.fromSettings;
  closeMemoryPackModal();
  await performDirectMemoryPackMount(info.rootPath, fromSettings);
}

async function switchModelAndMountMemoryPack(rawModelName) {
  const info = mempackModalState.info;
  if (!info || mempackModalState.indexing) return;
  const modelName = String(rawModelName || "");
  if (!modelName) return;
  mempackModalState.indexing = true;
  setMemoryPackControlsDisabled(true);
  const progBox = document.getElementById("mempack-progress-box");
  if (progBox) progBox.hidden = false;
  renderMemoryPackProgressFrame({ phase: "loading_model", progressFraction: 0.45, message: `Switching active model to ${modelName}…` });
  try {
    const loadRes = await fetch(`${API_BASE}/v1/models/load`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model: modelName })
    });
    const loadData = await loadRes.json().catch(() => ({}));
    if (!loadRes.ok) throw new Error(loadData.message || `HTTP ${loadRes.status}`);
    if (typeof fetchEngineInfo === "function") await fetchEngineInfo();
    const reinspectRes = await fetch(`${API_BASE}/v1/memory/inspect`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path: info.rootPath })
    });
    const nextInfo = await reinspectRes.json().catch(() => ({}));
    if (!reinspectRes.ok) throw new Error(nextInfo.message || `HTTP ${reinspectRes.status}`);
    if (nextInfo.status === "ready") {
      const fromSettings = mempackModalState.fromSettings;
      closeMemoryPackModal();
      await performDirectMemoryPackMount(nextInfo.rootPath || info.rootPath, fromSettings);
    } else {
      openMemoryPackModalWithInspection(nextInfo, mempackModalState.fromSettings);
      notifyMemoryPackStatus(`Switched active model to ${modelName}; review staleness before mounting.`);
    }
  } catch (err) {
    mempackModalState.indexing = false;
    setMemoryPackControlsDisabled(false);
    notifyMemoryPackStatus("Switch model failed: " + err.message, true);
  }
}

function scheduleMemoryPackProgressRender(ev) {
  mempackModalState.pendingEvent = ev;
  if (typeof requestAnimationFrame !== "function") { renderMemoryPackProgressFrame(ev); return; }
  if (mempackModalState.rafId) return;
  mempackModalState.rafId = requestAnimationFrame(() => {
    mempackModalState.rafId = 0;
    if (mempackModalState.pendingEvent) renderMemoryPackProgressFrame(mempackModalState.pendingEvent);
  });
}

function renderMemoryPackProgressFrame(ev) {
  if (typeof document === "undefined" || !ev) return;
  const pct = Math.max(0, Math.min(100, Math.round((ev.progressFraction || 0) * 100)));
  const phaseEl = document.getElementById("mempack-progress-phase");
  const pctEl = document.getElementById("mempack-progress-pct");
  const trackEl = document.getElementById("mempack-progress-track");
  const fillEl = document.getElementById("mempack-progress-fill");
  const filesEl = document.getElementById("mempack-progress-files");
  const toksEl = document.getElementById("mempack-progress-tokens");
  const fileEl = document.getElementById("mempack-progress-current-file");
  if (phaseEl) phaseEl.textContent = ev.message || ev.phase || "Indexing…";
  if (pctEl) pctEl.textContent = `${pct}%`;
  if (trackEl) trackEl.setAttribute("aria-valuenow", String(pct));
  if (fillEl) fillEl.style.width = `${pct}%`;
  if (filesEl) filesEl.textContent = `${ev.processedFiles || 0} / ${ev.totalFiles || 0} files`;
  if (toksEl) {
    const tokText = ev.prefillTotalTokens > 0 ? `${ev.prefillProcessedTokens || 0} / ${ev.prefillTotalTokens} prefill tok` : `${ev.cortexTokens || 0} tok`;
    toksEl.textContent = tokText;
  }
  if (fileEl) fileEl.textContent = ev.currentFile || (ev.phase === "prefilling" ? "Running Metal prefill…" : "—");
}

async function startMemoryPackIndexing() {
  const info = mempackModalState.info;
  if (!info || !info.rootPath || mempackModalState.indexing) return;
  const kvEl = document.getElementById("mempack-opt-kv");
  const lsaEl = document.getElementById("mempack-opt-lsa");
  const scopeEl = document.getElementById("mempack-opt-scope");
  const testsEl = document.getElementById("mempack-opt-tests");
  const ctrl = typeof AbortController !== "undefined" ? new AbortController() : null;
  mempackModalState.indexing = true;
  mempackModalState.abortCtrl = ctrl;
  setMemoryPackControlsDisabled(true);
  const progBox = document.getElementById("mempack-progress-box");
  if (progBox) progBox.hidden = false;
  renderMemoryPackProgressFrame({ phase: "discovering", progressFraction: 0.02, message: "Starting live .qwcache indexer…" });

  let completedResult = null;
  try {
    const res = await fetch(`${API_BASE}/v1/memory/index`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      signal: ctrl ? ctrl.signal : undefined,
      body: JSON.stringify({
        path: info.rootPath,
        kvMode: kvEl ? kvEl.value : undefined,
        lsaMode: lsaEl ? lsaEl.value : undefined,
        scope: scopeEl ? scopeEl.value : "cortex_vault",
        includeTests: testsEl ? Boolean(testsEl.checked) : false,
        mountAfterIndex: true,
        stream: true
      })
    });
    if (!res.ok) {
      const errData = await res.json().catch(() => ({}));
      throw new Error(errData.message || `HTTP ${res.status}`);
    }
    if (!res.body || typeof res.body.getReader !== "function") {
      completedResult = await res.json();
    } else {
      const reader = res.body.getReader();
      const decoder = typeof TextDecoder !== "undefined" ? new TextDecoder() : null;
      let buf = "";
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += decoder ? decoder.decode(value, { stream: true }) : String(value || "");
        let nlIdx;
        while ((nlIdx = buf.indexOf("\n")) !== -1) {
          const line = buf.slice(0, nlIdx).trim();
          buf = buf.slice(nlIdx + 1);
          if (!line.startsWith("data: ")) continue;
          const payload = line.slice(6).trim();
          if (payload === "[DONE]") continue;
          let ev = null;
          try { ev = JSON.parse(payload); } catch (_) { continue; }
          if (ev.phase === "error") throw new Error(ev.error || ev.message || "Indexing failed");
          if (ev.result) completedResult = ev.result;
          scheduleMemoryPackProgressRender(ev);
        }
      }
    }
    mempackModalState.indexing = false;
    mempackModalState.abortCtrl = null;
    if (!completedResult) throw new Error("Indexing finished without completion summary.");
    renderMemoryPackProgressFrame({
      phase: "completed",
      processedFiles: completedResult.fileCount,
      totalFiles: completedResult.fileCount,
      prefillProcessedTokens: completedResult.tokenCount,
      prefillTotalTokens: completedResult.tokenCount,
      progressFraction: 1,
      message: "Completed and mounted."
    });
    syncWorkspaceAfterMemoryPackAction(completedResult.rootPath || info.rootPath);
    if (typeof fetchDashboard === "function") await fetchDashboard();
    if (mempackModalState.fromSettings || (typeof loadedSettingsObj !== "undefined" && loadedSettingsObj)) {
      if (typeof loadLiveSettings === "function") await loadLiveSettings();
    }
    const mib = (completedResult.cacheBytes / 1048576).toFixed(2);
    closeMemoryPackModal();
    notifyMemoryPackStatus(`Indexed & mounted '${completedResult.packName}' (${completedResult.fileCount} files, ${completedResult.tokenCount} tok, ${mib} MiB).`);
  } catch (err) {
    mempackModalState.indexing = false;
    mempackModalState.abortCtrl = null;
    if (err && err.name === "AbortError") return;
    setMemoryPackControlsDisabled(false);
    renderMemoryPackProgressFrame({ phase: "error", progressFraction: 0, message: "Error: " + err.message });
    notifyMemoryPackStatus("Memory Pack indexing failed: " + err.message, true);
  }
}

if (typeof document !== "undefined") {
  document.addEventListener("click", (event) => {
    const btn = event.target && event.target.closest ? event.target.closest("[data-mempack-switch-model]") : null;
    if (btn) switchModelAndMountMemoryPack(btn.getAttribute("data-mempack-switch-model") || "");
  });
}
