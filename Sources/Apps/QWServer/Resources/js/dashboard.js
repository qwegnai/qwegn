async function fetchEngineInfo() {
  try {
    const res = await fetch(`${API_BASE}/v1/engine/info`);
    if (!res.ok) throw new Error("HTTP " + res.status);
    const data = await res.json();
    const verEl = document.getElementById("header-version");
    if (verEl && data.version) {
      verEl.textContent = data.version.startsWith("v") ? data.version : `v${data.version}`;
    }
    const endpointEl = document.getElementById("ftr-endpoint");
    if (endpointEl) endpointEl.textContent = `Endpoint: ${API_BASE}`;

    const bandwidth = data.hardware?.dramBwGBs;
    const bwEl = document.getElementById("mem-dram-bw");
    const bwNote = document.getElementById("mem-dram-bw-note");
    if (bwEl) {
      const measured = typeof bandwidth === "number" && bandwidth > 0;
      bwEl.textContent = measured ? `${Math.round(bandwidth)} GB/s` : "Not measured";
      bwEl.className = measured ? "mono" : "panel-sub";
      if (bwNote) {
        bwNote.textContent = "Peak DRAM bandwidth";
      }
    }
  } catch (err) {
    console.warn("fetchEngineInfo error:", err);
  }
}

function updateStatusIndicators(lifecyclePhase, generationPhase, detail) {
  const daemonPulse = document.getElementById("daemon-pulse");
  const daemonStatus = document.getElementById("daemon-status");
  const heroDot = document.getElementById("hero-status-dot");
  const heroLabel = document.getElementById("hero-endpoint-label");
  const ftrStatus = document.getElementById("ftr-status-ready");

  const say = (dotText, pulseCls, heroText, footerText, inkCls) => {
    if (daemonPulse) daemonPulse.className = `dot ${pulseCls}`;
    if (daemonStatus) {
      daemonStatus.textContent = dotText;
      daemonStatus.className = inkCls;
    }
    if (heroDot) heroDot.className = `dot ${pulseCls}`;
    if (heroLabel) heroLabel.textContent = heroText;
    if (ftrStatus) {
      ftrStatus.textContent = footerText;
      ftrStatus.className = inkCls;
    }
  };

  if (!isConnected) {
    say("Disconnected", "dot-error", "Disconnected • No response", "No response from the server", "ink-error");
    return;
  }
  if (lifecyclePhase === "failed") {
    say("Failed", "dot-error", detail ? `Failed • ${detail}` : "Failed", detail || "The engine failed", "ink-error");
    return;
  }
  if (lifecyclePhase === "loading") {
    say("Loading", "dot-prefill", "Loading model", "Loading model weights…", "ink-tertiary");
    return;
  }
  if (lifecyclePhase === "unloading") {
    say("Unloading", "dot-prefill", "Unloading model", "Unloading model weights…", "ink-tertiary");
    return;
  }
  if (lifecyclePhase === "unloaded") {
    say("Model Unloaded", "dot-prefill", "Model Unloaded (Weights freed)", "Weights unloaded • Waiting for model load", "ink-tertiary");
    return;
  }
  if (lifecyclePhase === "loaded") {
    if (generationPhase === "prefill") {
      say("Prefilling Prompt", "dot-prefill", "Active • Prefill Phase", "Processing prompt prefill…", "ink-tertiary");
    } else if (generationPhase === "decode") {
      say("Streaming Output", "dot-decode", "Active • Decode Phase", "Generating tokens…", "ink-primary");
    } else if (generationPhase === "idle") {
      say("Daemon Running", "dot-active", "Connected • Idle (Ready)", "Ready for prompts", "ink-secondary");
    } else {
      say("State Unknown", "dot-idle", "Connected • State Unknown", "Engine state unknown", "ink-muted");
    }
    return;
  }
  say("State Unknown", "dot-idle", "Connected • State Unknown", "Engine state unknown", "ink-muted");
}

async function fetchDashboard() {
  const startedAt = modelGeneration;
  try {
    const res = await fetch(`${API_BASE}/v1/dashboard`);
    if (!res.ok) throw new Error("HTTP " + res.status);
    const model = await res.json();
    if (modelGeneration !== startedAt) return;
    renderDashboard(model);
  } catch (err) {
    console.warn("fetchDashboard error:", err);
    if (modelGeneration !== startedAt) return;
    renderDisconnected();
  }
}

function initMetricsStream() {
  if (metricsEventSource) metricsEventSource.close();
  metricsEventSource = new EventSource(`${API_BASE}/v1/dashboard/stream`);
  metricsEventSource.onmessage = (event) => {
    try {
      renderDashboard(JSON.parse(event.data));
    } catch (e) {
      console.error("Error parsing dashboard stream:", e);
    }
  };
  metricsEventSource.onerror = () => {
    metricsEventSource.close();
    renderDisconnected();
    setTimeout(initMetricsStream, 3000);
  };
}

function cardFor(model, id) {
  return (model.cards || []).find((c) => c.id === id) || null;
}

function cardText(card) {
  if (!card) return "—";
  if (card.value === null || card.value === undefined) return "—";
  return card.unit ? `${card.value} ${card.unit}` : `${card.value}`;
}

function cardReason(card) {
  if (!card) return "";
  if (card.value === null || card.value === undefined) return card.reason || "not measured";
  if (card.number === null || card.number === undefined) return card.reason || "";
  if (card.isStale) return ageText(card) || "stale";
  return "";
}

function ageText(card) {
  if (!card || typeof card.age !== "number") return "";
  if (card.age < 60) return `${Math.floor(card.age)} s ago`;
  return `${Math.floor(card.age / 60)} min ago`;
}

function renderCards(model) {
  document.querySelectorAll("[data-card]").forEach((el) => {
    const card = cardFor(model, el.getAttribute("data-card"));
    if (el.dataset.cardUnit === "separate") {
      el.textContent = card && card.value !== null && card.value !== undefined ? card.value : "—";
    } else {
      el.textContent = cardText(card);
    }
  });
  document.querySelectorAll("[data-card-reason]").forEach((el) => {
    const card = cardFor(model, el.getAttribute("data-card-reason"));
    el.textContent = cardReason(card);
  });
}

function renderPrefillBanner(model) {
  const banner = document.getElementById("prefill-banner");
  const fill = document.getElementById("prefill-progress-fill");
  const titleEl = document.getElementById("prefill-banner-title");
  const countWrap = document.getElementById("prefill-count-wrap");
  const completeCountEl = document.getElementById("prefill-complete-count");
  const pctEl = document.getElementById("prefill-pct");
  if (!banner) return;
  const cards = groupCards(model, "prefill");
  const prog = cards[0] && typeof cards[0].number === "number" ? cards[0].number : null;
  const total = cards[1] && typeof cards[1].number === "number" ? cards[1].number : null;
  const isPrefilling = model.generation === "prefill" && prog !== null && total !== null && total > 0;
  const inactiveReason = (cards[0] && cards[0].reason) || (cards[1] && cards[1].reason) || "";

  if (isPrefilling) {
    if (prefillHoldTimer) {
      clearTimeout(prefillHoldTimer);
      prefillHoldTimer = null;
    }
    wasPrefilling = true;
    lastPrefillTargetValue = cards[1] && cards[1].value !== null && cards[1].value !== undefined ? cards[1].value : null;
    lastPrefillTargetText = cardText(cards[1]);
    banner.classList.remove("is-complete");
    banner.classList.add("is-active");
    banner.removeAttribute("title");
    if (titleEl) titleEl.textContent = "Prefill in Progress";
    if (countWrap) countWrap.hidden = false;
    if (completeCountEl) {
      completeCountEl.hidden = true;
      completeCountEl.textContent = "";
    }
    const pct = Math.min(100, Math.max(0, (prog / total) * 100));
    if (pctEl) pctEl.textContent = `${Math.round(pct)}%`;
    if (fill) fill.style.width = `${pct.toFixed(1)}%`;
    return;
  }

  if (wasPrefilling && lastPrefillTargetValue !== null) {
    wasPrefilling = false;
    banner.classList.remove("is-active");
    banner.classList.add("is-complete");
    banner.removeAttribute("title");
    if (titleEl) titleEl.textContent = "Prefill Complete";
    if (countWrap) countWrap.hidden = true;
    if (completeCountEl) {
      completeCountEl.textContent = `${lastPrefillTargetValue} / ${lastPrefillTargetText}`;
      completeCountEl.hidden = false;
    }
    if (pctEl) pctEl.textContent = "100%";
    if (fill) fill.style.width = "100%";
    prefillHoldTimer = setTimeout(() => {
      prefillHoldTimer = null;
      lastPrefillTargetValue = null;
      lastPrefillTargetText = "";
      banner.classList.remove("is-active", "is-complete");
      if (inactiveReason) banner.title = inactiveReason;
      else banner.removeAttribute("title");
      if (titleEl) titleEl.textContent = "Prefill Progress";
      if (countWrap) countWrap.hidden = false;
      if (completeCountEl) {
        completeCountEl.hidden = true;
        completeCountEl.textContent = "";
      }
      if (pctEl) pctEl.textContent = "—";
      if (fill) fill.style.width = "0%";
    }, 650);
    return;
  }

  wasPrefilling = false;
  if (!prefillHoldTimer) {
    lastPrefillTargetValue = null;
    lastPrefillTargetText = "";
    banner.classList.remove("is-active", "is-complete");
    if (inactiveReason) banner.title = inactiveReason;
    else banner.removeAttribute("title");
    if (titleEl) titleEl.textContent = "Prefill Progress";
    if (countWrap) countWrap.hidden = false;
    if (completeCountEl) {
      completeCountEl.hidden = true;
      completeCountEl.textContent = "";
    }
    if (pctEl) pctEl.textContent = "—";
    if (fill) fill.style.width = "0%";
  }
}

function renderVelocityCards(model) {
  const host = document.getElementById("velocity-cards");
  const cards = groupCards(model, "throughput");
  if (host) {
    host.innerHTML = "";
    cards.forEach((card, index) => {
      const node = document.createElement("div");
      node.className = "velocity-card";
      const accent = index === 0 ? "ink-tertiary" : "ink-primary";
      const display = card.value === null || card.value === undefined ? "—" : escapeHtml(card.value);
      node.innerHTML = `
        <span class="metric-title">${escapeHtml(card.title || "")}</span>
        <span class="velocity-val mono ${accent}">${display}</span>
        <div style="display:flex;justify-content:space-between;align-items:center;gap:6px;">
          <span class="panel-sub">${escapeHtml(card.unit || "")}</span>
          <span class="metric-reason mono">${escapeHtml(cardReason(card))}</span>
        </div>`;
      host.appendChild(node);
    });
  }

  const multiHost = document.getElementById("multi-horizon-grid");
  if (multiHost) {
    const extra = [...groupCards(model, "latency"), ...groupCards(model, "session")];
    multiHost.innerHTML = "";
    extra.forEach((card) => {
      multiHost.appendChild(metricCellNode(card, "ink-primary"));
    });
  }

  const slots = cards.slice(0, 2);
  const first = slots[0] && hasSample(slots[0]) ? slots[0].series : null;
  const second = slots[1] && hasSample(slots[1]) ? slots[1].series : null;
  updateSparkline(first, second);
  renderTraceLegend(slots);
}

function hasSample(card) {
  return Array.isArray(card.series)
    && card.series.some((val) => val !== null && val !== undefined);
}

function renderTraceLegend(slots) {
  const host = document.getElementById("trace-legend");
  if (!host) return;
  host.innerHTML = "";
  const colors = ["#f0b429", "#388bfd"];
  const inks = ["ink-tertiary", "ink-primary"];
  slots.slice(0, 2).forEach((card, index) => {
    if (!card || !hasSample(card)) return;
    const node = document.createElement("span");
    node.className = inks[index];
    node.style.cssText = "display:inline-flex;align-items:center;gap:5px;font-weight:500;";
    node.innerHTML = `
      <span style="width:8px;height:8px;border-radius:999px;background:${colors[index]};display:inline-block;"></span>
      <span>${escapeHtml(card.title || "")}</span>`;
    host.appendChild(node);
  });
}

function renderActions(model) {
  const actions = model.actions || [];
  const byId = new Map(actions.map((a) => [a.id, a]));
  document.querySelectorAll("[data-action]").forEach((el) => {
    const action = byId.get(el.getAttribute("data-action"));
    el.hidden = !action;
    if (!action) return;
    el.disabled = !action.isEnabled;
    if (action.reason) el.title = action.reason;
    else el.removeAttribute("title");
    const label = el.querySelector("[data-action-label]");
    if (label && action.title) label.textContent = action.title;
  });
}

function renderLifecycle(model) {
  const lifecycle = model.lifecycle || {};
  const phase = lifecycle.phase || null;
  const detail = lifecycle.detail || null;
  const generation = model.generation || null;
  isConnected = true;
  updateStatusIndicators(phase, generation, detail);
  const dot = document.getElementById("vel-status-dot");
  if (dot) {
    dot.className = phase === "loaded" ? "dot dot-active"
      : phase === "failed" ? "dot dot-error"
      : "dot dot-idle";
  }
  const label = document.getElementById("vel-status-text");
  if (label) label.textContent = phase || "unknown";
  const liveText = document.getElementById("sparkline-live-text");
  if (liveText) {
    liveText.textContent = generation === "prefill" || generation === "decode" ? "Live" : "Idle";
  }
}

function knobNumber(knob) {
  if (!knob) return null;
  if (typeof knob === "number") return knob;
  for (const key of Object.keys(knob)) {
    const value = knob[key];
    if (typeof value === "number") return value;
  }
  return null;
}

function knobString(model, id) {
  const knob = (model.knobs || {})[id];
  if (!knob || typeof knob !== "object") return null;
  const value = knob.string;
  return typeof value === "string" ? value : null;
}

function cardGroup(card) {
  return String(card.id || "").split(".")[0];
}

function metricCellNode(card, accent) {
  const node = document.createElement("div");
  node.className = "metric-cell";
  const hasValue = !(card.value === null || card.value === undefined);
  const valHtml = hasValue
    ? `${escapeHtml(card.value)}${card.unit ? `<span class="metric-unit">${escapeHtml(card.unit)}</span>` : ""}`
    : "—";
  const percent = card.unit === "%" && typeof card.number === "number" ? card.number : null;
  const barColor = accent === "ink-secondary" ? "var(--active)" : (accent === "ink-tertiary" ? "var(--prefill)" : "var(--accent)");
  const reason = cardReason(card);
  const titleText = String(card.title || "");
  node.innerHTML = `
    <div class="metric-top">
      <span class="metric-title" title="${escapeHtml(titleText)}">${escapeHtml(titleText)}</span>
      <span class="metric-value mono ${hasValue ? accent : "ink-muted"}">${valHtml}</span>
    </div>
    <div class="metric-bar-slot">
      ${percent === null ? "" : `
      <div class="bar-bg">
        <div class="bar-fg" style="width:${Math.min(100, Math.max(0, percent))}%;background:${barColor};"></div>
      </div>`}
    </div>
    <span class="metric-reason mono" title="${escapeHtml(reason)}">${escapeHtml(reason)}</span>`;
  return node;
}

function groupCards(model, group) {
  return (model.cards || []).filter((c) => cardGroup(c) === group);
}

function renderHardwareCards(model) {
  const host = document.getElementById("hardware-cards");
  if (!host) return;
  const cards = [
    ...groupCards(model, "gpu"),
    ...groupCards(model, "hardware"),
    ...groupCards(model, "memory"),
    ...groupCards(model, "decode")
  ];
  host.innerHTML = "";
  for (const card of cards) {
    const grp = cardGroup(card);
    const tint = grp === "gpu" || grp === "decode" ? "ink-primary" : "ink-secondary";
    host.appendChild(metricCellNode(card, tint));
  }
}

function renderCacheSections(model) {
  const cards = groupCards(model, "cache");
  const host = document.getElementById("cache-cards");
  if (host) {
    host.innerHTML = "";
    for (const card of cards) {
      host.appendChild(metricCellNode(card, "ink-secondary"));
    }
  }
  for (const id of ["cache-section", "tiers-section"]) {
    const section = document.getElementById(id);
    if (section) section.hidden = cards.length === 0;
  }
  const reused = cards.find((c) => String(c.unit || "").toLowerCase() === "tok");
  const chip = document.getElementById("cache-hit-chip");
  if (chip) {
    chip.textContent = reused && reused.value !== null && reused.value !== undefined
      ? `${reused.value} ${reused.unit || ""} reused` : "—";
  }
}

function renderContext(model) {
  const card = (model.cards || []).find((c) => cardGroup(c) === "context");
  const capacity = knobNumber((model.knobs && model.knobs.contextTokens) || null);
  const active = card && typeof card.number === "number" ? card.number : null;
  const label = document.getElementById("ctx-tokens-label");
  if (label) {
    label.textContent = active === null ? "—"
      : `${formatTokenCount(active)}${capacity === null ? " tok" : ` / ${formatTokenCount(capacity)} tok`}`;
  }
  const chip = document.getElementById("ctx-used-chip");
  const radialLabel = document.getElementById("radial-ctx-label");
  const radial = document.getElementById("radial-ctx-path");
  const percent = active !== null && capacity ? (active / capacity) * 100 : null;
  const pctInt = percent === null ? null : Math.round(percent);
  if (chip) chip.textContent = pctInt === null ? "—" : `${pctInt}% Used`;
  if (radialLabel) radialLabel.textContent = pctInt === null ? "—" : `${pctInt}%`;
  const filled = percent === null ? 0 : Math.min(100, Math.max(0, percent));
  if (radial) radial.setAttribute("stroke-dasharray", `${filled}, 100`);
  const bar = document.getElementById("ctx-linear-bar");
  if (bar) bar.style.width = `${filled}%`;
}

function syncHeroTitleMarquee() {
  const viewport = document.querySelector("#hero-model-title .hero-model-marquee");
  if (!viewport) return;
  const track = viewport.querySelector(".hero-model-track");
  const primary = track ? track.querySelector(".hero-model-copy") : null;
  if (!track || !primary) return;

  const extraCopy = track.querySelector('.hero-model-copy[aria-hidden="true"]');
  const availableWidth = viewport.clientWidth;
  const textWidth = primary.getBoundingClientRect().width;
  if (availableWidth > 0 && textWidth > availableWidth + 2) {
    if (!extraCopy) {
      const clone = document.createElement("span");
      clone.className = "hero-model-copy";
      clone.setAttribute("aria-hidden", "true");
      clone.textContent = primary.textContent;
      track.appendChild(clone);
    }
    const shiftDistance = Math.ceil(textWidth + 32);
    const durationSec = Math.max(4.5, Math.round(((shiftDistance / 36) / 0.84) * 10) / 10);
    viewport.style.setProperty("--marquee-shift", `-${shiftDistance}px`);
    viewport.style.setProperty("--marquee-duration", `${durationSec}s`);
    viewport.classList.add("is-scrolling");
  } else {
    if (extraCopy) extraCopy.remove();
    viewport.style.removeProperty("--marquee-shift");
    viewport.style.removeProperty("--marquee-duration");
    viewport.classList.remove("is-scrolling");
  }
}

function renderFacts(model) {
  const facts = model.facts || {};
  const setText = (id, val) => {
    const el = document.getElementById(id);
    if (el) el.textContent = val;
  };
  setText("ftr-metal-version", facts.metalFamily || facts.gpuName || "—");
  setText("chip-cores", facts.hostLogicalCores ?? "—");
  setText("chip-uma", typeof facts.physicalMemoryGB === "number" ? `${Math.round(facts.physicalMemoryGB)} GiB` : "—");
  setText("chip-gdn", facts.gdnPrefillMode || "—");
  setText("chip-lsa", facts.lsaMode || "—");
  setText("chip-mempack", facts.mountedMemoryPack || "—");
  setText("hw-chip-badge", facts.gpuName || "—");

  const contextKnob = knobNumber((model.knobs || {}).contextTokens || null);
  setText("chip-context", contextKnob === null ? "—" : `${Math.round(contextKnob / 1024)}k`);
  setText("ftr-context-budget", contextKnob === null ? "Context: —" : `Context: ${formatTokenCount(contextKnob)} ctx`);

  const title = document.getElementById("hero-model-title");
  if (title) {
    if (model.modelId) {
      const cleanQuant = String(facts.quantization || "").replace(/^\((.*)\)$/, "$1");
      const sig = `${model.modelId}\n${cleanQuant}`;
      if (title.dataset.sig !== sig) {
        title.dataset.sig = sig;
        const modelHtml = `<span class="hero-model-marquee" title="${escapeHtml(model.modelId)}"><span class="hero-model-track"><span class="hero-model-copy">${escapeHtml(model.modelId)}</span></span></span>`;
        title.innerHTML = cleanQuant
          ? `${modelHtml}<span class="hero-quant-pill mono" title="${escapeHtml(cleanQuant)}">${escapeHtml(cleanQuant)}</span>`
          : modelHtml;
      }
      requestAnimationFrame(syncHeroTitleMarquee);
    } else {
      delete title.dataset.sig;
      title.textContent = "—";
    }
  }
  const badge = document.getElementById("hero-chip-badge");
  if (badge) {
    const parts = [facts.gpuName, facts.metalFamily].filter(Boolean);
    badge.textContent = parts.length ? parts.join(" · ") : "—";
  }

  const kv = knobString(model, "kvMode");
  setText("chip-kv-quant", kv ? kv.toUpperCase() : "—");
  const drafts = knobNumber((model.knobs || {}).mtpDrafts || null);
  const backend = knobString(model, "mtpBackend");
  setText("chip-mtp", drafts === null ? "—" : (drafts > 0 ? `${drafts} drafts${backend ? ` (${backend})` : ""}` : "Off"));
  const cacheDir = knobString(model, "cacheDir");
  setText("tier-cache-path", cacheDir ? `Cache Path: ${cacheDir}` : "Cache Path: —");

  const knobsHost = document.getElementById("knobs-provenance-row");
  const summaryBadge = document.getElementById("knobs-summary-badge");
  if (knobsHost) {
    const sources = model.knobSources || {};
    const knobs = model.knobs || {};
    const keys = Object.keys(knobs).sort();
    let overrideCount = 0;
    knobsHost.innerHTML = keys.map((key) => {
      const valObj = knobs[key];
      let valText = "—";
      if (valObj && typeof valObj === "object") {
        const firstKey = Object.keys(valObj)[0];
        if (firstKey) valText = String(valObj[firstKey]);
      } else if (valObj !== null && valObj !== undefined) {
        valText = String(valObj);
      }
      const src = sources[key];
      if (src && src !== "default") overrideCount += 1;
      const pairText = `${key}=${valText}`;
      return `<span class="knob-pill mono"><span class="knob-kv" title="${escapeHtml(pairText)}">${escapeHtml(pairText)}</span>${src ? `<span class="knob-src">${escapeHtml(src)}</span>` : ""}</span>`;
    }).join("");
    if (summaryBadge) {
      summaryBadge.textContent = keys.length === 0
        ? "—"
        : (overrideCount > 0 ? `${keys.length} params · ${overrideCount} custom` : `${keys.length} params`);
    }
  }
}

function renderModelAge(model) {
  const el = document.getElementById("model-age");
  if (!el) return;
  const sampledAt = model.sampledAt ? Date.parse(model.sampledAt) : NaN;
  if (Number.isNaN(sampledAt)) { el.textContent = "—"; return; }
  const age = Math.max(0, (Date.now() - sampledAt) / 1000);
  el.textContent = age < 1 ? "Updated just now" : `Updated ${age.toFixed(1)} s ago`;
}

function renderDisconnected() {
  isConnected = false;
  publishedModel = null;
  document.querySelectorAll("[data-card]").forEach((el) => { el.textContent = "—"; });
  document.querySelectorAll("[data-card-reason]").forEach((el) => { el.textContent = ""; });
  for (const id of ["velocity-cards", "multi-horizon-grid", "hardware-cards", "cache-cards", "knobs-provenance-row"]) {
    const host = document.getElementById(id);
    if (host) host.innerHTML = "";
  }
  const legend = document.getElementById("trace-legend");
  if (legend) legend.innerHTML = "";
  for (const suffix of ["a", "b"]) {
    for (const part of ["path", "area", "guide", "markers"]) {
      const path = document.getElementById(`sparkline-series-${suffix}-${part}`);
      if (path) path.setAttribute("d", "");
    }
    const dot = document.getElementById(`sparkline-series-${suffix}-dot`);
    if (dot) dot.classList.add("opacity-0");
  }
  for (const id of ["hero-model-title", "hero-chip-badge", "hw-chip-badge",
                    "chip-cores", "chip-uma", "chip-gdn", "chip-context",
                    "chip-kv-quant", "chip-mtp", "chip-lsa", "chip-mempack",
                    "knobs-summary-badge", "mem-dram-bw",
                    "ftr-metal-version", "cache-hit-chip", "ctx-tokens-label",
                    "ctx-used-chip", "radial-ctx-label"]) {
    const el = document.getElementById(id);
    if (el) el.textContent = "—";
    if (id === "hero-model-title" && el && el.dataset) delete el.dataset.sig;
  }
  const budget = document.getElementById("ftr-context-budget");
  if (budget) budget.textContent = "Context: —";
  const cachePath = document.getElementById("tier-cache-path");
  if (cachePath) cachePath.textContent = "Cache Path: —";
  const traceLive = document.getElementById("sparkline-live-text");
  if (traceLive) traceLive.textContent = "Not connected";
  const velText = document.getElementById("vel-status-text");
  if (velText) velText.textContent = "—";
  const velDot = document.getElementById("vel-status-dot");
  if (velDot) velDot.className = "dot dot-idle";
  if (prefillHoldTimer) {
    clearTimeout(prefillHoldTimer);
    prefillHoldTimer = null;
  }
  wasPrefilling = false;
  lastPrefillTargetValue = null;
  lastPrefillTargetText = "";
  const prefillBanner = document.getElementById("prefill-banner");
  if (prefillBanner) {
    prefillBanner.classList.remove("is-active", "is-complete");
    prefillBanner.removeAttribute("title");
  }
  const prefillTitle = document.getElementById("prefill-banner-title");
  if (prefillTitle) prefillTitle.textContent = "Prefill Progress";
  const prefillCountWrap = document.getElementById("prefill-count-wrap");
  if (prefillCountWrap) prefillCountWrap.hidden = false;
  const prefillCompleteCount = document.getElementById("prefill-complete-count");
  if (prefillCompleteCount) {
    prefillCompleteCount.hidden = true;
    prefillCompleteCount.textContent = "";
  }
  const prefillPct = document.getElementById("prefill-pct");
  if (prefillPct) prefillPct.textContent = "—";
  const prefillFill = document.getElementById("prefill-progress-fill");
  if (prefillFill) prefillFill.style.width = "0%";

  document.querySelectorAll("[data-action]").forEach((el) => {
    el.disabled = true;
    el.title = "No response from the server";
  });
  const radial = document.getElementById("radial-ctx-path");
  if (radial) radial.setAttribute("stroke-dasharray", "0, 100");
  const bar = document.getElementById("ctx-linear-bar");
  if (bar) bar.style.width = "0%";
  const age = document.getElementById("model-age");
  if (age) age.textContent = "Not connected";
  renderRecentRequestsTable();
  updateStatusIndicators(null, null, null);
  if (typeof syncChatHeaderWithDashboard === "function") syncChatHeaderWithDashboard();
  if (typeof renderDecisionTelemetryCard === "function") renderDecisionTelemetryCard(null);
}

function renderDashboard(model) {
  if (!model) return;
  modelGeneration += 1;
  publishedModel = model;
  renderLifecycle(model);
  renderCards(model);
  renderPrefillBanner(model);
  renderVelocityCards(model);
  renderHardwareCards(model);
  renderCacheSections(model);
  renderContext(model);
  renderFacts(model);
  renderModelAge(model);
  renderActions(model);
  renderRecentRequestsTable();
  if (typeof syncChatHeaderWithDashboard === "function") syncChatHeaderWithDashboard();
  if (typeof renderDecisionTelemetryCard === "function") renderDecisionTelemetryCard(model);
  if (typeof updateDecisionSidecarPanel === "function") updateDecisionSidecarPanel(model);
}

function toggleRequestInspector(requestId) {
  expandedRequestId = expandedRequestId === requestId ? null : requestId;
  renderRecentRequestsTable();
}

function renderRecentRequestsTable() {
  const tbody = document.getElementById("req-table-body");
  if (!tbody) return;
  const requests = (publishedModel && publishedModel.requests) || [];
  const filter = document.getElementById("req-filter-select")?.value || "all";
  const filtered = requests.filter((req) => {
    const status = typeof req.status === "number" ? req.status : null;
    const outcome = String(req.outcome || "").toLowerCase();
    if (filter === "success") return (status !== null && status >= 200 && status < 300) || outcome === "ok";
    if (filter === "error") return (status !== null && status >= 400) || (outcome !== "" && outcome !== "ok");
    return true;
  });

  if (filtered.length === 0) {
    tbody.innerHTML = `<tr><td colspan="9" class="ink-muted" style="text-align:center;padding:24px;">No requests have been recorded yet.</td></tr>`;
    return;
  }

  const newestFirst = filtered.slice(-64).reverse();
  tbody.innerHTML = newestFirst.map((req) => {
    const tierBadge = req.cacheHitTier === "warm_ram"
      ? `<span class="knob-pill mono ink-secondary">WARM RAM</span>`
      : (req.cacheHitTier === "cold_ssd"
          ? `<span class="knob-pill mono ink-primary">COLD SSD</span>`
          : `<span class="ink-muted">—</span>`);
    const isExpanded = expandedRequestId !== null && expandedRequestId === req.id;
    const mainRow = `
      <tr class="req-row" onclick="toggleRequestInspector('${escapeHtml(req.id || "")}')">
        <td class="mono panel-sub">${escapeHtml(req.time || "—")}</td>
        <td class="mono ink-primary">${escapeHtml(req.id || "—")}</td>
        <td>${escapeHtml(req.modelId || req.model || "—")}</td>
        <td class="mono">${typeof req.promptTokens === "number" ? formatTokenCount(req.promptTokens) : "—"} / ${typeof req.generatedTokens === "number" ? formatTokenCount(req.generatedTokens) : "—"}</td>
        <td class="mono">${typeof req.cachedTokens === "number" ? formatTokenCount(req.cachedTokens) : "—"}</td>
        <td class="mono ink-secondary">${typeof req.decodeTps === "number" ? `${req.decodeTps.toFixed(1)} t/s` : "—"}</td>
        <td class="mono">${typeof req.ttftMs === "number" ? `${Math.round(req.ttftMs)} ms` : "—"}</td>
        <td>${tierBadge}</td>
        <td style="text-align:right;" class="mono">${escapeHtml(req.outcome ?? req.status ?? "—")}</td>
      </tr>`;
    if (!isExpanded) return mainRow;

    const preMs = typeof req.prefillMs === "number" ? req.prefillMs : null;
    const decMs = typeof req.decodeMs === "number" ? req.decodeMs : null;
    const totalMs = (preMs ?? 0) + (decMs ?? 0);
    const prePct = totalMs > 0 && preMs !== null ? (preMs / totalMs) * 100 : 0;
    const decPct = totalMs > 0 && decMs !== null ? (decMs / totalMs) * 100 : 0;
    const stages = Object.entries(req.stageGPU || {}).sort((a, b) => b[1] - a[1]);
    const maxStage = stages.reduce((acc, pair) => Math.max(acc, pair[1]), 0.001);
    const stageBars = stages.length
      ? `<div class="stage-list">${stages.map(([stageName, ms]) => `
          <div class="stage-row">
            <div class="stage-meta mono"><span>${escapeHtml(stageName)}</span><span>${ms.toFixed(2)} ms</span></div>
            <div class="bar-bg"><div class="bar-fg" style="width:${Math.min(100, (ms / maxStage) * 100).toFixed(1)}%;background:var(--cool);"></div></div>
          </div>`).join("")}</div>`
      : `<span class="panel-sub">No per-stage GPU timings recorded for this turn.</span>`;
    const cfgPills = Object.entries(req.config || {}).map(([k, v]) => `<span class="knob-pill mono">${escapeHtml(k)}=${escapeHtml(v)}</span>`).join("");

    return mainRow + `
      <tr class="inspector-row">
        <td colspan="9">
          <div style="display:flex;justify-content:space-between;flex-wrap:wrap;gap:12px;">
            <span class="mono">Prefill: ${preMs === null ? "—" : `${preMs.toFixed(1)} ms`} · Decode: ${decMs === null ? "—" : `${decMs.toFixed(1)} ms`}</span>
            <div class="knobs-row" style="margin-top:0;">${cfgPills}</div>
          </div>
          <div class="waterfall-bar">
            <div style="width:${prePct.toFixed(1)}%;background:var(--prefill);"></div>
            <div style="width:${decPct.toFixed(1)}%;background:var(--accent);"></div>
          </div>
          ${stageBars}
        </td>
      </tr>`;
  }).join("");
}

function exportRequestsJson() {
  const requests = (publishedModel && publishedModel.requests) || [];
  const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(requests, null, 2));
  const a = document.createElement("a");
  a.href = dataStr;
  a.download = `qwegine-requests-${Date.now()}.json`;
  a.click();
}
