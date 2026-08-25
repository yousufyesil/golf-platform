async function api(path, options = {}) {
  const res = await fetch(path, options);
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail || JSON.stringify(body);
    } catch (e) { /* keep statusText */ }
    throw new Error(detail);
  }
  const contentType = res.headers.get("content-type") || "";
  return contentType.includes("application/json") ? res.json() : res.text();
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
}

function formatPercent(x) {
  return `${(x * 100).toFixed(1)}%`;
}

/** Grobe relative Zeitangabe ("vor 4 Min.", "vor 2 Tagen") für Zeitstempel wie store.built_at. */
function formatRelativeTime(iso) {
  if (!iso) return "–";
  const diffMs = Date.now() - new Date(iso).getTime();
  const min = Math.round(diffMs / 60000);
  if (min < 1) return "gerade eben";
  if (min < 60) return `vor ${min} Min.`;
  const h = Math.round(min / 60);
  if (h < 24) return `vor ${h} Std.`;
  const d = Math.round(h / 24);
  return `vor ${d} Tag${d === 1 ? "" : "en"}`;
}

const RECENT_JUDGMENTS_KEY = "golf_recent_judgments";
const RECENT_JUDGMENTS_MAX = 8;

/** Merkt ein Klassifikations-Urteil clientseitig (für "Letzte Urteile" im Dashboard). */
function pushRecentJudgment({ pred, conf }) {
  const list = getRecentJudgments();
  list.push({ pred, conf, ts: Date.now() });
  while (list.length > RECENT_JUDGMENTS_MAX) list.shift();
  localStorage.setItem(RECENT_JUDGMENTS_KEY, JSON.stringify(list));
}

function getRecentJudgments() {
  try {
    return JSON.parse(localStorage.getItem(RECENT_JUDGMENTS_KEY) || "[]");
  } catch (e) {
    return [];
  }
}

/**
 * Rendert eine sortierte Konfidenz-Balkenliste (ein Hue, Magnitude via Balkenlänge).
 * probDict: {klasse: wahrscheinlichkeit(0..1)}. Die Top-Klasse wird hervorgehoben
 * (Fettschrift + Badge), Farbe allein trägt keine Bedeutung.
 */
function renderBars(container, probDict, { top = null, predicted = null } = {}) {
  let entries = Object.entries(probDict).sort((a, b) => b[1] - a[1]);
  if (top) entries = entries.slice(0, top);
  const topClass = predicted || (entries.length ? entries[0][0] : null);

  container.innerHTML = entries.map(([cls, p]) => {
    const isTop = cls === topClass;
    return `
      <div class="bar-row ${isTop ? "is-top" : ""}">
        <div class="bar-label">${escapeHtml(cls)}${isTop ? ' <span class="badge good">&#10003;</span>' : ""}</div>
        <div class="bar-track"><div class="bar-fill" style="width:${(p * 100).toFixed(1)}%"></div></div>
        <div class="bar-value">${formatPercent(p)}</div>
      </div>`;
  }).join("");
}

/**
 * Rendert die k einbezogenen Nachbar-Bilder eines Inferenz-Ergebnisses (Datei,
 * Klasse, Ähnlichkeit) als kleines Thumbnail-Grid für die "Erweitert"-Ansicht.
 */
function renderNeighbors(container, neighbors) {
  if (!neighbors || !neighbors.length) {
    container.innerHTML = `<p class="empty-hint">Keine Nachbarn verfügbar.</p>`;
    return;
  }
  container.innerHTML = neighbors.map(n => {
    const slash = n.file.lastIndexOf("/");
    const klasse = n.file.substring(0, slash);
    const filename = n.file.substring(slash + 1);
    const src = `/api/classes/${encodeURIComponent(klasse)}/images/${encodeURIComponent(filename)}/file`;
    return `
      <figure class="neighbor-tile">
        <img src="${src}" loading="lazy">
        <figcaption>
          <span class="neighbor-class" title="${escapeHtml(n.class)}">${escapeHtml(n.class)}</span>
          <span class="neighbor-score">${formatPercent(n.similarity)}</span>
        </figcaption>
      </figure>`;
  }).join("");
}

/**
 * Wie renderNeighbors, aber nach Klasse gruppiert (Titelzeile: Klasse, Anzahl,
 * Ø-Ähnlichkeit; Bilder je Gruppe nach Ähnlichkeit sortiert) statt einer flachen
 * Liste — macht auf einen Blick sichtbar, wie eindeutig eine Klasse dominiert vs.
 * knapp gegen eine zweite steht. Zeigt je Bild die Perspektive statt der Klasse
 * in der Bildunterschrift, da die Klasse bereits der Gruppentitel ist.
 */
function renderNeighborGroups(container, neighbors) {
  if (!neighbors || !neighbors.length) {
    container.innerHTML = `<p class="empty-hint">Keine Nachbarn verfügbar.</p>`;
    return;
  }
  const groups = new Map();
  for (const n of neighbors) {
    if (!groups.has(n.class)) groups.set(n.class, []);
    groups.get(n.class).push(n);
  }
  const avgOf = (items) => items.reduce((s, n) => s + n.similarity, 0) / items.length;
  const ordered = [...groups.entries()].sort((a, b) => avgOf(b[1]) - avgOf(a[1]));

  container.innerHTML = ordered.map(([cls, items]) => {
    const sorted = [...items].sort((a, b) => b.similarity - a.similarity);
    return `
      <div class="neighbor-group">
        <div class="neighbor-group-title">
          <span>${escapeHtml(cls)}</span>
          <span class="meta">${items.length} Bild${items.length === 1 ? "" : "er"} · Ø ${formatPercent(avgOf(items))}</span>
        </div>
        <div class="neighbor-grid">
          ${sorted.map(n => {
            const slash = n.file.lastIndexOf("/");
            const klasse = n.file.substring(0, slash);
            const filename = n.file.substring(slash + 1);
            const src = `/api/classes/${encodeURIComponent(klasse)}/images/${encodeURIComponent(filename)}/file`;
            return `
              <figure class="neighbor-tile">
                <img src="${src}" loading="lazy">
                <figcaption>
                  <span class="neighbor-class" title="${escapeHtml(n.perspektive || "")}">${escapeHtml(n.perspektive || "–")}</span>
                  <span class="neighbor-score">${formatPercent(n.similarity)}</span>
                </figcaption>
              </figure>`;
          }).join("")}
        </div>
      </div>`;
  }).join("");
}

/**
 * Lädt die Attention-Heatmap für ein Bild verzögert nach (erst wenn die
 * "Erweitert"-Ansicht tatsächlich geöffnet wird, nicht vorab für jedes
 * Ergebnis) und cached das Resultat am img-Element. formFields liefert entweder
 * { file: File } (lokal hochgeladenes Bild) oder { image_url: "..." }
 * (Kleinanzeigen-Bild) an POST /api/heatmap.
 */
async function loadHeatmap(imgEl, statusEl, formFields) {
  if (imgEl.dataset.loaded) return;
  const fd = new FormData();
  for (const [k, v] of Object.entries(formFields)) {
    if (v) fd.append(k, v);
  }
  if (![...fd.keys()].length) {
    statusEl.textContent = "Original-Bild nicht mehr verfügbar.";
    return;
  }

  statusEl.textContent = "Berechne Heatmap…";
  try {
    const res = await fetch("/api/heatmap", { method: "POST", body: fd });
    if (!res.ok) {
      let detail = res.statusText;
      try { detail = (await res.json()).detail || detail; } catch (e) { /* keep statusText */ }
      throw new Error(detail);
    }
    const blob = await res.blob();
    imgEl.src = URL.createObjectURL(blob);
    imgEl.dataset.loaded = "1";
    statusEl.textContent = "";
  } catch (err) {
    statusEl.textContent = err.message;
  }
}

/**
 * Verkabelt einen "Erweitert"-Button mit dem dazugehörigen Panel (Umschalten
 * von Sichtbarkeit + Button-Beschriftung). onFirstExpand (optional) feuert
 * einmalig beim ersten Öffnen, z.B. um die Heatmap erst dann nachzuladen.
 */
function wireExpandToggle(btn, panel, onFirstExpand) {
  let expanded = false;
  btn.addEventListener("click", () => {
    const hidden = panel.classList.toggle("hidden");
    btn.textContent = hidden ? "Erweitert ▾" : "Erweitert ▴";
    if (!hidden && !expanded) {
      expanded = true;
      if (onFirstExpand) onFirstExpand();
    }
  });
}

/**
 * Verkabelt ein .dropzone-Element (Klick öffnet Dateiauswahl, Drag&Drop
 * funktioniert ebenso) mit einem callback(files: FileList).
 */
function setupDropzone(el, onFiles) {
  const input = el.querySelector('input[type="file"]');
  el.addEventListener("click", () => input.click());
  input.addEventListener("change", () => {
    if (input.files.length) onFiles(input.files);
  });
  ["dragenter", "dragover"].forEach(evt => el.addEventListener(evt, (e) => {
    e.preventDefault();
    el.classList.add("is-dragover");
  }));
  ["dragleave", "drop"].forEach(evt => el.addEventListener(evt, (e) => {
    e.preventDefault();
    el.classList.remove("is-dragover");
  }));
  el.addEventListener("drop", (e) => {
    const files = e.dataTransfer.files;
    if (files.length) onFiles(files);
  });
}
