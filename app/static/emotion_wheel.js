(() => {
  const EMOTION_ORDER = [
    "joy",
    "anticipation",
    "anger",
    "disgust",
    "sadness",
    "surprise",
    "fear",
    "trust",
  ];

  const CANONICAL_ANGLES = {
    joy: 0,
    anticipation: 45,
    anger: 90,
    disgust: 135,
    sadness: 180,
    surprise: 225,
    fear: 270,
    trust: 315,
  };

  const EMOTION_LABELS_KO = {
    joy: "기쁨",
    trust: "신뢰",
    fear: "두려움",
    surprise: "놀람",
    sadness: "슬픔",
    disgust: "혐오",
    anger: "분노",
    anticipation: "기대",
  };

  const clamp = (value, min, max) => Math.max(min, Math.min(max, value));
  const formatScore = (value) => {
    const parsed = Number(value);
    if (!Number.isFinite(parsed)) return "-";
    return parsed.toFixed(2);
  };

  const canonicalToSvgRad = (canonicalDeg) => ((canonicalDeg - 90) * Math.PI) / 180;

  const toPoint = (canonicalDeg, ratio, cx, cy, maxR) => {
    const rad = canonicalToSvgRad(canonicalDeg);
    const r = clamp(ratio, 0, 1) * maxR;
    return {
      x: cx + Math.cos(rad) * r,
      y: cy + Math.sin(rad) * r,
    };
  };

  const setText = (id, value) => {
    if (!id) return;
    const el = document.getElementById(id);
    if (!el) return;
    el.textContent = value;
  };

  const setBand = (id, band) => {
    if (!id) return;
    const el = document.getElementById(id);
    if (!el) return;
    el.textContent = band;
    const key = String(band || "");
    const cls = key === "높음" ? "band-high" : key === "보통" ? "band-mid" : key === "낮음" ? "band-low" : "band-na";
    el.className = `alignment-badge ${cls}`;
  };

  const drawWheel = (svg, profile, energy, labels) => {
    const cx = 180;
    const cy = 180;
    const maxR = 130;

    const spokes = EMOTION_ORDER.map((name) => {
      const outer = toPoint(CANONICAL_ANGLES[name], 1, cx, cy, maxR);
      const label = labels[name] || EMOTION_LABELS_KO[name] || name;
      return `
        <line x1="${cx}" y1="${cy}" x2="${outer.x.toFixed(2)}" y2="${outer.y.toFixed(2)}" stroke="#d8cdbd" stroke-width="1"></line>
        <text x="${outer.x.toFixed(2)}" y="${outer.y.toFixed(2)}" fill="#5b5144" font-size="12" text-anchor="middle" dominant-baseline="middle">${label}</text>
      `;
    }).join("");

    const polygon = EMOTION_ORDER.map((name) => {
      const ratio = clamp(Number(profile[name] || 0), 0, 1);
      const point = toPoint(CANONICAL_ANGLES[name], ratio, cx, cy, maxR);
      return `${point.x.toFixed(2)},${point.y.toFixed(2)}`;
    }).join(" ");

    const angleReliable = Boolean(energy.angle_reliable);
    const magnitude = Number(energy.magnitude || 0);
    const coherence = clamp(Number(energy.coherence || 0), 0, 1);
    const arrowRatio = clamp(coherence > 0 ? coherence : (magnitude > 0 ? 0.7 : 0), 0, 1);
    let arrow = "";
    if (angleReliable && magnitude > 0) {
      const angle = Number(energy.angle || 0);
      const tip = toPoint(angle, arrowRatio, cx, cy, maxR * 0.92);
      arrow = `
        <defs>
          <marker id="wheel-arrow-head" markerWidth="8" markerHeight="8" refX="6" refY="3.5" orient="auto">
            <polygon points="0 0, 7 3.5, 0 7" fill="#8e2f2f"></polygon>
          </marker>
        </defs>
        <line x1="${cx}" y1="${cy}" x2="${tip.x.toFixed(2)}" y2="${tip.y.toFixed(2)}"
              stroke="#8e2f2f" stroke-width="2.4" marker-end="url(#wheel-arrow-head)"></line>
      `;
    }

    svg.innerHTML = `
      <circle cx="${cx}" cy="${cy}" r="${maxR}" fill="#faf5ec" stroke="#d8cdbd"></circle>
      <circle cx="${cx}" cy="${cy}" r="${(maxR * 0.66).toFixed(2)}" fill="none" stroke="#e6dccf"></circle>
      <circle cx="${cx}" cy="${cy}" r="${(maxR * 0.33).toFixed(2)}" fill="none" stroke="#eee4d8"></circle>
      ${spokes}
      <polygon points="${polygon}" fill="rgba(39,93,78,0.25)" stroke="#275d4e" stroke-width="2"></polygon>
      ${arrow}
    `;
  };

  const renderEmotionWheel = (opts = {}) => {
    const svgId = opts.svgId || "emotion-wheel-svg";
    const dataId = opts.dataId || "emotion-wheel-data";
    const svg = document.getElementById(svgId);
    const dataScript = document.getElementById(dataId);
    if (!svg || !dataScript) return;

    let payload = {};
    try {
      payload = JSON.parse(dataScript.textContent || "{}");
    } catch (err) {
      payload = {};
    }

    const profileRaw = payload.profile && typeof payload.profile === "object" ? payload.profile : payload;
    const profile = {};
    EMOTION_ORDER.forEach((name) => {
      profile[name] = clamp(Number(profileRaw[name] || 0), 0, 1);
    });

    const energy = payload.energy && typeof payload.energy === "object" ? payload.energy : {};
    const alignment = payload.alignment && typeof payload.alignment === "object" ? payload.alignment : {};
    const labelsRaw = payload.labels_ko && typeof payload.labels_ko === "object" ? payload.labels_ko : {};
    const labels = {};
    EMOTION_ORDER.forEach((name) => {
      labels[name] = typeof labelsRaw[name] === "string" && labelsRaw[name].trim() ? labelsRaw[name] : EMOTION_LABELS_KO[name] || name;
    });
    drawWheel(svg, profile, energy, labels);

    if (opts.legendId) {
      const legend = document.getElementById(opts.legendId);
      if (legend) {
        legend.innerHTML = EMOTION_ORDER.map((name) => {
          const label = labels[name] || EMOTION_LABELS_KO[name] || name;
          return `<div class="badge">${label}: ${formatScore(profile[name])}</div>`;
        }).join("");
      }
    }

    const score = alignment.consistency_score;
    setText(opts.scoreId, score === null || score === undefined ? "-" : String(score));
    setBand(opts.bandId, alignment.consistency_band || "판단불가");

    const angleReliable = Boolean(energy.angle_reliable);
    const reliableMessage = angleReliable ? "방향 해석 가능" : "방향 해석 불가(상쇄 상태)";
    setText(opts.reliabilityId, reliableMessage);
  };

  window.renderEmotionWheel = renderEmotionWheel;
})();
