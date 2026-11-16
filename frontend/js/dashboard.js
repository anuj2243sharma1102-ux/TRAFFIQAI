// Dashboard: asks the backend for the latest numbers every second and updates the page.
// No libraries are used.

const TYPE_COLORS = {
  car: "#00c800", motorcycle: "#ffa500", bus: "#0078ff", truck: "#dc0000",
  auto: "#ffff00", taxi: "#ff1493", tempo: "#9600c8",
};
const RULE_NAMES = { wrong_way: "Wrong-way vehicles", stopped: "Stopped vehicles", red_light: "Red-light violations" };

// These numbers must match TrafficDensity in detection/traffic_density.py
const OCC_MAX = 0.40;    // top of the gauge and of the chart
const LOW_MAX = 0.13;
const HIGH_MIN = 0.20;
const MAX_POINTS = 120;  // 2 minutes of history (one point per second)

const $ = (id) => document.getElementById(id);
const history = [];
let simulatedNow = false;

$("markLow").style.left = (LOW_MAX / OCC_MAX * 100) + "%";
$("markHigh").style.left = (HIGH_MIN / OCC_MAX * 100) + "%";

function setConnection(kind, text) {
  const badge = $("conn");
  badge.className = "badge badge-" + kind;
  badge.textContent = text;
}

function renderVehicles(v) {
  $("vTotal").textContent = v.total;
  $("vLive").textContent = `${v.live} (avg ${v.smoothed})`;
  $("vDir").textContent = `${v.up} up / ${v.down} down`;

  const entries = Object.entries(v.by_type).sort((a, b) => b[1] - a[1]);
  const max = entries.length ? entries[0][1] : 1;
  const box = $("typeBars");
  box.innerHTML = "";
  entries.forEach(([name, count]) => {
    const row = document.createElement("div");
    row.className = "bar-row";
    const label = document.createElement("span");
    label.textContent = name;
    const bar = document.createElement("div");
    bar.className = "bar";
    const fill = document.createElement("i");
    fill.style.width = (count / max * 100) + "%";
    fill.style.background = TYPE_COLORS[name] || "#8b98a8";
    bar.appendChild(fill);
    const number = document.createElement("b");
    number.textContent = count;
    row.append(label, bar, number);
    box.appendChild(row);
  });
}

function renderDensity(d) {
  const chip = $("dStatus");
  chip.textContent = d.status;
  chip.className = "chip chip-" + d.status.toLowerCase();

  const fill = $("gFill");
  fill.style.width = Math.min(100, d.smoothed / OCC_MAX * 100) + "%";
  fill.style.background =
    d.status === "High" ? "var(--red)" : d.status === "Medium" ? "var(--amber)" : "var(--green)";

  $("dValue").textContent = `Road occupancy ${(d.smoothed * 100).toFixed(0)}%`;
  $("dRoi").textContent = d.in_roi;
}

function renderSignal(sg) {
  $("lampR").className = sg.phase === "RED" ? "on-red" : "";
  $("lampY").className = sg.phase === "YELLOW" ? "on-yellow" : "";
  $("lampG").className = sg.phase === "GREEN" ? "on-green" : "";

  const known = ["RED", "YELLOW", "GREEN"].includes(sg.phase);
  $("sCount").textContent = known ? Math.ceil(sg.time_left) + " s" : "--";
  $("sPhase").textContent = known ? sg.phase + " phase" : "waiting ...";
  $("sNext").textContent = known ? sg.next_green + " s" : "--";

  const tag = $("sPriority");
  if (sg.priority_note) {
    tag.textContent = sg.priority_note;
    tag.classList.remove("hidden");
  } else {
    tag.classList.add("hidden");
  }
  $("sScale").textContent = sg.time_scale !== 1 ? `Demo speed: signal clock x${sg.time_scale}` : "";
}

function renderEmergency(e, sg) {
  simulatedNow = e.simulated;

  const state = $("emState");
  if (e.active) {
    state.className = "state state-alert";
    state.textContent = e.detected.length
      ? "Emergency vehicle detected - priority active"
      : "Emergency vehicle (simulated) - priority active";
  } else {
    state.className = "state state-ok";
    state.textContent = "No emergency vehicle";
  }

  const list = $("emList");
  list.innerHTML = "";
  e.detected.forEach((d) => {
    const li = document.createElement("li");
    li.textContent = `${d.label} (${Math.round(d.score * 100)}%)`;
    list.appendChild(li);
  });

  $("emergencyBanner").classList.toggle("hidden", !e.active);
  $("bannerText").textContent = sg.priority_note ? " - " + sg.priority_note : " - signal priority requested";

  const button = $("simBtn");
  button.textContent = e.simulated ? "Clear simulated emergency" : "Simulate emergency vehicle";
  button.classList.toggle("active", e.simulated);
}

function renderViolations(v) {
  $("vioTotal").textContent = v.total;
  const parts = Object.entries(v.by_type).map(([rule, n]) => `${RULE_NAMES[rule] || rule}: ${n}`);
  $("vioList").textContent = parts.length ? parts.join(" | ") : "No violations detected";
}

function renderEvents(list) {
  const ul = $("events");
  ul.innerHTML = "";
  if (!list.length) {
    const li = document.createElement("li");
    li.textContent = "No alerts yet";
    ul.appendChild(li);
    return;
  }
  list.forEach((ev) => {
    const li = document.createElement("li");
    li.className = "ev-" + ev.type;
    const dot = document.createElement("span");
    dot.className = "ev-dot";
    const time = document.createElement("span");
    time.className = "ev-time";
    time.textContent = ev.wall_time;
    const text = document.createElement("span");
    text.textContent = ev.text;
    li.append(dot, time, text);
    ul.appendChild(li);
  });
}

function drawChart() {
  const canvas = $("chart");
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  if (!w || !h) return;
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);

  const pad = { l: 40, r: 10, t: 10, b: 22 };
  const plotW = w - pad.l - pad.r;
  const plotH = h - pad.t - pad.b;
  const y = (v) => pad.t + plotH * (1 - Math.min(v, OCC_MAX) / OCC_MAX);

  // coloured zones: Low / Medium / High
  const zones = [
    [0, LOW_MAX, "rgba(46,204,113,0.10)"],
    [LOW_MAX, HIGH_MIN, "rgba(245,166,35,0.10)"],
    [HIGH_MIN, OCC_MAX, "rgba(255,77,79,0.10)"],
  ];
  zones.forEach(([from, to, color]) => {
    ctx.fillStyle = color;
    ctx.fillRect(pad.l, y(to), plotW, y(from) - y(to));
  });

  // grid lines and labels
  ctx.strokeStyle = "#2a3340";
  ctx.fillStyle = "#8b98a8";
  ctx.font = "12px Segoe UI, Arial";
  ctx.lineWidth = 1;
  for (let v = 0; v <= OCC_MAX + 1e-9; v += 0.10) {
    ctx.beginPath();
    ctx.moveTo(pad.l, y(v));
    ctx.lineTo(w - pad.r, y(v));
    ctx.stroke();
    ctx.fillText(Math.round(v * 100) + "%", 4, y(v) + 4);
  }
  ctx.textAlign = "left";
  ctx.fillText("older", pad.l, h - 6);
  ctx.textAlign = "right";
  ctx.fillText("now", w - pad.r, h - 6);

  // the density line
  if (history.length < 2) return;
  ctx.beginPath();
  ctx.strokeStyle = "#3b82f6";
  ctx.lineWidth = 2;
  const offset = MAX_POINTS - history.length;
  history.forEach((value, i) => {
    const px = pad.l + plotW * ((i + offset) / (MAX_POINTS - 1));
    if (i === 0) ctx.moveTo(px, y(value));
    else ctx.lineTo(px, y(value));
  });
  ctx.stroke();
}

function renderStatus(s) {
  if (s.status === "running") setConnection("live", "Live");
  else if (s.status === "loading" || s.status === "starting") setConnection("wait", s.message || "Loading ...");
  else setConnection("error", s.message || s.status);

  $("fps").textContent = s.video.fps.toFixed(1) + " FPS";
  $("videoTime").textContent = s.video.name
    ? `${s.video.name} - ${s.video.time.toFixed(0)} s - round ${s.video.loops + 1}` : "";

  renderVehicles(s.vehicles);
  renderDensity(s.density);
  renderSignal(s.signal);
  renderEmergency(s.emergency, s.signal);
  renderViolations(s.violations);

  if (s.status === "running") {
    history.push(s.density.smoothed);
    if (history.length > MAX_POINTS) history.shift();
    drawChart();
  }
}

async function refreshStatus() {
  try {
    const response = await fetch("/api/status", { cache: "no-store" });
    renderStatus(await response.json());
  } catch (error) {
    setConnection("error", "Server not reachable");
  }
}

async function refreshEvents() {
  try {
    const response = await fetch("/api/events?limit=15", { cache: "no-store" });
    renderEvents(await response.json());
  } catch (error) {
    /* the status request already shows that the server is not reachable */
  }
}

$("simBtn").addEventListener("click", async () => {
  try {
    await fetch("/api/emergency", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ active: !simulatedNow }),
    });
  } catch (error) { /* ignored */ }
  refreshStatus();
  refreshEvents();
});

window.addEventListener("resize", drawChart);
setInterval(refreshStatus, 1000);
setInterval(refreshEvents, 2000);
refreshStatus();
refreshEvents();

// ---------- history and statistics (read from the database) ----------
const STATUS_COLORS = { Low: "#2ecc71", Medium: "#f5a623", High: "#ff4d4f" };

function renderSummary(s) {
  $("hSamples").textContent = s.samples;
  $("hCounted").textContent = s.vehicles_counted;
  $("hAvgOcc").textContent = Math.round(s.avg_occupancy * 100) + "%";
  $("hPeak").textContent = s.peak_in_view;
  $("hEvents").textContent = Object.values(s.events).reduce((a, b) => a + b, 0);

  const share = $("hShare");
  const legend = $("hLegend");
  share.innerHTML = "";
  legend.innerHTML = "";
  ["Low", "Medium", "High"].forEach((name) => {
    const fraction = s.status_share[name] || 0;
    const segment = document.createElement("i");
    segment.style.width = (fraction * 100) + "%";
    segment.style.background = STATUS_COLORS[name];
    share.appendChild(segment);
    const item = document.createElement("span");
    item.textContent = `${name}: ${Math.round(fraction * 100)}% of the time`;
    item.style.color = STATUS_COLORS[name];
    legend.appendChild(item);
  });
}

function renderRows(rows) {
  const body = document.querySelector("#hTable tbody");
  body.innerHTML = "";
  rows.slice().reverse().forEach((r) => {          // newest first
    const tr = document.createElement("tr");
    const cells = [
      r.ts.slice(11),
      r.vehicles_live,
      r.vehicles_total,
      `${r.density_status} (${Math.round(r.occupancy * 100)}%)`,
      `${r.signal_phase} ${Math.ceil(r.signal_time_left)} s`,
      r.emergency_active ? "YES" : "-",
    ];
    cells.forEach((value) => {
      const td = document.createElement("td");
      td.textContent = value;
      tr.appendChild(td);
    });
    body.appendChild(tr);
  });
}

async function refreshHistory() {
  try {
    const [summary, rows] = await Promise.all([
      fetch("/api/summary", { cache: "no-store" }).then((r) => r.json()),
      fetch("/api/history?limit=8", { cache: "no-store" }).then((r) => r.json()),
    ]);
    renderSummary(summary);
    renderRows(rows);
  } catch (error) {
    /* the status request already shows that the server is not reachable */
  }
}

setInterval(refreshHistory, 10000);
refreshHistory();

// ---------- demo e-challans (simulation) ----------
const VIOLATION_TITLES = {
  wrong_way: "Wrong-way driving",
  stopped: "Stopped on the carriageway",
  red_light: "Red-light violation",
  over_speed: "Over-speeding",
  no_helmet: "No helmet",
  seat_belt: "No seat belt",
  mobile_phone: "Mobile phone use",
};

function makeCell(text) {
  const td = document.createElement("td");
  td.textContent = text;
  return td;
}

function makeLink(text, href) {
  const a = document.createElement("a");
  a.textContent = text;
  a.href = href;
  a.target = "_blank";
  return a;
}

function renderChallans(rows) {
  const body = document.querySelector("#cTable tbody");
  body.innerHTML = "";
  if (!rows.length) {
    const tr = document.createElement("tr");
    const td = makeCell("No demo challans yet. Wait for a detection or press 'Simulate violation'.");
    td.colSpan = 7;
    tr.appendChild(td);
    body.appendChild(tr);
    return;
  }
  rows.forEach((r) => {
    const tr = document.createElement("tr");
    tr.append(
      makeCell(r.challan_no || "..."),
      makeCell(r.ts.slice(11)),
      makeCell(VIOLATION_TITLES[r.violation] || r.violation),
    );
    const source = document.createElement("td");
    const tag = document.createElement("span");
    tag.className = "src src-" + r.source;
    tag.textContent = r.source.toUpperCase();
    source.appendChild(tag);
    tr.appendChild(source);
    tr.append(makeCell(`${r.plate} (${r.vehicle_class})`), makeCell("Rs " + r.fine + " (sample)"));
    const files = document.createElement("td");
    if (r.evidence_file) {
      files.append(makeLink("Evidence", "/challan_files/" + r.evidence_file), " ",
                   makeLink("PDF", "/challan_files/" + r.pdf_file));
    } else {
      files.textContent = "creating ...";
    }
    tr.appendChild(files);
    body.appendChild(tr);
  });
}

async function refreshChallans() {
  try {
    const response = await fetch("/api/challans?limit=8", { cache: "no-store" });
    renderChallans(await response.json());
  } catch (error) {
    /* the status request already shows that the server is not reachable */
  }
}

$("simViolationBtn").addEventListener("click", async () => {
  try {
    await fetch("/api/simulate_violation", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ type: $("simType").value }),
    });
  } catch (error) { /* ignored */ }
  refreshEvents();
  setTimeout(refreshChallans, 1500);
  setTimeout(refreshChallans, 4000);
});

setInterval(refreshChallans, 3000);
refreshChallans();