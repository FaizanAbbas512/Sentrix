// ============================================================
//  SENTRIX dashboard — live updates
//  /api/stats aur /api/alerts poll karke UI refresh
// ============================================================

const $ = (id) => document.getElementById(id);

function fmtClock() {
  return new Date().toLocaleTimeString("en-GB");
}

const THREAT_META = {
  NONE: "System nominal",
  LOW: "Minor activity",
  MEDIUM: "Review feed",
  HIGH: "Immediate attention",
};

async function pollStats() {
  try {
    const s = await (await fetch("/api/stats")).json();

    $("v-people").textContent = s.people;
    $("v-loiter").textContent = s.loitering;
    $("v-abandon").textContent = s.abandoned;
    $("v-fall").textContent = s.fall;
    $("v-fight").textContent = s.fight;
    $("v-crowd").textContent = s.crowded ? "YES" : "—";
    $("t-fps").textContent = s.fps + " FPS";

    // card highlights
    $("c-loiter").classList.toggle("active", s.loitering > 0);
    $("c-abandon").classList.toggle("active", s.abandoned > 0);
    $("c-fall").classList.toggle("active", s.fall > 0);
    $("c-fight").classList.toggle("active", s.fight > 0);
    $("c-crowd").classList.toggle("warn", s.crowded);

    // THREAT hero (fusion ka intelligent faisla)
    const lvl = (s.threat || "NONE").toLowerCase();
    const hero = $("threat-hero");
    hero.className = "threat-hero " + (lvl === "none" ? "" : lvl);
    $("threat-level").textContent = s.threat || "NONE";
    $("threat-meta").textContent =
      (THREAT_META[s.threat] || "") + "  ·  score " + (s.threat_score || 0);
  } catch (e) {
    $("threat-meta").textContent = "CONNECTION LOST";
  }
}

async function pollAlerts() {
  try {
    const events = await (await fetch("/api/alerts")).json();
    const list = $("alert-list");
    $("alert-count").textContent = events.length;

    if (!events.length) {
      list.innerHTML =
        '<li class="empty">Koi event nahi — feed monitor ho rahi hai.</li>';
      return;
    }
    list.innerHTML = events
      .map(
        (e) => `
      <li class="alert-item">
        <span class="badge ${e.type}">${e.type}</span>
        <div class="alert-detail">
          <b>${e.type === "CROWD" ? "Crowd" : "ID #" + e.track_id}</b>
          <span>${e.detail}</span>
        </div>
        <span class="alert-time">${e.timestamp.split(" ")[1]}</span>
      </li>`
      )
      .join("");
  } catch (e) {}
}

setInterval(() => ($("t-clock").textContent = fmtClock()), 1000);
setInterval(pollStats, 1000);
setInterval(pollAlerts, 1500);
$("t-clock").textContent = fmtClock();
pollStats();
pollAlerts();
