// ============================================================
//  SENTRIX dashboard — live updates
//  Har second /api/stats aur /api/alerts poll karke UI refresh
// ============================================================

const $ = (id) => document.getElementById(id);

function fmtClock() {
  const d = new Date();
  return d.toLocaleTimeString("en-GB"); // HH:MM:SS
}

async function pollStats() {
  try {
    const r = await fetch("/api/stats");
    const s = await r.json();

    $("v-people").textContent = s.people;
    $("v-loiter").textContent = s.loitering;
    $("v-abandon").textContent = s.abandoned;
    $("t-fps").textContent = s.fps + " FPS";

    // stat card states
    $("c-loiter").classList.toggle("active", s.loitering > 0);
    $("c-abandon").classList.toggle("active", s.abandoned > 0);
    $("c-people").classList.toggle("warn", s.crowded);

    // totals
    const t = s.totals || {};
    $("tot-loiter").textContent = t.LOITERING || 0;
    $("tot-abandon").textContent = t.ABANDONED || 0;
    $("tot-crowd").textContent = t.CROWD || 0;

    // status banner
    const active = s.loitering > 0 || s.abandoned > 0 || s.crowded;
    const banner = $("status");
    banner.classList.toggle("alert", active);
    $("status-text").textContent = active
      ? "ALERT — REVIEW FEED"
      : "SYSTEM NOMINAL";
  } catch (e) {
    $("status-text").textContent = "CONNECTION LOST";
  }
}

async function pollAlerts() {
  try {
    const r = await fetch("/api/alerts");
    const events = await r.json();
    const list = $("alert-list");
    $("alert-count").textContent = events.length;

    if (!events.length) {
      list.innerHTML = '<li class="empty">Koi event nahi — feed monitor ho rahi hai.</li>';
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
  } catch (e) {
    /* ignore */
  }
}

setInterval(() => ($("t-clock").textContent = fmtClock()), 1000);
setInterval(pollStats, 1000);
setInterval(pollAlerts, 1500);
fmtClock();
pollStats();
pollAlerts();
