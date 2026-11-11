/**
 * Interactive Generals.io board: canvas renderer plus input handling.
 *
 * Plain ES-module JavaScript with no imports, so the Next.js app and the local
 * Python server can both load it without the other's build step. The Next.js
 * copy in `web/public/board.mjs` is synced from here by `scripts/sync-shared.mjs`.
 *
 *   const board = createBoard(canvas, { onAppend, onClear });
 *   board.setPlayer(0);
 *   board.update(snapshot);          // { h, w, owner, army, mountain, ... }
 *   board.setServerQueue(seq, moves);
 */

const DIRS = [
  [-1, 0],
  [1, 0],
  [0, -1],
  [0, 1],
];
const KEY_DIR = {
  ArrowUp: 0, ArrowDown: 1, ArrowLeft: 2, ArrowRight: 3,
  w: 0, s: 1, a: 2, d: 3, W: 0, S: 1, A: 2, D: 3,
};

const FALLBACK = {
  p0: "#bf3b34", p1: "#2c5f96", p2: "#4c8b52", p3: "#7a5aa8",
  neutral: "#b6c0bf", castle: "#8e9c9c", mountain: "#77868a",
  grid: "rgba(19,26,28,.16)", tileInk: "#ffffff", plainInk: "#2a3538",
  brass: "#9a6b12", ink3: "#7b8a8e",
};

function tokens(el) {
  const s = getComputedStyle(el);
  const get = (name, fb) => (s.getPropertyValue(name) || "").trim() || fb;
  return {
    players: [
      get("--p0", FALLBACK.p0), get("--p1", FALLBACK.p1),
      get("--p2", FALLBACK.p2), get("--p3", FALLBACK.p3),
    ],
    neutral: get("--neutral-land", FALLBACK.neutral),
    castle: get("--neutral-castle", FALLBACK.castle),
    mountain: get("--mountain", FALLBACK.mountain),
    grid: get("--grid", FALLBACK.grid),
    tileInk: get("--tile-ink", FALLBACK.tileInk),
    plainInk: get("--plain-ink", FALLBACK.plainInk),
    brass: get("--brass", FALLBACK.brass),
    ink3: get("--ink-3", FALLBACK.ink3),
  };
}

export function createBoard(canvas, opts = {}) {
  const ctx = canvas.getContext("2d");
  const onAppend = opts.onAppend || (() => {});
  const onClear = opts.onClear || (() => {});

  let snap = null;
  let me = -1;
  let cell = 28;
  let dpr = 1;
  let C = tokens(document.documentElement);

  /** Moves the server has confirmed are still pending, plus our unacknowledged ones. */
  let serverQueue = [];
  let serverSeq = 0;
  let localQueue = []; // { seq, from, dir, half }
  let seqCounter = 0;
  let selection = -1;
  /** Cells the cursor has walked through, newest last, so a broken path can fall back. */
  let trail = [];
  let interactive = false;

  const queue = () => serverQueue.concat(localQueue.filter((m) => m.seq > serverSeq));

  /**
   * The tile the next arrow key acts from.
   *
   * It advances the instant you queue a move, exactly like the browser game:
   * tapping a direction four times walks the cursor four tiles and queues four
   * moves. Deriving it from the pending queue instead would snap the cursor back
   * to the start every time a move was consumed.
   */
  const cursor = () => selection;

  // ---- layout -------------------------------------------------------------
  function layout() {
    if (!snap) return;
    const avail = (canvas.parentElement?.clientWidth || 640) - 4;
    const maxH = opts.maxHeight || Math.max(320, window.innerHeight - 260);
    cell = Math.max(12, Math.min(36, Math.floor(avail / snap.w), Math.floor(maxH / snap.h)));
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = snap.w * cell * dpr;
    canvas.height = snap.h * cell * dpr;
    canvas.style.width = snap.w * cell + "px";
    canvas.style.height = snap.h * cell + "px";
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  // ---- drawing ------------------------------------------------------------
  function crown(x, y, s, col) {
    ctx.fillStyle = col;
    ctx.beginPath();
    ctx.moveTo(x - s, y + s * 0.55);
    ctx.lineTo(x - s, y - s * 0.35);
    ctx.lineTo(x - s * 0.42, y + s * 0.1);
    ctx.lineTo(x, y - s * 0.75);
    ctx.lineTo(x + s * 0.42, y + s * 0.1);
    ctx.lineTo(x + s, y - s * 0.35);
    ctx.lineTo(x + s, y + s * 0.55);
    ctx.closePath();
    ctx.fill();
  }

  function arrow(from, dir, half, colour, alpha) {
    const [dr, dc] = DIRS[dir];
    const r0 = Math.floor(from / snap.w);
    const c0 = from % snap.w;
    const x0 = c0 * cell + cell / 2;
    const y0 = r0 * cell + cell / 2;
    const x1 = (c0 + dc) * cell + cell / 2;
    const y1 = (r0 + dr) * cell + cell / 2;
    const mx = x0 + (x1 - x0) * 0.78;
    const my = y0 + (y1 - y0) * 0.78;
    ctx.save();
    ctx.globalAlpha = alpha;
    ctx.lineCap = "round";
    ctx.strokeStyle = "rgba(0,0,0,.45)";
    ctx.lineWidth = cell * 0.19;
    ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(mx, my); ctx.stroke();
    ctx.strokeStyle = colour;
    ctx.lineWidth = cell * 0.1;
    ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(mx, my); ctx.stroke();
    const ang = Math.atan2(y1 - y0, x1 - x0);
    const hd = cell * 0.25;
    ctx.fillStyle = colour;
    ctx.beginPath();
    ctx.moveTo(x0 + (x1 - x0) * 0.96, y0 + (y1 - y0) * 0.96);
    ctx.lineTo(mx - hd * Math.cos(ang - 0.5), my - hd * Math.sin(ang - 0.5));
    ctx.lineTo(mx - hd * Math.cos(ang + 0.5), my - hd * Math.sin(ang + 0.5));
    ctx.closePath();
    ctx.fill();
    if (half) {
      ctx.fillStyle = "rgba(255,255,255,.92)";
      ctx.beginPath(); ctx.arc(x0, y0, cell * 0.085, 0, Math.PI * 2); ctx.fill();
    }
    ctx.restore();
  }

  function draw() {
    if (!snap) return;
    const { h, w } = snap;
    ctx.clearRect(0, 0, w * cell, h * cell);
    const fs = Math.max(8, Math.round(cell * 0.4));
    ctx.font = `600 ${fs}px "IBM Plex Mono", ui-monospace, monospace`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";

    for (let i = 0; i < h * w; i++) {
      const r = Math.floor(i / w);
      const c = i % w;
      const x = c * cell;
      const y = r * cell;
      const o = snap.owner[i];
      const isCastle = !!snap.castle[i];
      const isGen = !!snap.general[i];

      ctx.fillStyle = snap.mountain[i]
        ? C.mountain
        : o >= 0
          ? C.players[o] || FALLBACK.p0
          : isCastle || isGen
            ? C.castle
            : C.neutral;
      ctx.fillRect(x, y, cell, cell);
      ctx.strokeStyle = C.grid;
      ctx.lineWidth = 1;
      ctx.strokeRect(x + 0.5, y + 0.5, cell - 1, cell - 1);
      if (snap.mountain[i]) continue;

      const cx = x + cell / 2;
      const cy = y + cell / 2;
      if (isGen) crown(cx, cy, cell * 0.3, o >= 0 ? "rgba(255,255,255,.35)" : "rgba(0,0,0,.2)");
      else if (isCastle) {
        ctx.strokeStyle = o >= 0 ? "rgba(255,255,255,.32)" : C.ink3;
        ctx.lineWidth = Math.max(1, cell * 0.09);
        ctx.beginPath(); ctx.arc(cx, cy, cell * 0.28, 0, Math.PI * 2); ctx.stroke();
      }

      const a = snap.army[i];
      if (a !== 0) {
        ctx.lineWidth = Math.max(2, cell * 0.11);
        ctx.strokeStyle = "rgba(0,0,0,.30)";
        ctx.lineJoin = "round";
        ctx.strokeText(a, cx, cy + 1);
        ctx.fillStyle = o >= 0 ? C.tileInk : C.plainInk;
        ctx.fillText(a, cx, cy + 1);
      }
    }

    // last resolved moves, so you can read what the opponent just committed
    if (snap.lastMoves) {
      for (let p = 0; p < snap.lastMoves.length; p++) {
        const m = snap.lastMoves[p];
        if (!m || m.dir < 0) continue;
        arrow(m.from, m.dir, m.half, C.players[p] || FALLBACK.p0, 0.85);
      }
    }

    // our own pending queue, drawn lighter and fading down the path
    const q = queue();
    q.forEach((m, k) => {
      arrow(m.from, m.dir, m.half, C.brass, Math.max(0.25, 0.8 - k * 0.07));
    });

    const cur = interactive ? cursor() : -1;
    if (cur >= 0 && cur < h * w) {
      ctx.strokeStyle = C.brass;
      ctx.lineWidth = 2.5;
      ctx.strokeRect(
        (cur % w) * cell + 1.5,
        Math.floor(cur / w) * cell + 1.5,
        cell - 3,
        cell - 3,
      );
    }
  }

  // ---- input --------------------------------------------------------------
  function cellAt(ev) {
    const rect = canvas.getBoundingClientRect();
    const c = Math.floor(((ev.clientX - rect.left) / rect.width) * snap.w);
    const r = Math.floor(((ev.clientY - rect.top) / rect.height) * snap.h);
    if (r < 0 || r >= snap.h || c < 0 || c >= snap.w) return -1;
    return r * snap.w + c;
  }

  function onPointerDown(ev) {
    if (!interactive || !snap) return;
    const i = cellAt(ev);
    if (i < 0) return;
    ev.preventDefault();
    if (snap.owner[i] === me) {
      localQueue = [];
      serverQueue = [];
      onClear();
      selection = i; // a click starts a fresh path from this tile
      trail = [i];
      draw();
    }
  }

  function push(dir, half) {
    if (!interactive || !snap || snap.done) return;
    const from = selection;
    if (from < 0) return;
    const r = Math.floor(from / snap.w) + DIRS[dir][0];
    const c = (from % snap.w) + DIRS[dir][1];
    if (r < 0 || r >= snap.h || c < 0 || c >= snap.w) return;
    const dest = r * snap.w + c;
    if (snap.mountain[dest]) return;
    const move = { seq: ++seqCounter, from, dir, half: !!half };
    localQueue.push(move);
    // The cursor travels with the army, so the next key chains off this move.
    selection = dest;
    trail.push(dest);
    if (trail.length > 96) trail.shift();
    onAppend({ from, dir, half: !!half, seq: move.seq });
    draw();
  }

  function clearQueue() {
    localQueue = [];
    serverQueue = [];
    // The path is gone, so the cursor no longer means anything -- deselect
    // rather than leave it pointing at a tile the army never reached.
    selection = -1;
    trail = [];
    onClear();
    draw();
  }

  function onKeyDown(ev) {
    if (!interactive || !snap) return;
    if (ev.metaKey || ev.ctrlKey || ev.altKey) return;
    const target = ev.target;
    if (target && /^(INPUT|TEXTAREA|SELECT)$/.test(target.tagName)) return;
    const dir = KEY_DIR[ev.key];
    if (dir !== undefined) {
      ev.preventDefault();
      push(dir, ev.shiftKey);
      return;
    }
    if (ev.key === "q" || ev.key === "Q" || ev.key === "Escape") {
      ev.preventDefault();
      clearQueue();
    }
  }

  canvas.addEventListener("pointerdown", onPointerDown);
  window.addEventListener("keydown", onKeyDown);

  const onResize = () => { layout(); draw(); };
  window.addEventListener("resize", onResize);
  const mq = window.matchMedia("(prefers-color-scheme: dark)");
  const onTheme = () => { C = tokens(document.documentElement); draw(); };
  mq.addEventListener("change", onTheme);
  const themeObserver = new MutationObserver(onTheme);
  themeObserver.observe(document.documentElement, {
    attributes: true, attributeFilter: ["data-theme"],
  });

  return {
    update(next) {
      const first = !snap;
      snap = next;
      if (first || next.h !== snap.h) layout();
      if (first) layout();
      // While a path is in flight the cursor legitimately sits on tiles we have
      // not taken yet. Once nothing is pending, a cursor we do not own means the
      // path stalled -- usually on a castle the stack could not afford. Fall back
      // along the trail to where the army actually is, rather than deselecting
      // and making the player re-click.
      if (selection >= 0 && snap.owner[selection] !== me && !queue().length) {
        while (trail.length && snap.owner[trail[trail.length - 1]] !== me) trail.pop();
        selection = trail.length ? trail[trail.length - 1] : -1;
      }
      draw();
    },
    setPlayer(p) { me = p; interactive = p >= 0; draw(); },
    setInteractive(v) { interactive = !!v; draw(); },
    setServerQueue(seq, moves) {
      serverSeq = seq;
      serverQueue = moves || [];
      localQueue = localQueue.filter((m) => m.seq > seq);
      draw();
    },
    queueLength() { return queue().length; },
    clearQueue,
    push,
    refreshTheme: onTheme,
    relayout: onResize,
    destroy() {
      canvas.removeEventListener("pointerdown", onPointerDown);
      window.removeEventListener("keydown", onKeyDown);
      window.removeEventListener("resize", onResize);
      mq.removeEventListener("change", onTheme);
      themeObserver.disconnect();
    },
  };
}
