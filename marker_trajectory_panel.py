from __future__ import annotations

from flask import Flask, Response


HTML = """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Marker Trajectory Panel</title>
    <style>
      :root {
        --bg: #f3f6f2;
        --ink: #15211c;
        --muted: #66756d;
        --panel: #fbfdf9;
        --line: #d5ddd6;
        --accent: #197278;
        --accent-2: #d95d39;
        --accent-3: #f4c95d;
      }

      * { box-sizing: border-box; }
      html, body { height: 100%; }
      body {
        margin: 0;
        font-family: "Trebuchet MS", Verdana, sans-serif;
        color: var(--ink);
        background:
          linear-gradient(180deg, rgba(25,114,120,0.08), transparent 28rem),
          linear-gradient(135deg, #edf4ee, #f6f1e6 60%, #eef1f8);
      }

      .shell {
        width: min(1240px, calc(100% - 24px));
        margin: 0 auto;
        padding: 18px 0 24px;
      }

      .hero {
        display: grid;
        grid-template-columns: 1fr auto;
        gap: 16px;
        align-items: end;
        margin-bottom: 16px;
      }

      h1 {
        margin: 0;
        font-size: clamp(2rem, 4vw, 3.4rem);
        font-family: Georgia, "Times New Roman", serif;
        font-weight: 700;
      }

      .sub {
        margin: 8px 0 0;
        max-width: 720px;
        color: var(--muted);
        line-height: 1.5;
      }

      .telemetry {
        display: grid;
        gap: 8px;
        min-width: 220px;
        padding: 14px 16px;
        border: 1px solid var(--line);
        border-radius: 18px;
        background: rgba(251,253,249,0.92);
      }

      .telemetry span {
        display: flex;
        justify-content: space-between;
        gap: 20px;
        font-variant-numeric: tabular-nums;
      }

      .grid {
        display: grid;
        grid-template-columns: minmax(0, 1fr) 340px;
        gap: 16px;
      }

      .panel {
        border: 1px solid var(--line);
        border-radius: 24px;
        background: rgba(251,253,249,0.9);
        box-shadow: 0 18px 44px rgba(20,33,28,0.08);
      }

      .stage {
        padding: 10px;
      }

      canvas {
        display: block;
        width: 100%;
        height: min(76vh, 760px);
        min-height: 520px;
        border-radius: 18px;
        background: #f9fbf8;
      }

      .sidebar {
        display: grid;
        align-content: start;
        gap: 14px;
        padding: 14px;
      }

      .block {
        padding: 16px;
        border: 1px solid var(--line);
        border-radius: 18px;
        background: var(--panel);
      }

      .block h2 {
        margin: 0 0 14px;
        font-size: 1rem;
      }

      .joystick-wrap {
        display: grid;
        gap: 10px;
      }

      .joystick {
        position: relative;
        width: min(220px, 100%);
        aspect-ratio: 1;
        margin: 0 auto;
        border: 1px solid var(--line);
        border-radius: 50%;
        background:
          linear-gradient(rgba(21,33,28,0.08) 1px, transparent 1px),
          linear-gradient(90deg, rgba(21,33,28,0.08) 1px, transparent 1px),
          radial-gradient(circle, rgba(25,114,120,0.08), rgba(217,93,57,0.05));
        background-size: 100% 50%, 50% 100%, auto;
        touch-action: none;
        user-select: none;
      }

      .joystick-stick {
        position: absolute;
        left: 50%;
        top: 50%;
        width: 56px;
        height: 56px;
        border-radius: 50%;
        border: 4px solid #fff;
        background: var(--accent-2);
        box-shadow: 0 10px 20px rgba(217,93,57,0.25);
        transform: translate(-50%, -50%);
      }

      .joystick.rotate {
        aspect-ratio: 2.1 / 1;
        border-radius: 999px;
      }

      .axis {
        position: absolute;
        background: rgba(21,33,28,0.12);
      }

      .axis.x {
        left: 12%;
        right: 12%;
        top: calc(50% - 1px);
        height: 2px;
      }

      .axis.y {
        top: 12%;
        bottom: 12%;
        left: calc(50% - 1px);
        width: 2px;
      }

      .actions {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 10px;
      }

      button {
        min-height: 48px;
        border: 0;
        border-radius: 14px;
        cursor: pointer;
        color: white;
        background: var(--ink);
        font: inherit;
        font-weight: 700;
      }

      button.secondary {
        color: var(--ink);
        background: #ebf0ea;
      }

      .legend {
        display: grid;
        gap: 10px;
        color: var(--muted);
        font-size: 0.92rem;
      }

      .legend-row {
        display: flex;
        align-items: center;
        gap: 10px;
      }

      .swatch {
        width: 16px;
        height: 16px;
        border-radius: 4px;
      }

      @media (max-width: 920px) {
        .hero, .grid { grid-template-columns: 1fr; }
        .telemetry { grid-template-columns: repeat(3, minmax(0, 1fr)); }
      }

      @media (max-width: 720px) {
        .telemetry { grid-template-columns: 1fr; }
        .actions { grid-template-columns: 1fr; }
        canvas { min-height: 420px; }
      }
    </style>
  </head>
  <body>
    <main class="shell">
      <section class="hero">
        <div>
          <h1>Marker Trajectory Panel</h1>
          <p class="sub">A standalone top-down field view. Marker IDs 0-3 are scattered as fixed landmarks, and the robot path is traced from the motion commands you drive with the on-screen joysticks.</p>
        </div>
        <div class="telemetry">
          <span>Pose X <strong id="poseX">0.00</strong></span>
          <span>Pose Y <strong id="poseY">0.00</strong></span>
          <span>Yaw <strong id="poseYaw">0.0°</strong></span>
        </div>
      </section>

      <section class="grid">
        <div class="panel stage">
          <canvas id="fieldCanvas" width="1100" height="760" aria-label="Top-down marker and trajectory map"></canvas>
        </div>

        <aside class="panel sidebar">
          <div class="block">
            <h2>Translation</h2>
            <div class="joystick-wrap">
              <div class="joystick" id="leftJoystick">
                <div class="axis x"></div>
                <div class="axis y"></div>
                <div class="joystick-stick" id="leftStick"></div>
              </div>
            </div>
          </div>

          <div class="block">
            <h2>Rotation</h2>
            <div class="joystick-wrap">
              <div class="joystick rotate" id="rightJoystick">
                <div class="axis x"></div>
                <div class="joystick-stick" id="rightStick"></div>
              </div>
            </div>
          </div>

          <div class="block">
            <h2>Field Tools</h2>
            <div class="actions">
              <button id="resetTrail">Reset Trail</button>
              <button id="scatterMarkers" class="secondary">Scatter Markers</button>
            </div>
          </div>

          <div class="block">
            <h2>Legend</h2>
            <div class="legend">
              <div class="legend-row"><span class="swatch" style="background:#197278"></span>Trajectory</div>
              <div class="legend-row"><span class="swatch" style="background:#d95d39"></span>Robot body</div>
              <div class="legend-row"><span class="swatch" style="background:#f4c95d"></span>ArUco markers 0-3</div>
            </div>
          </div>
        </aside>
      </section>
    </main>

    <script>
      const canvas = document.querySelector("#fieldCanvas");
      const ctx = canvas.getContext("2d");
      const leftJoystick = document.querySelector("#leftJoystick");
      const rightJoystick = document.querySelector("#rightJoystick");
      const leftStick = document.querySelector("#leftStick");
      const rightStick = document.querySelector("#rightStick");
      const resetTrailButton = document.querySelector("#resetTrail");
      const scatterMarkersButton = document.querySelector("#scatterMarkers");
      const poseX = document.querySelector("#poseX");
      const poseY = document.querySelector("#poseY");
      const poseYaw = document.querySelector("#poseYaw");

      const FIELD_W = 5.0;
      const FIELD_H = 3.5;
      const MAX_LINEAR_SPEED = 1.2;
      const MAX_YAW_SPEED = 1.2;
      const MAX_TRAIL_POINTS = 900;
      const ROBOT_RADIUS_M = 0.16;

      const state = {
        left: { x: 0, y: 0 },
        right: { x: 0, y: 0 },
        pose: { x: FIELD_W / 2, y: FIELD_H / 2, yaw: -Math.PI / 2 },
        trail: [],
        markers: [],
        lastFrame: performance.now(),
      };

      function clamp(value, min, max) {
        return Math.min(max, Math.max(min, value));
      }

      function resizeCanvas() {
        const rect = canvas.getBoundingClientRect();
        const scale = window.devicePixelRatio || 1;
        canvas.width = Math.round(rect.width * scale);
        canvas.height = Math.round(rect.height * scale);
        ctx.setTransform(scale, 0, 0, scale, 0, 0);
      }

      function worldToCanvas(x, y) {
        const width = canvas.clientWidth;
        const height = canvas.clientHeight;
        const margin = 36;
        const usableW = width - margin * 2;
        const usableH = height - margin * 2;
        return {
          x: margin + (x / FIELD_W) * usableW,
          y: margin + (y / FIELD_H) * usableH,
        };
      }

      function scatterMarkers() {
        state.markers = [
          { id: 0, x: 0.7, y: 0.65 },
          { id: 1, x: 4.15, y: 0.9 },
          { id: 2, x: 1.35, y: 2.65 },
          { id: 3, x: 4.25, y: 2.75 },
        ].map((marker, index) => ({
          id: marker.id,
          x: marker.x + (Math.random() - 0.5) * 0.25 * (index % 2 ? 1 : -1),
          y: marker.y + (Math.random() - 0.5) * 0.25,
        })).map((marker) => ({
          id: marker.id,
          x: clamp(marker.x, 0.4, FIELD_W - 0.4),
          y: clamp(marker.y, 0.4, FIELD_H - 0.4),
        }));
      }

      function resetTrail() {
        state.pose = { x: FIELD_W / 2, y: FIELD_H / 2, yaw: -Math.PI / 2 };
        state.trail = [];
      }

      function renderStick(pad, stick, value) {
        const rect = pad.getBoundingClientRect();
        const maxX = rect.width / 2 - 28;
        const maxY = rect.height / 2 - 28;
        stick.style.transform = `translate(calc(-50% + ${value.x * maxX}px), calc(-50% + ${value.y * maxY}px))`;
      }

      function setupJoystick(pad, stick, stateKey, horizontalOnly = false) {
        function setFromPointer(event) {
          const rect = pad.getBoundingClientRect();
          const radiusX = rect.width / 2;
          const radiusY = rect.height / 2;
          let x = (event.clientX - rect.left - radiusX) / (radiusX - 28);
          let y = (event.clientY - rect.top - radiusY) / (radiusY - 28);

          if (horizontalOnly) {
            y = 0;
            x = clamp(x, -1, 1);
          } else {
            const mag = Math.hypot(x, y);
            if (mag > 1) {
              x /= mag;
              y /= mag;
            }
          }

          state[stateKey].x = clamp(x, -1, 1);
          state[stateKey].y = clamp(y, -1, 1);
          renderStick(pad, stick, state[stateKey]);
        }

        pad.addEventListener("pointerdown", (event) => {
          pad.setPointerCapture(event.pointerId);
          setFromPointer(event);
        });

        pad.addEventListener("pointermove", (event) => {
          if (pad.hasPointerCapture(event.pointerId)) {
            setFromPointer(event);
          }
        });

        function release(event) {
          if (pad.hasPointerCapture(event.pointerId)) {
            pad.releasePointerCapture(event.pointerId);
          }
          state[stateKey].x = 0;
          state[stateKey].y = 0;
          renderStick(pad, stick, state[stateKey]);
        }

        pad.addEventListener("pointerup", release);
        pad.addEventListener("pointercancel", release);
      }

      function drawField() {
        const width = canvas.clientWidth;
        const height = canvas.clientHeight;

        ctx.clearRect(0, 0, width, height);
        ctx.fillStyle = "#f9fbf8";
        ctx.fillRect(0, 0, width, height);

        const topLeft = worldToCanvas(0, 0);
        const bottomRight = worldToCanvas(FIELD_W, FIELD_H);

        ctx.strokeStyle = "#cfd9d1";
        ctx.lineWidth = 2;
        ctx.strokeRect(topLeft.x, topLeft.y, bottomRight.x - topLeft.x, bottomRight.y - topLeft.y);

        ctx.strokeStyle = "rgba(21,33,28,0.08)";
        ctx.lineWidth = 1;
        for (let x = 0.5; x < FIELD_W; x += 0.5) {
          const a = worldToCanvas(x, 0);
          const b = worldToCanvas(x, FIELD_H);
          ctx.beginPath();
          ctx.moveTo(a.x, a.y);
          ctx.lineTo(b.x, b.y);
          ctx.stroke();
        }
        for (let y = 0.5; y < FIELD_H; y += 0.5) {
          const a = worldToCanvas(0, y);
          const b = worldToCanvas(FIELD_W, y);
          ctx.beginPath();
          ctx.moveTo(a.x, a.y);
          ctx.lineTo(b.x, b.y);
          ctx.stroke();
        }
      }

      function drawMarkers() {
        state.markers.forEach((marker) => {
          const p = worldToCanvas(marker.x, marker.y);
          const size = 34;
          ctx.fillStyle = "#fffdf6";
          ctx.strokeStyle = "#15211c";
          ctx.lineWidth = 2;
          ctx.fillRect(p.x - size / 2, p.y - size / 2, size, size);
          ctx.strokeRect(p.x - size / 2, p.y - size / 2, size, size);

          ctx.fillStyle = "#15211c";
          ctx.fillRect(p.x - 10, p.y - 10, 20, 20);
          ctx.fillStyle = "#f4c95d";
          ctx.font = "700 14px Trebuchet MS";
          ctx.textAlign = "center";
          ctx.textBaseline = "top";
          ctx.fillText(`ID ${marker.id}`, p.x, p.y + size / 2 + 8);
        });
      }

      function drawTrail() {
        if (state.trail.length < 2) {
          return;
        }
        ctx.strokeStyle = "#197278";
        ctx.lineWidth = 4;
        ctx.lineCap = "round";
        ctx.lineJoin = "round";
        ctx.beginPath();
        state.trail.forEach((point, index) => {
          const p = worldToCanvas(point.x, point.y);
          if (index === 0) {
            ctx.moveTo(p.x, p.y);
          } else {
            ctx.lineTo(p.x, p.y);
          }
        });
        ctx.stroke();
      }

      function drawRobot() {
        const p = worldToCanvas(state.pose.x, state.pose.y);
        const scale = (canvas.clientWidth - 72) / FIELD_W;
        const radius = ROBOT_RADIUS_M * scale;

        ctx.save();
        ctx.translate(p.x, p.y);
        ctx.rotate(state.pose.yaw);

        ctx.fillStyle = "#d95d39";
        ctx.strokeStyle = "#15211c";
        ctx.lineWidth = 3;
        ctx.beginPath();
        for (let i = 0; i < 3; i += 1) {
          const angle = -Math.PI / 2 + (i * Math.PI * 2) / 3;
          const x = Math.cos(angle) * radius * 1.55;
          const y = Math.sin(angle) * radius * 1.55;
          if (i === 0) {
            ctx.moveTo(x, y);
          } else {
            ctx.lineTo(x, y);
          }
        }
        ctx.closePath();
        ctx.fill();
        ctx.stroke();

        ctx.strokeStyle = "#fff";
        ctx.lineWidth = 4;
        ctx.beginPath();
        ctx.moveTo(0, 0);
        ctx.lineTo(0, -radius * 1.35);
        ctx.stroke();
        ctx.restore();
      }

      function update(deltaSeconds) {
        const vx = state.left.x * MAX_LINEAR_SPEED;
        const vy = -state.left.y * MAX_LINEAR_SPEED;
        const om = state.right.x * MAX_YAW_SPEED;

        state.pose.x = clamp(state.pose.x + vx * deltaSeconds, 0.18, FIELD_W - 0.18);
        state.pose.y = clamp(state.pose.y + vy * deltaSeconds, 0.18, FIELD_H - 0.18);
        state.pose.yaw += om * deltaSeconds;

        const last = state.trail[state.trail.length - 1];
        if (!last || Math.hypot(last.x - state.pose.x, last.y - state.pose.y) > 0.01) {
          state.trail.push({ x: state.pose.x, y: state.pose.y });
          if (state.trail.length > MAX_TRAIL_POINTS) {
            state.trail.shift();
          }
        }

        poseX.textContent = state.pose.x.toFixed(2);
        poseY.textContent = state.pose.y.toFixed(2);
        poseYaw.textContent = `${(state.pose.yaw * 180 / Math.PI).toFixed(1)}°`;
      }

      function frame(now) {
        const dt = Math.min((now - state.lastFrame) / 1000, 0.04);
        state.lastFrame = now;
        update(dt);
        drawField();
        drawMarkers();
        drawTrail();
        drawRobot();
        requestAnimationFrame(frame);
      }

      setupJoystick(leftJoystick, leftStick, "left", false);
      setupJoystick(rightJoystick, rightStick, "right", true);
      resetTrailButton.addEventListener("click", resetTrail);
      scatterMarkersButton.addEventListener("click", scatterMarkers);
      window.addEventListener("resize", () => {
        resizeCanvas();
        renderStick(leftJoystick, leftStick, state.left);
        renderStick(rightJoystick, rightStick, state.right);
      });

      resizeCanvas();
      renderStick(leftJoystick, leftStick, state.left);
      renderStick(rightJoystick, rightStick, state.right);
      scatterMarkers();
      resetTrail();
      requestAnimationFrame(frame);
    </script>
  </body>
</html>
"""


app = Flask(__name__)


@app.get("/")
def index() -> Response:
    return Response(HTML, mimetype="text/html")


if __name__ == "__main__":
    print("Serving marker trajectory panel on http://0.0.0.0:8010")
    app.run(host="0.0.0.0", port=8010, threaded=True)
