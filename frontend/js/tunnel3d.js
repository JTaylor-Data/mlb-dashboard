// Tunneling+ 3D viewer. Every trajectory here is real, unadjusted physics
// (real release, real velocity, real movement, ending at each pitch type's
// own real average location) - this is the VISUAL layer for the validated
// Tunneling+ model (analytics/tunneling_model.py), not a second scoring
// system. Every number shown (decision point, pair scores, tunneling_plus)
// is pulled from that model's own endpoints, not recomputed here.
import * as THREE from "three";
import { OrbitControls } from "../vendor/addons/controls/OrbitControls.js";

const container = document.getElementById("t3-container");

// Statcast (x, y, z) -> Three.js (x, y-up, z). x is negated: standing AT
// HOME PLATE facing the pitcher, 1B is on your right (undisputable); the
// pitcher faces the opposite direction, so mirrored, 1B is on the
// PITCHER's LEFT. Verified against a broad real sample (1,994 RHPs, 3.1M
// pitches) that positive statcast x really is the 1B side.
function toThree(p) {
  return new THREE.Vector3(-p.x, p.z, p.y ?? 0);
}

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x8fc8ee);
scene.fog = new THREE.Fog(0x8fc8ee, 70, 160);

function containerSize() {
  return { w: container.clientWidth, h: container.clientHeight };
}

const camera = new THREE.PerspectiveCamera(50, containerSize().w / containerSize().h, 0.1, 300);
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(containerSize().w, containerSize().h);
container.insertBefore(renderer.domElement, container.firstChild);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;

const VIEWS = {
  pitcher: { pos: [0, 9, 63], target: [0, 3, 4] },
  side: { pos: [42, 8, 27], target: [0, 3, 27] },
  ump: { pos: [0, 6.5, -11], target: [0, 3, 45] },
};
function setView(name) {
  const v = VIEWS[name];
  camera.position.set(...v.pos);
  controls.target.set(...v.target);
  controls.update();
  document.querySelectorAll("#t3-views button").forEach((b) => b.classList.toggle("active", b.dataset.view === name));
}
setView("pitcher");

scene.add(new THREE.AmbientLight(0xffffff, 0.7));
const dirLight = new THREE.DirectionalLight(0xffffff, 0.6);
dirLight.position.set(10, 20, 10);
scene.add(dirLight);

function buildField() {
  const grass = new THREE.Mesh(new THREE.CircleGeometry(65, 48), new THREE.MeshStandardMaterial({ color: 0x4f9a56 }));
  grass.rotation.x = -Math.PI / 2;
  grass.position.set(0, -0.02, 28);
  scene.add(grass);

  const dirtMat = new THREE.MeshStandardMaterial({ color: 0xb08a63 });
  const plateCircle = new THREE.Mesh(new THREE.CircleGeometry(12, 40), dirtMat);
  plateCircle.rotation.x = -Math.PI / 2;
  plateCircle.position.set(0, -0.01, 1);
  scene.add(plateCircle);

  const moundCircle = new THREE.Mesh(new THREE.CircleGeometry(9, 32), dirtMat.clone());
  moundCircle.rotation.x = -Math.PI / 2;
  moundCircle.position.set(0, -0.005, 54);
  scene.add(moundCircle);

  const half = 0.708;
  const shape = new THREE.Shape();
  shape.moveTo(-half, 0.15);
  shape.lineTo(half, 0.15);
  shape.lineTo(half, -0.4);
  shape.lineTo(0, -0.75);
  shape.lineTo(-half, -0.4);
  shape.closePath();
  const plate = new THREE.Mesh(new THREE.ShapeGeometry(shape), new THREE.MeshStandardMaterial({ color: 0xdedede }));
  plate.rotation.x = -Math.PI / 2;
  plate.position.set(0, 0.01, 0);
  scene.add(plate);

  for (const side of [-1, 1]) {
    const x0 = side * 1.2, x1 = side * 5.2;
    const z0 = -1.5, z1 = 4.5;
    const boxPts = [
      new THREE.Vector3(x0, 0.015, z0), new THREE.Vector3(x1, 0.015, z0),
      new THREE.Vector3(x1, 0.015, z1), new THREE.Vector3(x0, 0.015, z1),
    ];
    const boxGeom = new THREE.BufferGeometry().setFromPoints(boxPts);
    scene.add(new THREE.LineLoop(boxGeom, new THREE.LineBasicMaterial({ color: 0xcfcfcf })));
  }

  for (const side of [-1, 1]) {
    const lineGeom = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(0, 0.02, 0), new THREE.Vector3(side * 60, 0.02, 60),
    ]);
    scene.add(new THREE.Line(lineGeom, new THREE.LineBasicMaterial({ color: 0xffffff })));
  }
}
buildField();

function buildStadium() {
  const owall = new THREE.Mesh(new THREE.BoxGeometry(100, 4, 1), new THREE.MeshStandardMaterial({ color: 0x1f4a30 }));
  owall.position.set(0, 2, 118);
  scene.add(owall);
}
buildStadium();

function makeTextSprite(text, color) {
  const canvas = document.createElement("canvas");
  canvas.width = 256; canvas.height = 128;
  const ctx = canvas.getContext("2d");
  ctx.font = "bold 64px system-ui, sans-serif";
  ctx.fillStyle = color;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(text, 128, 64);
  const tex = new THREE.CanvasTexture(canvas);
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, transparent: true, depthTest: false }));
  sprite.scale.set(4.5, 2.25, 1);
  return sprite;
}
function buildOrientationLabels() {
  const first = makeTextSprite("1B", "#ffd54a");
  first.position.set(-9, 3, 9);
  scene.add(first);
  const third = makeTextSprite("3B", "#ffd54a");
  third.position.set(9, 3, 9);
  scene.add(third);
}
buildOrientationLabels();

// Batter silhouette - two separate real images, one per handedness, no
// runtime flipping. The file NAMES are swapped from their real visual
// content - "batter_rhh.png" is the file that's actually correct for a LHH
// pose (and vice versa) - confirmed by the user directly after two earlier
// rounds of getting this backwards, same real-content correction applied
// consistently in player.js's Pitch Locations/Hit Zones charts (see those
// files' own comments on this exact swap). Hence the load-order swap below
// looking backwards at a glance - it's intentional, not a bug; do not
// "fix" it back to the naive filename<->handedness mapping. Used as a
// billboard sprite so it always faces the camera regardless of view.
const BATTER_ASPECT = 501 / 1262; // real cropped image width/height
const BATTER_HEIGHT = 5.5; // feet, roughly a real batter's height in this crouch
const batterTextures = {}; // { R: texture, L: texture }
let batterSprite = null;
let pendingStand = null;
const batterLoader = new THREE.TextureLoader();
batterLoader.load("img/batter_rhh.png", (tex) => { batterTextures.L = tex; if (pendingStand) setBatterSilhouette(pendingStand); });
batterLoader.load("img/batter_lhh.png", (tex) => { batterTextures.R = tex; if (pendingStand) setBatterSilhouette(pendingStand); });

function setBatterSilhouette(stand) {
  pendingStand = stand;
  const tex = batterTextures[stand];
  if (!tex) return; // will render once that stand's texture finishes loading
  if (batterSprite) {
    scene.remove(batterSprite);
    batterSprite.material.dispose();
    batterSprite = null;
  }
  const mat = new THREE.SpriteMaterial({ map: tex, transparent: true, depthTest: false });
  batterSprite = new THREE.Sprite(mat);
  batterSprite.scale.set(BATTER_HEIGHT * BATTER_ASPECT, BATTER_HEIGHT, 1);
  // RHH = 3B side = positive three.x, LHH = 1B side = negative three.x.
  const boxSide = stand === "R" ? 1 : -1;
  batterSprite.position.set(boxSide * 3.2, BATTER_HEIGHT / 2, 1.5);
  scene.add(batterSprite);
}

const ZONE_X0 = -0.7085, ZONE_X1 = 0.7085, ZONE_Y0 = 1.5, ZONE_Y1 = 3.5;
const zoneGeom = new THREE.BoxGeometry(ZONE_X1 - ZONE_X0, ZONE_Y1 - ZONE_Y0, 0.02);
const zoneEdges = new THREE.LineSegments(new THREE.EdgesGeometry(zoneGeom), new THREE.LineBasicMaterial({ color: 0x4a5a7a }));
zoneEdges.position.set(0, 2.5, 0);
scene.add(zoneEdges);

function buildZoneGrid() {
  const mat = new THREE.LineDashedMaterial({ color: 0x5a6a8a, dashSize: 0.05, gapSize: 0.05, transparent: true, opacity: 0.55 });
  const xs = [ZONE_X0 + (ZONE_X1 - ZONE_X0) / 3, ZONE_X0 + 2 * (ZONE_X1 - ZONE_X0) / 3];
  const ys = [ZONE_Y0 + (ZONE_Y1 - ZONE_Y0) / 3, ZONE_Y0 + 2 * (ZONE_Y1 - ZONE_Y0) / 3];
  for (const x of xs) {
    const g = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(x, ZONE_Y0, 0), new THREE.Vector3(x, ZONE_Y1, 0)]);
    const line = new THREE.Line(g, mat);
    line.computeLineDistances();
    scene.add(line);
  }
  for (const y of ys) {
    const g = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(ZONE_X0, y, 0), new THREE.Vector3(ZONE_X1, y, 0)]);
    const line = new THREE.Line(g, mat);
    line.computeLineDistances();
    scene.add(line);
  }
}
buildZoneGrid();

const legendEl = document.getElementById("t3-legend");
let pitches = [];
let dynamicObjects = [];
let trajData = null;   // /api/tunneling/{id}/trajectories response
let pairDetail = null; // /api/tunneling/pitcher/{id} response (validated scores), or null if not qualified
let currentStand = "R";
let currentPitcherId = null, currentSeason = null;

let playing = false;
let elapsed = 0;
let speed = 0.5;
const clock = new THREE.Clock();

// --- Delivery skeleton (real video-derived pose data, Yamamoto only so far)
// ---
// Real per-frame 2D pose landmarks (x/y only - monocular video has no depth
// channel) from the deception_v4 research pipeline's YOLO-pose extraction,
// backward-chained real phase timestamps (coming_set/leg_lift/foot_plant/
// release) from that same pipeline - see frontend/data/delivery_yamamoto.json's
// own "note" field for the full honesty disclosure. Depth (toward/away from
// the plate) isn't in that file at all; it's supplied here by interpolating
// the whole-body root from an approximate rubber position to the real,
// exact release position already used by the pitch trajectories, timed
// against the real leg-lift-to-release phase window rather than linear time
// (almost all forward drive happens in that back half of a real delivery).
const DELIVERY_ASSETS = { 808967: "data/delivery_yamamoto.json" };
const SKELETON_BONES = [
  ["l_shoulder", "r_shoulder"], ["l_hip", "r_hip"],
  ["l_shoulder", "l_hip"], ["r_shoulder", "r_hip"],
  ["l_shoulder", "l_elbow"], ["l_elbow", "l_wrist"],
  ["r_shoulder", "r_elbow"], ["r_elbow", "r_wrist"],
  ["l_hip", "l_knee"], ["l_knee", "l_ankle"],
  ["r_hip", "r_knee"], ["r_knee", "r_ankle"],
  ["nose", "l_shoulder"], ["nose", "r_shoulder"],
];
let deliveryData = null;     // parsed asset for the current pitcher, or null
let deliveryDuration = 0;    // seconds from coming_set to release (0 = no delivery phase, old release-start behavior)
let deliveryAnchor = null;   // { start: Vector3 (approx rubber), end: Vector3 (real release point) }
let skeletonGroup = null, skeletonGeom = null;

function buildSkeleton() {
  skeletonGeom = new THREE.BufferGeometry();
  skeletonGeom.setAttribute("position", new THREE.Float32BufferAttribute(new Float32Array(SKELETON_BONES.length * 6), 3));
  const lines = new THREE.LineSegments(skeletonGeom, new THREE.LineBasicMaterial({ color: 0xffe08a, linewidth: 2 }));
  const head = new THREE.Mesh(new THREE.SphereGeometry(0.28, 12, 12), new THREE.MeshStandardMaterial({ color: 0xffe08a }));
  skeletonGroup = new THREE.Group();
  skeletonGroup.add(lines);
  skeletonGroup.add(head);
  skeletonGroup.userData.lines = lines;
  skeletonGroup.userData.head = head;
  skeletonGroup.visible = false;
  scene.add(skeletonGroup);
}
buildSkeleton();

function lerpFrame(frames, t) {
  if (t <= frames[0].t) return frames[0];
  if (t >= frames[frames.length - 1].t) return frames[frames.length - 1];
  let lo = 0, hi = frames.length - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (frames[mid].t <= t) lo = mid; else hi = mid;
  }
  const a = frames[lo], b = frames[hi];
  const f = (t - a.t) / (b.t - a.t || 1);
  const joints = {};
  for (const name of Object.keys(a.joints)) {
    if (!b.joints[name]) continue;
    const [ax, ay] = a.joints[name], [bx, by] = b.joints[name];
    joints[name] = [ax + (bx - ax) * f, ay + (by - ay) * f];
  }
  return { joints };
}

function smoothstep(f) {
  const x = Math.max(0, Math.min(1, f));
  return x * x * (3 - 2 * x);
}

function updateSkeletonPose(t) {
  if (!deliveryData || !skeletonGroup) return;
  const ph = deliveryData.phases;
  const driveFrac = ph.release > ph.leg_lift_start
    ? smoothstep((t - ph.leg_lift_start) / (ph.release - ph.leg_lift_start))
    : t >= ph.release ? 1 : 0;
  const root = deliveryAnchor.start.clone().lerp(deliveryAnchor.end, driveFrac);

  const frame = lerpFrame(deliveryData.frames, t);
  const positions = skeletonGeom.attributes.position.array;
  let i = 0;
  for (const [a, b] of SKELETON_BONES) {
    const ja = frame.joints[a], jb = frame.joints[b];
    if (!ja || !jb) { positions.fill(0, i, i + 6); i += 6; continue; }
    positions[i++] = root.x + ja[0]; positions[i++] = root.y + ja[1]; positions[i++] = root.z;
    positions[i++] = root.x + jb[0]; positions[i++] = root.y + jb[1]; positions[i++] = root.z;
  }
  skeletonGeom.attributes.position.needsUpdate = true;
  const nose = frame.joints.nose;
  if (nose) skeletonGroup.userData.head.position.set(root.x + nose[0], root.y + nose[1], root.z);
}

function resetToIdle() {
  playing = false;
  elapsed = 0;
  for (const p of pitches) {
    p.mesh.position.copy(p.releaseVec);
    p.geom.setDrawRange(0, 0);
    p.markerShouldShow = false;
    if (p.marker) p.marker.visible = false;
  }
  if (skeletonGroup) {
    skeletonGroup.visible = !!deliveryData;
    if (deliveryData) updateSkeletonPose(0);
  }
  const btn = document.getElementById("t3-btn-play");
  if (btn) btn.textContent = "Play";
}

function togglePlay() {
  if (pitches.length === 0) return;
  const allDone = pitches.every((p) => elapsed >= deliveryDuration + p.t_plate);
  if (allDone) resetToIdle();
  playing = !playing;
  document.getElementById("t3-btn-play").textContent = playing ? "Pause" : "Play";
}

function setSpeed(v) {
  speed = v;
  document.querySelectorAll("#t3-speeds button").forEach((b) => b.classList.toggle("active", parseFloat(b.dataset.speed) === v));
}

function disposeDynamic() {
  for (const obj of dynamicObjects) {
    scene.remove(obj);
    if (obj.geometry) obj.geometry.dispose();
    if (obj.material) obj.material.dispose();
  }
  dynamicObjects = [];
  pitches = [];
}

function togglePitchVisibility(pitchType, on) {
  const p = pitches.find((x) => x.pitchType === pitchType);
  if (!p) return;
  p.hidden = !on;
  p.line.visible = on;
  p.mesh.visible = on;
  if (p.marker) p.marker.visible = on && p.markerShouldShow;
  renderPairScoreForCheckedTypes();
}

function checkedPitchTypes() {
  return Array.from(document.querySelectorAll("#t3-legend .t3-leg-row input[type=checkbox]:checked"))
    .map((el) => el.dataset.pitchType);
}

function findPairRow(typeA, typeB) {
  if (!pairDetail) return null;
  const [a, b] = [typeA, typeB].sort();
  return pairDetail.pairs.find((r) => r.type_a === a && r.type_b === b) || null;
}

function renderPairScoreForCheckedTypes() {
  const el = document.getElementById("t3-pair-score");
  if (!el) return;
  const checked = checkedPitchTypes();
  if (checked.length !== 2) {
    el.innerHTML = "Check exactly 2 pitch types to see that pair's real Tunneling+ score.";
    return;
  }
  const row = findPairRow(checked[0], checked[1]);
  if (!row) {
    el.innerHTML = `${checked[0]} &harr; ${checked[1]}: not a scored pair (needs 100+ pitches of each, this pitcher's whole-season arsenal).`;
    return;
  }
  el.innerHTML = `
    <b>${row.type_a} &harr; ${row.type_b}</b>${row.is_anchor ? " (anchor pair, counts 2x)" : ""}<br/>
    Release Sim+ ${row.release_similarity_plus.toFixed(0)} &middot; Decision Sim+ ${row.decision_similarity_plus.toFixed(0)}
    &middot; Plate Sep+ ${row.plate_separation_plus.toFixed(0)}<br/>
    Tunneling+ <b>${row.tunneling_plus.toFixed(0)}</b>
  `;
}

function renderSplit(stand) {
  currentStand = stand;
  document.querySelectorAll("#t3-splits button").forEach((b) => b.classList.toggle("active", b.dataset.stand === stand));
  disposeDynamic();
  setBatterSilhouette(stand);

  const split = trajData.splits[stand];
  const overallHtml = pairDetail
    ? `Whole-arsenal Tunneling+: <b>${pairDetail.tunneling_plus.toFixed(0)}</b> &middot; release consistency <b>${pairDetail.release_consistency.toFixed(1)}"</b>`
    : `No validated Tunneling+ score for this pitcher (needs 2+ pitch types with 300+ pitches this season).`;

  const deliveryHtml = deliveryData
    ? `Delivery: real broadcast-video pose data (coming set &rarr; leg lift &rarr; plant &rarr; release), 2D only - depth is interpolated, not measured<br/>`
    : "";

  const tunnelHtml = `
    <div class="t3-tunnel-info">
      Real release, real movement, real average location per pitch type &middot; nothing adjusted<br/>
      Decision point: ${trajData.decision_point_y}ft from plate (same as the validated model) &middot; diamonds = decision point<br/>
      ${deliveryHtml}
      ${overallHtml}<br/>
      <div id="t3-pair-score" style="margin-top:6px;"></div>
    </div>`;

  legendEl.innerHTML = `<h1>${trajData.name} - ${trajData.season} (vs ${stand}HH)</h1>` + (split.pitches.length === 0
    ? `<div class="t3-tunnel-info">No qualifying pitches vs ${stand}HH.</div>`
    : split.pitches.map((p) => `
        <div class="t3-leg-row">
          <input type="checkbox" checked data-pitch-type="${p.pitch_type}" onchange="window.__t3.toggleVisibility('${p.pitch_type}', this.checked)" />
          <span class="t3-swatch" style="background:${p.color};"></span>
          <span>${p.pitch_type} - ${p.velo_mph} mph - ${(p.usage * 100).toFixed(0)}%</span>
        </div>
      `).join("") + tunnelHtml);

  for (const p of split.pitches) {
    const pts = p.points.map(toThree);
    const curve = new THREE.CatmullRomCurve3(pts);
    const geom = new THREE.BufferGeometry().setFromPoints(pts);
    geom.setDrawRange(0, 0);
    const mat = new THREE.LineBasicMaterial({ color: p.color, linewidth: 2 });
    const line = new THREE.Line(geom, mat);
    scene.add(line);
    dynamicObjects.push(line);

    const ball = new THREE.Mesh(
      new THREE.SphereGeometry(0.12, 12, 12),
      new THREE.MeshStandardMaterial({ color: p.color, emissive: p.color, emissiveIntensity: 0.4 })
    );
    ball.position.copy(pts[0]);
    scene.add(ball);
    dynamicObjects.push(ball);

    let marker = null;
    if (p.decision_point) {
      marker = new THREE.Mesh(
        new THREE.OctahedronGeometry(0.16, 0),
        new THREE.MeshStandardMaterial({ color: 0xffffff, emissive: 0xffffff, emissiveIntensity: 0.5 })
      );
      marker.position.copy(toThree(p.decision_point));
      marker.visible = false;
      scene.add(marker);
      dynamicObjects.push(marker);
    }

    pitches.push({
      pitchType: p.pitch_type, curve, geom, nPoints: pts.length, line, mesh: ball, marker,
      t_plate: solveTPlateFromPoints(p.points), releaseVec: pts[0].clone(), hidden: false, markerShouldShow: false,
    });
  }

  if (deliveryData && pitches.length) {
    const avgRelease = pitches.reduce((acc, p) => acc.add(p.releaseVec), new THREE.Vector3()).divideScalar(pitches.length);
    deliveryAnchor = {
      end: avgRelease,
      // Approx rubber position: further from the plate (larger z, toward
      // the mound) and a bit taller (pitchers stand closer to full height
      // at coming_set than at the reaching, bent-forward instant of release).
      start: avgRelease.clone().add(new THREE.Vector3(0, 1.5, 6)),
    };
    deliveryDuration = deliveryData.phases.release;
  } else {
    deliveryAnchor = null;
    deliveryDuration = 0;
  }

  resetToIdle();
  renderPairScoreForCheckedTypes();
}

// The trajectories endpoint samples points evenly in Y (not time), since it
// reuses the model's own position_at_y solve - approximate a per-pitch-type
// playback duration from real velocity instead of an explicit t_plate field.
function solveTPlateFromPoints(points) {
  const p0 = points[0], p1 = points[points.length - 1];
  const dist = Math.hypot(p1.x - p0.x, p1.y - p0.y, p1.z - p0.z);
  return Math.max(0.32, Math.min(0.55, dist / 130)); // ~130 ft/s reference speed, clamped to a realistic MLB range
}

async function loadPitcher(pitcherId, season) {
  currentPitcherId = pitcherId;
  currentSeason = season;
  legendEl.innerHTML = `<h1>Loading...</h1>`;
  const deliveryUrl = DELIVERY_ASSETS[pitcherId];
  const [traj, detail, delivery] = await Promise.all([
    getJSON(`/api/tunneling/${pitcherId}/trajectories?season=${season}`),
    getJSON(`/api/tunneling/pitcher/${pitcherId}?season=${season}`).catch(() => null),
    deliveryUrl ? getJSON(deliveryUrl).catch(() => null) : Promise.resolve(null),
  ]);
  trajData = traj;
  pairDetail = detail;
  deliveryData = delivery;
  renderSplit(currentStand);
}

function onResize() {
  const { w, h } = containerSize();
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  renderer.setSize(w, h);
}
new ResizeObserver(onResize).observe(container);

function animate() {
  requestAnimationFrame(animate);
  const dt = clock.getDelta();
  if (playing) {
    elapsed += dt * speed;

    if (deliveryData) {
      if (elapsed < deliveryDuration) {
        skeletonGroup.visible = true;
        updateSkeletonPose(elapsed);
      } else if (skeletonGroup.visible) {
        // Release handoff - the real ball trajectory takes over from here,
        // so the stylized skeleton (no depth data past this point anyway)
        // steps aside rather than drifting out of sync with it.
        skeletonGroup.visible = false;
      }
    }

    let anyMoving = false;
    for (const p of pitches) {
      if (p.hidden) continue;
      const frac = Math.min(1, (elapsed - deliveryDuration) / p.t_plate);
      if (frac > 0) {
        if (frac < 1) anyMoving = true;
        p.curve.getPointAt(Math.max(0, frac), p.mesh.position);
        p.geom.setDrawRange(0, Math.max(1, Math.round(Math.max(0, frac) * (p.nPoints - 1)) + 1));
        p.markerShouldShow = true;
        if (p.marker) p.marker.visible = true;
      } else {
        anyMoving = true; // still in the pre-release delivery phase
      }
    }
    if (!anyMoving) {
      playing = false;
      document.getElementById("t3-btn-play").textContent = "Play";
    }
  }
  controls.update();
  renderer.render(scene, camera);
}
animate();

window.__t3 = {
  setView, setSplit: renderSplit, togglePlay, restart: resetToIdle, setSpeed, toggleVisibility: togglePitchVisibility,
};

async function init() {
  const seasonSelect = createSeasonSelect(document.getElementById("season-select"), (season) => {
    if (currentPitcherId) loadPitcher(currentPitcherId, season);
  });
  const defaultSeason = await seasonSelect.load();

  const params = new URLSearchParams(location.search);
  const urlPitcherId = params.get("pitcher_id");

  const search = createPlayerSearch(
    document.getElementById("player-search-input"),
    document.getElementById("player-search-dropdown"),
    (player) => {
      document.getElementById("t3-status").textContent = "";
      loadPitcher(player.player_id, seasonSelect.value);
    },
    { roleFilter: "pitcher" }
  );
  await search.load(defaultSeason);

  const remembered = PlayerContext.getCurrent();
  if (urlPitcherId) {
    document.getElementById("t3-status").textContent = "";
    loadPitcher(Number(urlPitcherId), Number(params.get("season")) || defaultSeason);
  } else if (remembered && remembered.role === "pitcher") {
    search.setValue(remembered);
    loadPitcher(remembered.player_id, defaultSeason);
  }
}
document.addEventListener("DOMContentLoaded", init);
