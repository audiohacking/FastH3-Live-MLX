import { durationIdForClip, loraIdsFromClip, resolutionIdForClip, snapshotFromClip } from "./clipEditor";
import type { Clip, Config, LoraPreset } from "./types";

function assert(cond: unknown, msg: string): asserts cond {
  if (!cond) throw new Error(msg);
}

const config = {
  resolution_presets: [
    { id: "512x512", label: "512", width: 512, height: 512 },
    { id: "512x512-aggressive", label: "512 agg", width: 512, height: 512, render_width: 384, render_height: 384 },
  ],
  duration_presets: [
    { id: "1s", label: "1s", seconds: 1, num_frames: 22 },
    { id: "2s", label: "2s", seconds: 2, num_frames: 39 },
  ],
} as Config;

const defaults = { numSteps: 20, layers: 50, reuse: 1, quality: "fast" };

function clip(partial: Partial<Clip> & Pick<Clip, "id">): Clip {
  return {
    prompt: "compiled Picture 1",
    label: "CURRENT",
    video_url: "/api/videos/a.mp4",
    filename: "a.mp4",
    chain_id: "c1",
    clip_index: 0,
    mode: "t2va",
    status: "done",
    created_at: "2026-01-01",
    ...partial,
  };
}

const old = clip({ id: "old", mode: "ref2va", prompt: "a person talks", width: 512, height: 512, num_frames: 22, seed: 7 });
const snapOld = snapshotFromClip(old, config, defaults);
assert(snapOld.prompt === "a person talks", "old prompt");
assert(snapOld.mode === "ref2va", "old mode");
assert(snapOld.routing === "ref2va", "old routing from mode");
assert(snapOld.refs.length === 0, "old clips have no refs");
assert(snapOld.seed === "7", "old seed");

const catalog: LoraPreset[] = [
  { id: "turbo1", label: "Turbo", spec: "tutututututu/Tutu-MiniMax-H3-AudioVideo-20to8-NFE-LoRA", scale: 0.8 },
];

const fresh = clip({
  id: "new",
  mode: "ref2va",
  prompt: "Picture 1 walks",
  width: 512,
  height: 512,
  num_frames: 39,
  recipe: {
    composer_prompt: "@img-1 walks",
    routing: "ref2va",
    mode: "ref2va",
    resolution_id: "512x512-aggressive",
    duration_id: "2s",
    token_reduction: false,
    ssd_streaming: true,
    turbo_enabled: true,
    turbo_tier: "medium",
    selected_cast_ids: ["cast_anna"],
    lora_preset_ids: ["gone"],
    loras: [{ spec: "tutututututu/Tutu-MiniMax-H3-AudioVideo-20to8-NFE-LoRA" }],
    refs: [
      { kind: "image", path: "/uploads/a.png", name: "face.png", available: true, preview_url: "/api/uploads/a.png" },
      { kind: "audio", path: "/uploads/gone.wav", name: "voice.wav", available: false },
    ],
    image_path: "/frames/x.png",
    image_path_available: false,
    image_name: "start.png",
  },
});

assert(resolutionIdForClip(fresh, config) === "512x512-aggressive", "recipe resolution");
assert(durationIdForClip(fresh, config) === "2s", "recipe duration");
assert(loraIdsFromClip(fresh, catalog)[0] === "turbo1", "lora matched by spec");

const snap = snapshotFromClip(fresh, config, defaults, catalog);
assert(snap.prompt === "@img-1 walks", "composer prompt restored");
assert(snap.routing === "ref2va", "routing");
assert(snap.refs.length === 1 && snap.refs[0].name === "face.png", "available ref kept");
assert(snap.refs[0].previewUrl === "/api/uploads/a.png", "preview url");
assert(snap.imagePath === null, "missing start frame dropped");
assert(snap.turboEnabled === true && snap.turboTier === "medium", "turbo");
assert(snap.tokenReduction === false && snap.ssdStreaming === true, "sampler flags");
assert(snap.selectedCastIds[0] === "cast_anna", "cast");
assert(snap.missingMedia.includes("voice.wav"), "missing audio named");
assert(snap.missingMedia.includes("start.png"), "missing frame named");

console.log("clipEditor tests ok");
