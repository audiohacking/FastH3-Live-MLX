import { compilePrompt, handleForRef, nearestDurationId, tokensForRefs } from "./compile";
import type { CastMember, ReferenceItem } from "./types";

function assert(cond: unknown, msg: string): asserts cond {
  if (!cond) throw new Error(msg);
}

function img(partial: Partial<ReferenceItem> & Pick<ReferenceItem, "id" | "path">): ReferenceItem {
  return { kind: "image", name: partial.name ?? partial.path, enabled: true, ...partial };
}

function aud(partial: Partial<ReferenceItem> & Pick<ReferenceItem, "id" | "path">): ReferenceItem {
  return { kind: "audio", name: partial.name ?? partial.path, enabled: true, ...partial };
}

const anna: CastMember = {
  id: "cast_anna",
  name: "Anna",
  description: "blonde woman",
  media: [{ id: "m1", type: "image", path: "/tmp/anna.png", label: "anna still" }],
  createdAt: "",
  updatedAt: "",
};

const bob: CastMember = {
  id: "cast_bob",
  name: "Bob",
  description: "man in a coat",
  media: [],
  createdAt: "",
  updatedAt: "",
};

export function runCompileSelfTest() {
  const refs: ReferenceItem[] = [
    img({ id: "1", path: "/a.png", name: "face" }),
    aud({ id: "2", path: "/a.wav", name: "voice", durationS: 8.01 }),
  ];
  assert(handleForRef(refs, 0) === "@img-1", "img handle");
  assert(handleForRef(refs, 1) === "@aud-1", "aud handle");
  assert(tokensForRefs(refs)[0].token === "Picture 1", "picture token");
  assert(tokensForRefs(refs)[1].token === "Audio 1", "audio token");

  const compiled = compilePrompt({
    prompt: "Keep @img-1 talking to the camera. Voice from @aud-1.",
    refs,
    castMembers: [],
    selectedCastIds: [],
    routing: "ref2va",
  });
  assert(compiled.compiledPrompt.includes("Picture 1"), compiled.compiledPrompt);
  assert(compiled.compiledPrompt.includes("Audio 1"), compiled.compiledPrompt);
  assert(!compiled.compiledPrompt.includes("@img-1"), "handle rewritten");
  assert(compiled.hadAtTokens, "detected @ tokens");
  assert(compiled.modeHint === "ref2va", "mode");

  const muted = compilePrompt({
    prompt: "Use @img-1",
    refs: [img({ id: "1", path: "/a.png", enabled: false }), img({ id: "2", path: "/b.png", name: "b" })],
    castMembers: [],
    selectedCastIds: [],
    routing: "ref2va",
  });
  assert(muted.refs.length === 1 && muted.refs[0].path === "/b.png", "muted omitted");
  assert(muted.compiledPrompt.includes("Picture 1"), "muted numbering");

  const passthrough = compilePrompt({
    prompt: "A quiet kitchen, no handles.",
    refs,
    castMembers: [anna],
    selectedCastIds: [],
    routing: "ref2va",
  });
  assert(passthrough.compiledPrompt === "A quiet kitchen, no handles.", "no @ passthrough");
  assert(passthrough.refs.length === 2, "no extra refs without selection");

  const selected = compilePrompt({
    prompt: "A quiet kitchen, no handles.",
    refs: [],
    castMembers: [anna],
    selectedCastIds: ["cast_anna"],
    routing: "ref2va",
  });
  assert(selected.refs.length === 1 && selected.refs[0].path === "/tmp/anna.png", "selected injects media");
  assert(selected.compiledPrompt === "A quiet kitchen, no handles.", "selected does not rewrite prompt");

  const mentioned = compilePrompt({
    prompt: "@Anna walks in.",
    refs: [],
    castMembers: [anna],
    selectedCastIds: [],
    routing: "ref2va",
  });
  assert(mentioned.compiledPrompt.includes("Picture 1"), mentioned.compiledPrompt);
  assert(mentioned.refs[0].castId === "cast_anna", "mention injects");

  const annaVoice: CastMember = {
    ...anna,
    media: [
      ...anna.media,
      { id: "m2", type: "audio", path: "/tmp/anna.wav", label: "anna voice", durationS: 6.2 },
    ],
  };
  const voiced = compilePrompt({
    prompt: "@Anna says hello.",
    refs: [],
    castMembers: [annaVoice],
    selectedCastIds: [],
    routing: "ref2va",
  });
  assert(voiced.refs.some((r) => r.kind === "image" && r.path === "/tmp/anna.png"), "cast still");
  assert(voiced.refs.some((r) => r.kind === "audio" && r.path === "/tmp/anna.wav"), "cast voice");
  assert(voiced.compiledPrompt.includes("Picture 1"), voiced.compiledPrompt);
  assert(voiced.compiledPrompt.includes("Audio 1"), voiced.compiledPrompt);

  const descOnly = compilePrompt({
    prompt: "@Bob nods.",
    refs: [],
    castMembers: [bob],
    selectedCastIds: [],
    routing: "auto",
  });
  assert(descOnly.compiledPrompt.includes("Bob nods"), descOnly.compiledPrompt);
  assert(descOnly.compiledPrompt.includes("Bob: man in a coat"), "description appended");
  assert(descOnly.refs.length === 0, "no fake refs");
  assert(descOnly.modeHint === "t2va", "auto without refs is t2va");

  const autoRefs = compilePrompt({
    prompt: "hello",
    refs,
    castMembers: [],
    selectedCastIds: [],
    routing: "auto",
  });
  assert(autoRefs.modeHint === "ref2va", "auto with refs");

  const fl2va = compilePrompt({
    prompt: "hello",
    refs,
    castMembers: [],
    selectedCastIds: [],
    routing: "fl2va",
    imagePath: "/start.png",
  });
  assert(fl2va.modeHint === "first_frame", "fl2va first");
  assert(fl2va.refs.length === 0, "fl2va drops refs");

  assert(nearestDurationId(8.01, [
    { id: "5s", seconds: 5 },
    { id: "10s", seconds: 10 },
    { id: "15s", seconds: 15 },
  ]) === "10s", "snap duration up");
}

runCompileSelfTest();
console.log("compile tests ok");
