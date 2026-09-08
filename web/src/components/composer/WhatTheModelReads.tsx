import { useMemo, useState } from "react";
import type { ReferenceItem } from "../../types";

/**
 * Continuity-style "what the model reads" panel.
 *
 * Renders a dynamic header mapping every loaded reference to the token the
 * model consumes (Picture N / Video N / Audio N), plus an OPTIONAL elaborator
 * that reproduces the structured breakdown the original app builds before a
 * run:
 *
 *   subject_definitions
 *   summary
 *   retention_analysis
 *   detailed_description
 *   overall_soundscape
 *
 * The elaborator is a best-effort, client-side synthesis from the loaded
 * references and the current prompt, mirroring the layout of the original
 * Continuity output so testers can see what the model is being told.
 */

interface WhatTheModelReadsProps {
  refs: ReferenceItem[];
  prompt: string;
  /** Optional pre-composed breakdown, if one is ever provided by the backend. */
  elaboration?: ModelReadout;
  disabled?: boolean;
}

export interface ModelReadout {
  subject_definitions: string[];
  summary: string;
  retention_analysis: string[];
  detailed_description: string;
  overall_soundscape: string;
}

interface TokenDescriptor {
  token: string;
  kindLabel: string;
  name: string;
  audioName?: string;
}

function tokenForIndex(refs: ReferenceItem[], index: number): TokenDescriptor {
  const ref = refs[index];
  let count = 0;
  let kindLabel = "Reference";
  for (let i = 0; i <= index; i++) {
    const r = refs[i];
    if (ref.kind === "image" && r.kind === "image") count++;
    else if (ref.kind === "audio" && r.kind === "audio") count++;
    else if (
      (ref.kind === "video" || ref.kind === "silent_video" || ref.kind === "video_audio") &&
      (r.kind === "video" || r.kind === "silent_video" || r.kind === "video_audio")
    )
      count++;
  }
  if (ref.kind === "image") {
    kindLabel = "Picture";
    return { token: `Picture ${count}`, kindLabel, name: ref.name, audioName: ref.audioName };
  }
  if (ref.kind === "audio") {
    kindLabel = "Audio";
    return { token: `Audio ${count}`, kindLabel, name: ref.name, audioName: ref.audioName };
  }
  kindLabel = ref.kind === "silent_video" ? "Silent video" : ref.kind === "video_audio" ? "Video + audio" : "Video";
  return { token: `Video ${count}`, kindLabel, name: ref.name, audioName: ref.audioName };
}

function buildReadout(refs: ReferenceItem[], prompt: string): ModelReadout {
  const tokens = refs.map((_, i) => tokenForIndex(refs, i));
  const pictures = tokens.filter((t) => t.token.startsWith("Picture "));
  const videos = tokens.filter((t) => t.token.startsWith("Video ") || t.kindLabel === "Silent video");
  const audios = tokens.filter((t) => t.token.startsWith("Audio "));
  const hasAudio = audios.length > 0;

  // subject_definitions
  const subject_definitions: string[] = [];
  for (const t of pictures) {
    subject_definitions.push(
      `${t.token} is a reference picture. What the target video takes from it is what the picture actually shows.`
    );
  }
  if (hasAudio) {
    subject_definitions.push(
      `${audios.map((a) => a.token).join(" and ")} ${audios.length === 1 ? "is" : "are"} reference audio clips.`
    );
  }
  for (const t of videos) {
    subject_definitions.push(
      `${t.token} is a reference video. The target video borrows its motion, framing, and overall look.`
    );
  }
  if (subject_definitions.length === 0) {
    subject_definitions.push(
      "No references loaded. The target video is generated purely from the text prompt."
    );
  }

  // summary
  const kindWords: string[] = [];
  if (pictures.length) kindWords.push("image reference");
  if (videos.length) kindWords.push("video reference");
  if (audios.length) kindWords.push("audio reference");
  let summary: string;
  if (subject_definitions.length === 0) {
    summary = "text-to-video: the target video is generated from the prompt alone.";
  } else {
    summary = `[reference generation + ${kindWords.join(" + ")}] The target video runs one shot.`;
    if (hasAudio) summary += " The audio clip guides the target video's sound.";
  }

  // retention_analysis
  const retention: string[] = [];
  const shotLabel = "Shot 1";
  for (const t of pictures) {
    retention.push(
      `${t.token} (appears in [${shotLabel}]): fully_preserved - what the picture shows is carried into the target video.`
    );
  }
  for (const t of videos) {
    retention.push(
      `${t.token} (appears in [${shotLabel}]): motion_and_framing - the target video follows the reference video's motion.`
    );
  }
  if (hasAudio) {
    for (const t of audios) {
      retention.push(`${t.token}: reference - the clip guides the target video's sound without being copied.`);
    }
  }
  if (retention.length === 0) {
    retention.push("No references loaded - no retention constraints. The prompt drives everything.");
  }

  // detailed_description — mirrors the prompt, making the references visible
  let detailed_description = prompt.trim();
  if (!detailed_description) {
    detailed_description = "[no prompt yet] Describe the scene, action, camera, and audio.";
  }
  if (tokens.length > 0) {
    const mention = tokens.filter((t) => !detailed_description.includes(t.token));
    if (mention.length > 0) {
      const suffix = ` The video should match ${mention.map((m) => m.token).join(" and ")}.`;
      detailed_description += suffix;
    }
  }

  // overall_soundscape
  let overall_soundscape = "public space. wind on microphone and sounds of nature";
  if (hasAudio) {
    overall_soundscape = `${audios.map((a) => a.token).join(" + ")} provide the reference for the sound. ` + overall_soundscape;
  }
  overall_soundscape += "\nnon_diegetic_music: the dialogue stays clear";

  return {
    subject_definitions,
    summary,
    retention_analysis: retention,
    detailed_description,
    overall_soundscape,
  };
}

export function WhatTheModelReads({
  refs,
  prompt,
  elaboration,
  disabled,
}: WhatTheModelReadsProps) {
  const [showDetail, setShowDetail] = useState(false);
  const tokens = refs.map((_, i) => tokenForIndex(refs, i));
  const computed = useMemo(() => buildReadout(refs, prompt), [refs, prompt]);
  const readout = elaboration ?? computed;

  return (
    <div className={`wtmr ${disabled ? "wtmr--disabled" : ""}`}>
      {/* Dynamic header: the references the model consumes */}
      <div className="wtmr__header">
        <span className="wtmr__label">The model reads:</span>
        {tokens.length === 0 ? (
          <span className="wtmr__none">(no references — prompt only)</span>
        ) : (
          <div className="wtmr__tokens">
            {tokens.map((t, i) => (
              <span key={i} className={`wtmr__token wtmr__token--${t.token.split(" ")[0].toLowerCase()}`}>
                {t.token}
                <span className="wtmr__token-name">{t.name}</span>
              </span>
            ))}
          </div>
        )}
        <button
          type="button"
          className="wtmr__toggle"
          onClick={() => setShowDetail((v) => !v)}
          disabled={disabled}
        >
          {showDetail ? "Hide details" : "What the model reads"}
        </button>
      </div>

      {/* Optional elaboration */}
      {showDetail && (
        <div className="wtmr__body">
          <section>
            <h4>subject_definitions</h4>
            {readout.subject_definitions.map((line, i) => (
              <p key={i}>{line}</p>
            ))}
          </section>
          <section>
            <h4>summary</h4>
            <p>{readout.summary}</p>
          </section>
          <section>
            <h4>retention_analysis</h4>
            {readout.retention_analysis.map((line, i) => (
              <p key={i}>{line}</p>
            ))}
          </section>
          <section>
            <h4>detailed_description</h4>
            <pre className="wtmr__pre">{readout.detailed_description}</pre>
          </section>
          <section>
            <h4>overall_soundscape</h4>
            <pre className="wtmr__pre">{readout.overall_soundscape}</pre>
          </section>
        </div>
      )}
    </div>
  );
}
