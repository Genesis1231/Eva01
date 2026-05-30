import { memo, useEffect, useMemo, useState } from "react";
import ReactPlayer from "react-player";
import ReactMarkdown from "react-markdown";
import DocViewer, { DocViewerRenderers } from "@cyntler/react-doc-viewer";
import "@cyntler/react-doc-viewer/dist/index.css";

// Stable references so DocViewer doesn't re-init when FileDoc re-renders.
const DOC_CONFIG = {
  header: { disableHeader: true, disableFileName: true },
  pdfVerticalScrollByDefault: true,
  pdfZoom: { defaultZoom: 1, zoomJump: 0.2 },
};
const DOC_STYLE = { height: "100%", width: "100%" };

// Everything shows the WHOLE artifact, never cropped: real media (image/browser/
// video) via FIT (object-contain), and the art scene via preserveAspectRatio="meet".
// The art breathes (capped ≤1 so it never zooms past the frame; frozen in
// capture-mode). Docs keep their own width-fit + vertical scroll.
const FIT = "absolute inset-0 h-full w-full object-contain";
const ART = `${FIT} animate-breathe`;

function MorningLight() {
  return (
    <svg
      className={ART}
      viewBox="0 0 800 860"
      preserveAspectRatio="xMidYMid meet"
      role="img"
      aria-label="morning light"
    >
      <defs>
        <linearGradient id="ml-sky" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor="#15131f" />
          <stop offset="0.45" stopColor="#34293a" />
          <stop offset="0.7" stopColor="#7d5a55" />
          <stop offset="0.86" stopColor="#c1885f" />
          <stop offset="1" stopColor="#e8b98a" />
        </linearGradient>
        <radialGradient id="ml-sun" cx="50%" cy="82%" r="55%">
          <stop offset="0" stopColor="#ffeccb" stopOpacity="0.95" />
          <stop offset="0.32" stopColor="#ffd5a0" stopOpacity="0.5" />
          <stop offset="1" stopColor="#ffd5a0" stopOpacity="0" />
        </radialGradient>
        <linearGradient id="ml-band" x1="0" y1="0" x2="1" y2="0">
          <stop offset="0" stopColor="#e8c79a" stopOpacity="0" />
          <stop offset="0.5" stopColor="#f0d7b2" stopOpacity="0.45" />
          <stop offset="1" stopColor="#cdaf86" stopOpacity="0" />
        </linearGradient>
        <filter id="ml-soft"><feGaussianBlur stdDeviation="9" /></filter>
        <filter id="ml-softer"><feGaussianBlur stdDeviation="22" /></filter>
      </defs>
      <rect width="800" height="860" fill="url(#ml-sky)" />
      <g fill="#f3ecd9" opacity="0.55">
        <circle cx="120" cy="80" r="1.2" />
        <circle cx="320" cy="54" r="1" />
        <circle cx="540" cy="100" r="1.3" />
        <circle cx="690" cy="66" r="1" />
        <circle cx="220" cy="150" r="1" />
        <circle cx="610" cy="170" r="1.1" />
      </g>
      <ellipse cx="400" cy="700" rx="520" ry="290" fill="url(#ml-sun)" filter="url(#ml-softer)" />
      <circle cx="400" cy="694" r="58" fill="#fff3df" opacity="0.92" filter="url(#ml-soft)" />
      <path d="M -40 420 C 230 350, 560 460, 860 372" stroke="url(#ml-band)" strokeWidth="24" fill="none" opacity="0.5" filter="url(#ml-soft)" />
      <path d="M -40 506 C 250 458, 540 548, 860 488" stroke="url(#ml-band)" strokeWidth="16" fill="none" opacity="0.4" filter="url(#ml-soft)" />
      <rect x="0" y="752" width="800" height="2" fill="#ffe6c2" opacity="0.32" />
      <rect x="0" y="788" width="800" height="72" fill="#0d0b12" opacity="0.92" />
      <rect x="96" y="766" width="150" height="26" rx="4" fill="#0d0b12" opacity="0.85" />
      <ellipse cx="600" cy="788" rx="32" ry="8" fill="#0d0b12" opacity="0.8" />
    </svg>
  );
}

// A page she's looking at — the url under slim browser chrome. The live page
// can't be embedded (most sites block iframing via CSP / X-Frame-Options), so a
// rendered frame, when needed, arrives separately as an `image` artifact; this
// surface just marks what she has open.
function Browser({ url }) {
  return (
    <div className="flex h-full w-full flex-col bg-ink">
      <div className="flex shrink-0 items-center gap-2 border-b border-rule bg-black/30 px-4 py-2.5">
        <span className="h-2.5 w-2.5 rounded-full bg-rule" />
        <span className="h-2.5 w-2.5 rounded-full bg-rule" />
        <span className="h-2.5 w-2.5 rounded-full bg-rule" />
        <span className="ml-3 truncate font-sans text-xs text-meta">{url}</span>
      </div>
      <div className="grid min-h-0 flex-1 place-items-center bg-[#14121a] font-sans text-sm text-meta">
        opening {url}…
      </div>
    </div>
  );
}

// Something she's reading — markdown via react-markdown (@cyntler doesn't format
// .md); any other document (PDF, DOCX, PPTX, XLSX, images) via the @cyntler
// multi-format viewer.
function Doc({ src }) {
  if (/\.md(\?|$)/i.test(src || "")) return <MarkdownDoc src={src} />;
  return <FileDoc src={src} />;
}

// PDF / DOCX / PPTX / XLSX / images — one multi-format viewer, chrome stripped
// for a clean canvas surface.
function FileDoc({ src }) {
  // Memoized so the array identity is stable across re-renders — otherwise a new
  // `[{ uri }]` each render makes DocViewer re-fetch and reload the document.
  const documents = useMemo(() => [{ uri: src }], [src]);
  return (
    <div className="doc-viewer h-full w-full overflow-hidden bg-white">
      <DocViewer
        documents={documents}
        pluginRenderers={DocViewerRenderers}
        config={DOC_CONFIG}
        style={DOC_STYLE}
      />
    </div>
  );
}

// A markdown note — fetched and rendered with @tailwindcss/typography defaults.
// Guards the fetch: a non-2xx status (404) or an HTML fallback page must NOT be
// rendered as markdown, and a network error is caught — all surface as an error
// state instead of leaking the error body into the document.
function MarkdownDoc({ src }) {
  // result is tagged with its src; a result from a previous src is ignored, which
  // also blanks the view while a new src loads (no synchronous setState needed).
  const [result, setResult] = useState({ src: null, text: "", error: null });
  useEffect(() => {
    let on = true;
    fetch(src)
      .then((res) => {
        if (!res.ok) throw new Error(`${res.status} ${res.statusText}`.trim());
        if ((res.headers.get("content-type") || "").includes("text/html")) {
          throw new Error("not a document");
        }
        return res.text();
      })
      .then((text) => { if (on) setResult({ src, text, error: null }); })
      .catch((e) => { if (on) setResult({ src, text: "", error: e.message || "failed to load" }); });
    return () => { on = false; };
  }, [src]);

  const current = result.src === src ? result : { text: "", error: null };
  if (current.error) {
    return (
      <div className="grid h-full w-full place-items-center bg-[#161320] px-10 text-center font-sans text-sm text-meta">
        couldn’t open {src} — {current.error}
      </div>
    );
  }
  return (
    <div className="h-full w-full overflow-y-auto bg-[#161320]">
      <article className="prose prose-invert prose-sm mx-auto max-w-2xl px-10 py-10">
        <ReactMarkdown>{current.text}</ReactMarkdown>
      </article>
    </div>
  );
}

// A clip on her desk — a lightweight react-player that autoplays once on load
// (muted, no loop — muted so the browser actually allows autoplay). On /canvas
// (capture) it holds a still placeholder instead, so the embedding stays
// deterministic. The inner <video> fills via the #canvas rule in globals.css.
function Video({ src, capture }) {
  if (!src || capture) {
    return (
      <div className="relative h-full w-full bg-black">
        <div className="h-full w-full bg-gradient-to-b from-[#1b2430] to-[#0b0e14]" />
      </div>
    );
  }
  return (
    <div className="relative h-full w-full bg-black">
      <ReactPlayer src={src} playing muted controls playsInline width="100%" height="100%" />
    </div>
  );
}

// A raw image (webcam frame, an upload) — the source itself, shown whole
// (object-contain) so the capture matches the file pixel-for-pixel.
function Photo({ src }) {
  if (src) return <img className={FIT} src={src} alt="" />;
  return (
    <div className="relative h-full w-full bg-gradient-to-br from-[#2a2230] via-[#1a1620] to-[#100d16]">
      <div className="absolute bottom-0 h-1/4 w-full bg-[#0d0b12]/90" />
      <div className="absolute left-[18%] top-[36%] h-24 w-24 rounded-full bg-parchment/10 blur-xl" />
    </div>
  );
}

// A canvas surface is posted as (kind, text): the kind IS the surface type
// (image / doc / video / browser / art); the text is its source path or url.
// No guessing — Eva names the type she already knows at the source.
function payloadFor(kind, text) {
  if (kind === "browser") return { url: text };
  if (kind === "art") return {};
  return { src: text }; // image, doc, video
}

function Surface({ mode, payload, capture }) {
  switch (mode) {
    case "browser":
      return <Browser {...payload} />;
    case "doc":
      return <Doc {...payload} />;
    case "video":
      return <Video {...payload} capture={capture} />;
    case "image":
      return <Photo {...payload} />;
    case "art":
    default:
      return <MorningLight />;
  }
}

// The artifact only — never a caption. The panel (Room) gets a drop shadow; the
// full-bleed /canvas capture surface gets none. Memoized on kind+text: the Room
// re-renders on every stream/mood event, but the canvas should only re-render
// when the surface changes — otherwise the PDF reloads / video re-nudges.
function Canvas({ kind = "art", text, variant = "panel" }) {
  const panel = variant !== "full";
  return (
    <div id="canvas" className={"relative h-full w-full overflow-hidden" + (panel ? " shadow-2xl" : "")}>
      <Surface mode={kind} payload={payloadFor(kind, text)} capture={!panel} />
    </div>
  );
}

export default memo(Canvas);
