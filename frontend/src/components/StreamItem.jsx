/**
 * One line of Eva's sensorimotor log — small and sans, like a log. A time +
 * what she sensed or said. Senses (kind "sense") render roman; her own speech
 * (kind "speech") renders italic with gold quotes; the latest line is brightened.
 */
function fmtTime(ts) {
  const d = new Date(ts);
  const h = d.getHours() % 12 || 12;
  return `${h}:${String(d.getMinutes()).padStart(2, "0")}`;
}

export default function StreamItem({ kind, text, ts, now }) {
  const isSpeech = kind === "speech";
  const txt = [
    "font-sans text-sm",
    isSpeech ? "italic text-parchment" : now ? "text-parchment" : "text-parchment-dim",
    isSpeech
      ? "before:text-gold before:content-['“'] after:text-gold after:content-['”']"
      : "",
  ]
    .filter(Boolean)
    .join(" ");

  return (
    <div className="flex animate-rise items-baseline gap-3">
      <time className="w-12 shrink-0 font-sans text-xs tabular-nums text-meta">{fmtTime(ts)}</time>
      <span className={txt}>{text}</span>
    </div>
  );
}
