import { useClock } from "../util/useClock";

/**
 * The top bar: date + time on the left, and on the right Eva's state as a living
 * sentence — "Eva is curious, tender" — her identity and mood in one line.
 * Owns its own clock so the per-second tick re-renders only this bar, not the
 * whole Room (which would churn the canvas — reloading the PDF, nudging the video).
 */
function MoodSentence({ mood }) {
  // mood is a string like "curious, tender" — the first (strongest) word leads
  // in gold, the rest in parchment. Empty ⇒ she hasn't felt anything yet.
  const parts = (mood || "").split(",").map((s) => s.trim()).filter(Boolean);
  if (parts.length === 0) return <span className="text-meta">Eva is asleep. zzZ</span>;
  return (
    <>
      <span className="text-meta">Eva is feeling: </span>{" "}
      {parts.map((word, i) => (
        <span key={i}>
          {i > 0 && ", "}
          <span className={i === 0 ? "text-gold" : "text-parchment"}>{word}</span>
        </span>
      ))}
    </>
  );
}

export default function InnerState({ mood }) {
  const { time, meridiem, date } = useClock();
  return (
    <header className="flex items-baseline justify-between border-b border-rule pb-5">
      <div className="flex items-end gap-3">
        <span className="flex items-end gap-1.5">
          <span className="font-sc text-2xl font-medium leading-none tracking-wider text-parchment">{time}</span>
          <span className="font-sc text-2xl font-medium leading-none tracking-wider text-parchment">{meridiem}</span>
        </span>
        <span className="pb-[0.08rem] font-sans text-xs uppercase leading-none tracking-widest text-meta">{date}</span>
      </div>
      <div className="text-xl italic text-parchment-dim">
        <MoodSentence mood={mood} />
      </div>
    </header>
  );
}
