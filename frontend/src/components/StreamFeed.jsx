import StreamItem from "./StreamItem";

/**
 * The living log of senses + actions (2/5 column), vertically centered and
 * dissolving symmetrically at top and bottom via the mask. Latest line is
 * brightened.
 */
export default function StreamFeed({ items }) {
  return (
    <section className="col-span-2 flex min-h-0 flex-col">
      <div className="stream-mask flex min-h-0 flex-1 flex-col justify-center gap-4 overflow-hidden">
        {items.map((it, i) => (
          <StreamItem key={it.id} {...it} now={i === items.length - 1} />
        ))}
      </div>
    </section>
  );
}
