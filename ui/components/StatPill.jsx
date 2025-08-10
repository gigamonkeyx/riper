export default function StatPill({tone="good",children}) {
  const c = {good:"bg-[color:var(--good)]/15 text-[color:var(--good)]",
             warn:"bg-[color:var(--warn)]/15 text-[color:var(--warn)]",
             bad:"bg-[color:var(--bad)]/15 text-[color:var(--bad)]"}[tone];
  return <span className={`px-2 py-0.5 rounded-full text-xs ${c}`}>{children}</span>;
}

