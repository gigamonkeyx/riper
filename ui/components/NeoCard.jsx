export default function NeoCard({children,className=""}) {
  return (
    <div className={`rounded-xl border border-[color:var(--border)]
      bg-[color:var(--panel-800)]/90 backdrop-blur-sm shadow-[0_8px_24px_rgba(0,0,0,.35)]
      hover:shadow-[0_12px_28px_rgba(0,0,0,.45)] transition ${className}`}>
      {children}
    </div>
  );
}

