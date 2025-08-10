export default function TopBar(){
  return (
    <header role="banner" className="h-14 cyber-panel rounded-xl mx-3 mt-3 mb-0 flex items-center justify-between px-4">
      <div className="text-sm text-[color:var(--muted-400)]">Rural Systems Simulator</div>
      <div className="flex items-center gap-2">
        <a className="btn-cyber text-sm" href="/funding">Funding</a>
        <a className="btn-cyber text-sm" href="/terminal">Terminal</a>
      </div>
    </header>
  );
}

