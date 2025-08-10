export default function Sidebar(){
  return (
    <aside className="w-full xl:w-60 cyber-panel rounded-xl h-[calc(100vh-6rem)] p-3 sticky top-24">
      <div className="text-[color:var(--text-100)] font-semibold mb-4 neon-text">Tonasket</div>
      <nav className="space-y-1 text-[color:var(--muted-400)]">
        <a href="/" className="block px-3 py-2 rounded hover:bg-[color:var(--panel-700)] hover:neon-border transition">Dashboard</a>
        <a href="/funding" className="block px-3 py-2 rounded hover:bg-[color:var(--panel-700)] hover:neon-border transition">Funding</a>
        <a href="/terminal" className="block px-3 py-2 rounded hover:bg-[color:var(--panel-700)] hover:neon-border transition">Terminal</a>
      </nav>
    </aside>
  );
}

