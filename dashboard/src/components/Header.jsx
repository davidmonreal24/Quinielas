export default function Header({ summary, onRefetch }) {
  const fechaMin = summary?.fecha_min ?? '—'
  const fechaMax = summary?.fecha_max ?? '—'
  const hasOdds  = summary?.has_odds ?? false

  return (
    <header className="sticky top-0 z-30 bg-[#0a0f1e]/90 backdrop-blur border-b border-white/5">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 py-4 flex items-center justify-between gap-4">

        {/* Logo + title */}
        <div className="flex items-center gap-3 min-w-0">
          <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-blue-500 to-indigo-600 flex items-center justify-center shrink-0 shadow-lg shadow-blue-500/20">
            <svg className="w-5 h-5 text-white" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path strokeLinecap="round" strokeLinejoin="round" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
            </svg>
          </div>
          <div className="min-w-0">
            <h1 className="text-base font-bold text-white leading-tight truncate">Data Analysis Picks</h1>
            <p className="text-xs text-gray-400 truncate">Liga MX · Clausura 2026</p>
          </div>
        </div>

        {/* Date range */}
        {summary && (
          <div className="hidden sm:flex items-center gap-2 text-xs text-gray-400 bg-white/5 rounded-lg px-3 py-2">
            <svg className="w-3.5 h-3.5 text-gray-500 shrink-0" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <rect x="3" y="4" width="18" height="18" rx="2" ry="2"/><line x1="16" y1="2" x2="16" y2="6"/><line x1="8" y1="2" x2="8" y2="6"/><line x1="3" y1="10" x2="21" y2="10"/>
            </svg>
            <span>{fechaMin} – {fechaMax}</span>
            {hasOdds && (
              <span className="ml-2 px-1.5 py-0.5 rounded bg-emerald-500/20 text-emerald-400 font-medium">con momios</span>
            )}
          </div>
        )}

        {/* Refresh */}
        <button
          onClick={onRefetch}
          className="shrink-0 p-2 rounded-lg text-gray-400 hover:text-white hover:bg-white/10 transition-colors"
          title="Actualizar datos"
        >
          <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <polyline points="23 4 23 10 17 10"/><polyline points="1 20 1 14 7 14"/>
            <path d="M3.51 9a9 9 0 0114.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0020.49 15"/>
          </svg>
        </button>
      </div>
    </header>
  )
}
