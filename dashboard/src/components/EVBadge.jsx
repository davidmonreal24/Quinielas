export default function EVBadge({ pick }) {
  const pred = pick?.prediccion ?? ''
  let ev = null
  if (pred === 'Local')      ev = pick?.ev_local
  else if (pred === 'Visitante') ev = pick?.ev_visit
  else if (pred === 'Empate')   ev = pick?.ev_empate

  if (ev == null) return null

  const pct   = (ev * 100).toFixed(1)
  const isPos = ev > 0

  return (
    <span className={`inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-xs font-bold border
      ${isPos
        ? 'bg-emerald-500/20 text-emerald-300 border-emerald-500/30'
        : 'bg-red-500/10 text-red-400 border-red-500/20'
      }`}>
      {isPos ? '⚡' : '▼'}
      {isPos ? `VALUE BET +${pct}%` : `EV ${pct}%`}
    </span>
  )
}
