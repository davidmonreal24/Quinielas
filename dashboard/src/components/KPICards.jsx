function KPICard({ label, value, sub, accent, icon }) {
  const accentMap = {
    blue:   'from-blue-500/20 to-blue-600/5 border-blue-500/20 text-blue-400',
    green:  'from-emerald-500/20 to-emerald-600/5 border-emerald-500/20 text-emerald-400',
    yellow: 'from-amber-500/20 to-amber-600/5 border-amber-500/20 text-amber-400',
    purple: 'from-purple-500/20 to-purple-600/5 border-purple-500/20 text-purple-400',
    red:    'from-red-500/20 to-red-600/5 border-red-500/20 text-red-400',
  }
  const cls = accentMap[accent] ?? accentMap.blue

  return (
    <div className={`relative rounded-2xl border bg-gradient-to-br p-5 ${cls} overflow-hidden`}>
      <div className="flex items-start justify-between mb-3">
        <span className="text-xs font-medium text-gray-400 uppercase tracking-wider">{label}</span>
        <span className="text-lg opacity-60">{icon}</span>
      </div>
      <div className="text-3xl font-extrabold text-white tabular-nums">{value ?? '—'}</div>
      {sub && <div className="mt-1 text-xs text-gray-400">{sub}</div>}
    </div>
  )
}

export default function KPICards({ summary }) {
  if (!summary) return null

  const {
    total_picks, alta, media, baja,
    value_bets, avg_ev, avg_pred_prob,
    local_preds, empate_preds, visita_preds,
  } = summary

  const evStr = avg_ev != null ? `${(avg_ev * 100).toFixed(1)}%` : '—'
  const probStr = avg_pred_prob != null ? `${avg_pred_prob.toFixed(1)}%` : '—'

  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
      <KPICard
        label="Total picks"
        value={total_picks}
        sub={`L: ${local_preds}  E: ${empate_preds}  V: ${visita_preds}`}
        accent="blue"
        icon="⚽"
      />
      <KPICard
        label="Confianza ALTA"
        value={alta}
        sub={`Media: ${media} · Baja: ${baja}`}
        accent="green"
        icon="🎯"
      />
      <KPICard
        label="Value Bets"
        value={value_bets}
        sub={value_bets > 0 ? `EV promedio ${evStr}` : 'Sin EV positivo'}
        accent={value_bets > 0 ? 'yellow' : 'red'}
        icon="💰"
      />
      <KPICard
        label="Prob. predicha avg"
        value={probStr}
        sub="Probabilidad del resultado predicho"
        accent="purple"
        icon="📊"
      />
    </div>
  )
}
