export default function ConfidenceBadge({ level }) {
  if (!level) return null

  const map = {
    ALTA:  { cls: 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30', dot: 'bg-emerald-400' },
    MEDIA: { cls: 'bg-amber-500/20 text-amber-400 border-amber-500/30',       dot: 'bg-amber-400'   },
    BAJA:  { cls: 'bg-red-500/20 text-red-400 border-red-500/30',             dot: 'bg-red-400'     },
  }
  const { cls, dot } = map[level] ?? map.BAJA

  return (
    <span className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-semibold border ${cls}`}>
      <span className={`w-1.5 h-1.5 rounded-full ${dot} animate-pulse-slow`} />
      {level}
    </span>
  )
}
