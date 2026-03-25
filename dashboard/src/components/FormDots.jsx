/**
 * Renders W/D/L form string as colored dots.
 * e.g. "WWD" → 🟢🟢🟡
 */
export default function FormDots({ forma, label }) {
  if (!forma) return null

  const chars = forma.replace(/[-/\s]/g, '').toUpperCase().split('')

  const dotMap = {
    W: 'bg-emerald-400 shadow-emerald-400/50',
    D: 'bg-amber-400 shadow-amber-400/50',
    L: 'bg-red-500 shadow-red-500/50',
  }

  return (
    <div className="flex items-center gap-1.5">
      {label && <span className="text-xs text-gray-500 mr-0.5">{label}</span>}
      {chars.map((c, i) => (
        <span
          key={i}
          className={`w-2 h-2 rounded-full shadow-sm ${dotMap[c] ?? 'bg-gray-600'}`}
          title={c === 'W' ? 'Victoria' : c === 'D' ? 'Empate' : 'Derrota'}
        />
      ))}
    </div>
  )
}
