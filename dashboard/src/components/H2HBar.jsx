/**
 * Mini H2H visualization: W-D-L bar for head-to-head record.
 */
export default function H2HBar({ wLocal, draws, wVisita, total, local, visitante }) {
  if (!total || total === 0) return <p className="text-xs text-gray-500 italic">Sin historial H2H</p>

  const pW = (wLocal  / total) * 100
  const pD = (draws   / total) * 100
  const pL = (wVisita / total) * 100

  return (
    <div className="space-y-2">
      <div className="flex justify-between text-xs text-gray-400 font-medium">
        <span className="truncate max-w-[40%]">{local}</span>
        <span className="text-gray-600">{total} partidos</span>
        <span className="truncate max-w-[40%] text-right">{visitante}</span>
      </div>

      <div className="flex h-2 rounded-full overflow-hidden">
        <div className="bg-blue-500"   style={{ width: `${pW}%` }} title={`Local: ${wLocal}`} />
        <div className="bg-gray-500"   style={{ width: `${pD}%` }} title={`Empates: ${draws}`} />
        <div className="bg-rose-500"   style={{ width: `${pL}%` }} title={`Visitante: ${wVisita}`} />
      </div>

      <div className="flex justify-between text-xs font-semibold">
        <span className="text-blue-400">{wLocal}V</span>
        <span className="text-gray-400">{draws}E</span>
        <span className="text-rose-400">{wVisita}V</span>
      </div>
    </div>
  )
}
