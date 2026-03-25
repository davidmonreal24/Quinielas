export default function ProbabilityBar({ pLocal, pEmpate, pVisita, prediccion }) {
  const local   = pLocal   ?? 0
  const empate  = pEmpate  ?? 0
  const visita  = pVisita  ?? 0

  const isLocal   = prediccion === 'Local'
  const isEmpate  = prediccion === 'Empate'
  const isVisita  = prediccion === 'Visitante'

  return (
    <div className="space-y-1.5">
      {/* Bar */}
      <div className="flex h-2.5 rounded-full overflow-hidden gap-px">
        <div
          className={`transition-all duration-500 rounded-l-full ${isLocal ? 'bg-blue-500' : 'bg-blue-500/40'}`}
          style={{ width: `${local}%` }}
        />
        <div
          className={`transition-all duration-500 ${isEmpate ? 'bg-amber-400' : 'bg-amber-400/40'}`}
          style={{ width: `${empate}%` }}
        />
        <div
          className={`transition-all duration-500 rounded-r-full ${isVisita ? 'bg-rose-500' : 'bg-rose-500/40'}`}
          style={{ width: `${visita}%` }}
        />
      </div>

      {/* Labels */}
      <div className="flex justify-between text-[10px] font-medium">
        <span className={isLocal ? 'text-blue-400' : 'text-gray-500'}>
          L {local.toFixed(0)}%
        </span>
        <span className={isEmpate ? 'text-amber-400' : 'text-gray-500'}>
          E {empate.toFixed(0)}%
        </span>
        <span className={isVisita ? 'text-rose-400' : 'text-gray-500'}>
          V {visita.toFixed(0)}%
        </span>
      </div>
    </div>
  )
}
