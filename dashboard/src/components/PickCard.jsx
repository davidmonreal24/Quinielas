import { useState } from 'react'
import ConfidenceBadge from './ConfidenceBadge.jsx'
import EVBadge from './EVBadge.jsx'
import ProbabilityBar from './ProbabilityBar.jsx'
import FormDots from './FormDots.jsx'
import H2HBar from './H2HBar.jsx'

function StatRow({ label, value, highlight }) {
  return (
    <div className="flex justify-between items-center py-1 border-b border-white/5 last:border-0">
      <span className="text-xs text-gray-500">{label}</span>
      <span className={`text-xs font-medium ${highlight ? 'text-white' : 'text-gray-300'}`}>{value ?? '—'}</span>
    </div>
  )
}

function OddsChip({ label, momio, ev }) {
  const isPos = ev != null && ev > 0
  return (
    <div className={`flex-1 rounded-xl p-3 text-center border ${
      isPos ? 'bg-emerald-500/10 border-emerald-500/30' : 'bg-white/5 border-white/10'
    }`}>
      <div className="text-[10px] text-gray-400 mb-1">{label}</div>
      <div className={`text-lg font-bold ${isPos ? 'text-emerald-300' : 'text-gray-200'}`}>
        {momio != null ? momio.toFixed(2) : '—'}
      </div>
      {ev != null && (
        <div className={`text-[10px] font-semibold mt-0.5 ${isPos ? 'text-emerald-400' : 'text-gray-500'}`}>
          EV {ev > 0 ? '+' : ''}{(ev * 100).toFixed(1)}%
        </div>
      )}
    </div>
  )
}

export default function PickCard({ pick }) {
  const [expanded, setExpanded] = useState(false)

  const {
    local, visitante, fecha, fase,
    prediccion, confianza,
    p_local, p_empate, p_visit,
    lambda_h, lambda_a,
    forma_h, forma_a,
    pts_forma_h, pts_forma_a,
    pos_h, pos_v,
    pts_tabla_h, pts_tabla_v,
    pj_h, pj_v,
    h2h_n, h2h_w_h, h2h_d, h2h_w_a,
    h2h_gf_h, h2h_gf_a,
    momio_local, momio_empate, momio_visit,
    ev_local, ev_empate, ev_visit,
    value_bet,
    att_h, def_h, att_a, def_a,
    n_bookmakers, casas,
  } = pick

  const isValueBet = value_bet === true
  const hasOdds    = momio_local != null

  // Predicted outcome label
  const predLabel =
    prediccion === 'Local'      ? local :
    prediccion === 'Visitante'  ? visitante :
    prediccion === 'Empate'     ? 'Empate' : prediccion

  // Predicted prob
  const predProb =
    prediccion === 'Local'     ? p_local :
    prediccion === 'Visitante' ? p_visit :
    prediccion === 'Empate'    ? p_empate : null

  return (
    <div className={`rounded-2xl border overflow-hidden transition-all duration-200 card-glow
      ${isValueBet
        ? 'border-emerald-500/25 bg-gradient-to-br from-[#0d1a12] to-[#0d1529]'
        : 'border-white/8 bg-[#0d1529]'
      }`}>

      {/* Header */}
      <div className="px-5 pt-5 pb-4">

        {/* Top row: date + badges */}
        <div className="flex items-center justify-between mb-3 flex-wrap gap-2">
          <div className="flex items-center gap-2">
            {fecha && (
              <span className="text-xs text-gray-500 bg-white/5 px-2 py-0.5 rounded-full">{fecha}</span>
            )}
            {fase && (
              <span className="text-xs text-gray-500">{fase}</span>
            )}
          </div>
          <div className="flex items-center gap-2">
            <ConfidenceBadge level={confianza} />
            {isValueBet && <EVBadge pick={pick} />}
          </div>
        </div>

        {/* Teams */}
        <div className="flex items-center gap-3 mb-4">
          <div className="flex-1 min-w-0">
            <div className={`font-bold text-base truncate ${prediccion === 'Local' ? 'text-white' : 'text-gray-300'}`}>
              {local}
            </div>
            <div className="text-xs text-gray-500 mt-0.5">
              {pos_h != null && `#${pos_h}`} {pts_tabla_h != null && `· ${pts_tabla_h} pts`}
            </div>
          </div>

          {/* Center: score prediction */}
          <div className="shrink-0 text-center bg-white/5 rounded-xl px-3 py-2 border border-white/8">
            <div className="text-xs text-gray-500 mb-0.5">λ esperado</div>
            <div className="font-bold text-white text-sm tabular-nums">
              {lambda_h != null ? lambda_h.toFixed(2) : '?'}
              <span className="text-gray-500 mx-1">:</span>
              {lambda_a != null ? lambda_a.toFixed(2) : '?'}
            </div>
          </div>

          <div className="flex-1 min-w-0 text-right">
            <div className={`font-bold text-base truncate ${prediccion === 'Visitante' ? 'text-white' : 'text-gray-300'}`}>
              {visitante}
            </div>
            <div className="text-xs text-gray-500 mt-0.5">
              {pos_v != null && `#${pos_v}`} {pts_tabla_v != null && `· ${pts_tabla_v} pts`}
            </div>
          </div>
        </div>

        {/* Probability bar */}
        <ProbabilityBar
          pLocal={p_local} pEmpate={p_empate} pVisita={p_visit}
          prediccion={prediccion}
        />

        {/* Prediction result */}
        <div className="mt-3 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="text-xs text-gray-500">Predicción:</span>
            <span className="text-sm font-bold text-white">{predLabel}</span>
            {predProb != null && (
              <span className="text-xs text-gray-400">({predProb.toFixed(0)}%)</span>
            )}
          </div>

          {/* Expand toggle */}
          <button
            onClick={() => setExpanded(!expanded)}
            className="text-xs text-gray-500 hover:text-gray-300 transition-colors flex items-center gap-1"
          >
            {expanded ? 'Menos' : 'Más info'}
            <svg className={`w-3 h-3 transition-transform ${expanded ? 'rotate-180' : ''}`} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
              <polyline points="6 9 12 15 18 9"/>
            </svg>
          </button>
        </div>
      </div>

      {/* Odds row (always visible if available) */}
      {hasOdds && (
        <div className="px-5 pb-4">
          <div className="flex gap-2">
            <OddsChip label="Local" momio={momio_local} ev={ev_local} />
            <OddsChip label="Empate" momio={momio_empate} ev={ev_empate} />
            <OddsChip label="Visitante" momio={momio_visit} ev={ev_visit} />
          </div>
          {n_bookmakers && (
            <p className="text-[10px] text-gray-600 mt-1.5 text-right">
              {n_bookmakers} casas · {casas ?? ''}
            </p>
          )}
        </div>
      )}

      {/* Expanded section */}
      {expanded && (
        <div className="border-t border-white/8 px-5 py-4 space-y-5 bg-white/3">

          {/* Form */}
          <div>
            <h4 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2.5">Forma reciente</h4>
            <div className="grid grid-cols-2 gap-3">
              <div className="space-y-1.5">
                <div className="text-xs font-medium text-gray-300 truncate">{local}</div>
                <FormDots forma={forma_h} />
                {pts_forma_h != null && (
                  <div className="text-xs text-gray-500">{pts_forma_h} pts (últimos 5)</div>
                )}
              </div>
              <div className="space-y-1.5 text-right">
                <div className="text-xs font-medium text-gray-300 truncate">{visitante}</div>
                <div className="flex justify-end">
                  <FormDots forma={forma_a} />
                </div>
                {pts_forma_a != null && (
                  <div className="text-xs text-gray-500">{pts_forma_a} pts (últimos 5)</div>
                )}
              </div>
            </div>
          </div>

          {/* H2H */}
          {h2h_n != null && h2h_n > 0 && (
            <div>
              <h4 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2.5">Historial H2H</h4>
              <H2HBar
                wLocal={h2h_w_h} draws={h2h_d} wVisita={h2h_w_a}
                total={h2h_n} local={local} visitante={visitante}
              />
              {(h2h_gf_h != null || h2h_gf_a != null) && (
                <p className="text-xs text-gray-500 mt-1.5">
                  Goles H2H: {local} {h2h_gf_h ?? 0} – {h2h_gf_a ?? 0} {visitante}
                </p>
              )}
            </div>
          )}

          {/* Model ratings */}
          <div>
            <h4 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">Ratings del modelo</h4>
            <div className="grid grid-cols-2 gap-x-6">
              <div>
                <StatRow label={`Ataque ${local}`}  value={att_h?.toFixed(3)} highlight={att_h > 1} />
                <StatRow label={`Defensa ${local}`}  value={def_h?.toFixed(3)} highlight={def_h < 1} />
                <StatRow label="PJ local"            value={pj_h} />
              </div>
              <div>
                <StatRow label={`Ataque ${visitante}`} value={att_a?.toFixed(3)} highlight={att_a > 1} />
                <StatRow label={`Defensa ${visitante}`} value={def_a?.toFixed(3)} highlight={def_a < 1} />
                <StatRow label="PJ visitante"          value={pj_v} />
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
