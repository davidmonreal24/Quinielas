import { PieChart, Pie, Cell, Tooltip, ResponsiveContainer, Legend } from 'recharts'

const COLORS = ['#3b82f6', '#f59e0b', '#ef4444']
const CONF_COLORS = { ALTA: '#10b981', MEDIA: '#f59e0b', BAJA: '#ef4444' }

function CustomTooltip({ active, payload }) {
  if (active && payload?.length) {
    const { name, value } = payload[0]
    return (
      <div className="bg-[#0d1529] border border-white/10 rounded-xl px-3 py-2 text-xs shadow-xl">
        <p className="font-semibold text-white">{name}</p>
        <p className="text-gray-300">{value} picks</p>
      </div>
    )
  }
  return null
}

export default function DistributionChart({ summary }) {
  if (!summary) return null

  const predData = [
    { name: 'Local',      value: summary.local_preds   ?? 0 },
    { name: 'Empate',     value: summary.empate_preds  ?? 0 },
    { name: 'Visitante',  value: summary.visita_preds  ?? 0 },
  ].filter(d => d.value > 0)

  const confData = [
    { name: 'ALTA',  value: summary.alta  ?? 0 },
    { name: 'MEDIA', value: summary.media ?? 0 },
    { name: 'BAJA',  value: summary.baja  ?? 0 },
  ].filter(d => d.value > 0)

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
      {/* Predicción distribution */}
      <div className="bg-[#0d1529] border border-white/8 rounded-2xl p-5">
        <h3 className="text-sm font-semibold text-gray-300 mb-4">Distribución de Predicciones</h3>
        <ResponsiveContainer width="100%" height={180}>
          <PieChart>
            <Pie data={predData} cx="50%" cy="50%" innerRadius={50} outerRadius={80}
              dataKey="value" paddingAngle={3} stroke="none">
              {predData.map((_, i) => (
                <Cell key={i} fill={COLORS[i % COLORS.length]} fillOpacity={0.9} />
              ))}
            </Pie>
            <Tooltip content={<CustomTooltip />} />
            <Legend
              iconType="circle" iconSize={8}
              formatter={(v) => <span className="text-xs text-gray-400">{v}</span>}
            />
          </PieChart>
        </ResponsiveContainer>
      </div>

      {/* Confianza distribution */}
      <div className="bg-[#0d1529] border border-white/8 rounded-2xl p-5">
        <h3 className="text-sm font-semibold text-gray-300 mb-4">Nivel de Confianza</h3>
        <ResponsiveContainer width="100%" height={180}>
          <PieChart>
            <Pie data={confData} cx="50%" cy="50%" innerRadius={50} outerRadius={80}
              dataKey="value" paddingAngle={3} stroke="none">
              {confData.map((entry) => (
                <Cell key={entry.name} fill={CONF_COLORS[entry.name] ?? '#6b7280'} fillOpacity={0.9} />
              ))}
            </Pie>
            <Tooltip content={<CustomTooltip />} />
            <Legend
              iconType="circle" iconSize={8}
              formatter={(v) => <span className="text-xs text-gray-400">{v}</span>}
            />
          </PieChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}
