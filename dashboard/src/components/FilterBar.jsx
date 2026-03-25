export default function FilterBar({ filter, onFilter, summary }) {
  const tabs = [
    { key: 'all',    label: 'Todos',      count: summary?.total_picks },
    { key: 'alta',   label: 'Confianza ALTA', count: summary?.alta },
    { key: 'value',  label: 'Value Bets', count: summary?.value_bets },
    { key: 'empate', label: 'Empates',    count: summary?.empate_preds },
  ]

  return (
    <div className="flex items-center gap-1.5 flex-wrap">
      {tabs.map(({ key, label, count }) => (
        <button
          key={key}
          onClick={() => onFilter(key)}
          className={`px-3.5 py-1.5 rounded-xl text-sm font-medium transition-all duration-150
            ${filter === key
              ? 'bg-blue-500 text-white shadow-lg shadow-blue-500/20'
              : 'bg-white/5 text-gray-400 hover:bg-white/10 hover:text-gray-200 border border-white/8'
            }`}
        >
          {label}
          {count != null && (
            <span className={`ml-1.5 text-xs px-1.5 py-0.5 rounded-full ${
              filter === key ? 'bg-white/20 text-white' : 'bg-white/10 text-gray-400'
            }`}>
              {count}
            </span>
          )}
        </button>
      ))}
    </div>
  )
}
