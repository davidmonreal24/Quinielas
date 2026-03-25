export default function EmptyState({ loading, error, onRefetch }) {
  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center py-24 gap-4">
        <div className="w-10 h-10 rounded-full border-2 border-blue-500/30 border-t-blue-500 animate-spin" />
        <p className="text-gray-400 text-sm">Cargando predicciones…</p>
      </div>
    )
  }

  if (error) {
    return (
      <div className="flex flex-col items-center justify-center py-24 gap-4 text-center">
        <div className="w-14 h-14 rounded-2xl bg-red-500/10 border border-red-500/20 flex items-center justify-center">
          <svg className="w-7 h-7 text-red-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/>
          </svg>
        </div>
        <div>
          <p className="text-white font-semibold">Error al cargar datos</p>
          <p className="text-gray-400 text-sm mt-1">{error}</p>
          <p className="text-gray-500 text-xs mt-2">
            Asegúrate de que la API esté corriendo en <code className="text-blue-400">localhost:8000</code>
          </p>
        </div>
        <button
          onClick={onRefetch}
          className="px-4 py-2 bg-blue-600 hover:bg-blue-500 text-white text-sm rounded-xl font-medium transition-colors"
        >
          Reintentar
        </button>
      </div>
    )
  }

  return (
    <div className="flex flex-col items-center justify-center py-24 gap-4 text-center">
      <div className="w-14 h-14 rounded-2xl bg-white/5 border border-white/10 flex items-center justify-center text-3xl">
        ⚽
      </div>
      <div>
        <p className="text-white font-semibold">Sin predicciones disponibles</p>
        <p className="text-gray-400 text-sm mt-1">
          Ejecuta <code className="text-blue-400">python predict_ligamx.py</code> para generar picks.
        </p>
      </div>
    </div>
  )
}
