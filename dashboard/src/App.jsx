import { useState, useMemo } from 'react'
import { usePicks } from './hooks/usePicks.js'
import Header from './components/Header.jsx'
import KPICards from './components/KPICards.jsx'
import FilterBar from './components/FilterBar.jsx'
import PickCard from './components/PickCard.jsx'
import DistributionChart from './components/DistributionChart.jsx'
import EmptyState from './components/EmptyState.jsx'
import Footer from './components/Footer.jsx'

export default function App() {
  const { picks, summary, loading, error, refetch } = usePicks()
  const [filter, setFilter] = useState('all')
  const [search, setSearch] = useState('')
  const [sortBy, setSortBy]  = useState('fecha')

  const filtered = useMemo(() => {
    let list = [...picks]

    // Filter by tab
    if (filter === 'alta')   list = list.filter(p => p.confianza === 'ALTA')
    if (filter === 'value')  list = list.filter(p => p.value_bet === true)
    if (filter === 'empate') list = list.filter(p => p.prediccion === 'Empate')

    // Search
    if (search.trim()) {
      const q = search.toLowerCase()
      list = list.filter(p =>
        (p.local ?? '').toLowerCase().includes(q) ||
        (p.visitante ?? '').toLowerCase().includes(q)
      )
    }

    // Sort
    if (sortBy === 'prob') {
      list.sort((a, b) => {
        const pA = a.prediccion === 'Local' ? a.p_local : a.prediccion === 'Visitante' ? a.p_visit : a.p_empate
        const pB = b.prediccion === 'Local' ? b.p_local : b.prediccion === 'Visitante' ? b.p_visit : b.p_empate
        return (pB ?? 0) - (pA ?? 0)
      })
    } else if (sortBy === 'ev') {
      list.sort((a, b) => {
        const evA = a.prediccion === 'Local' ? a.ev_local : a.prediccion === 'Visitante' ? a.ev_visit : a.ev_empate
        const evB = b.prediccion === 'Local' ? b.ev_local : b.prediccion === 'Visitante' ? b.ev_visit : b.ev_empate
        return (evB ?? -999) - (evA ?? -999)
      })
    }
    // default: mantener orden del CSV (fecha)

    return list
  }, [picks, filter, search, sortBy])

  const showCharts = !loading && !error && picks.length > 0

  return (
    <div className="min-h-screen flex flex-col">
      <Header summary={summary} onRefetch={refetch} />

      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 py-8 space-y-8">

        {/* KPIs */}
        {!loading && !error && summary && (
          <section>
            <KPICards summary={summary} />
          </section>
        )}

        {/* Charts */}
        {showCharts && (
          <section>
            <DistributionChart summary={summary} />
          </section>
        )}

        {/* Picks section */}
        <section className="space-y-5">
          {/* Toolbar */}
          {!loading && !error && picks.length > 0 && (
            <div className="flex flex-col sm:flex-row gap-3 items-start sm:items-center justify-between">
              <FilterBar filter={filter} onFilter={setFilter} summary={summary} />

              <div className="flex items-center gap-2 w-full sm:w-auto">
                {/* Search */}
                <div className="relative flex-1 sm:w-52">
                  <svg className="absolute left-3 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-gray-500" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>
                  </svg>
                  <input
                    type="text"
                    placeholder="Buscar equipo…"
                    value={search}
                    onChange={e => setSearch(e.target.value)}
                    className="w-full pl-8 pr-3 py-1.5 bg-white/5 border border-white/8 rounded-xl text-sm text-gray-200
                      placeholder-gray-600 focus:outline-none focus:border-blue-500/50 focus:bg-white/8 transition-all"
                  />
                </div>

                {/* Sort */}
                <select
                  value={sortBy}
                  onChange={e => setSortBy(e.target.value)}
                  className="bg-white/5 border border-white/8 rounded-xl text-sm text-gray-400 px-3 py-1.5
                    focus:outline-none focus:border-blue-500/50 cursor-pointer transition-all appearance-none"
                >
                  <option value="fecha">Por fecha</option>
                  <option value="prob">Por probabilidad</option>
                  <option value="ev">Por EV</option>
                </select>
              </div>
            </div>
          )}

          {/* Empty / loading / error states */}
          {(loading || error || picks.length === 0) && (
            <EmptyState loading={loading} error={error} onRefetch={refetch} />
          )}

          {/* Results count */}
          {!loading && !error && filtered.length > 0 && filtered.length !== picks.length && (
            <p className="text-xs text-gray-500">{filtered.length} de {picks.length} picks</p>
          )}

          {/* Pick grid */}
          {!loading && !error && filtered.length > 0 && (
            <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
              {filtered.map((pick, i) => (
                <PickCard key={`${pick.local}-${pick.visitante}-${pick.fecha}-${i}`} pick={pick} />
              ))}
            </div>
          )}

          {/* No results after filter */}
          {!loading && !error && picks.length > 0 && filtered.length === 0 && (
            <div className="text-center py-12 text-gray-500 text-sm">
              No hay picks con ese criterio.
              <button className="ml-2 text-blue-400 hover:underline" onClick={() => { setFilter('all'); setSearch('') }}>
                Limpiar filtros
              </button>
            </div>
          )}
        </section>
      </main>

      <Footer />
    </div>
  )
}
