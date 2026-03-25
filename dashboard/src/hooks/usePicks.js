import { useState, useEffect, useCallback } from 'react'

const API = import.meta.env.VITE_API_URL || ''

export function usePicks() {
  const [picks, setPicks]     = useState([])
  const [summary, setSummary] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError]     = useState(null)

  const fetchAll = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const [picksRes, summaryRes] = await Promise.all([
        fetch(`${API}/api/picks`),
        fetch(`${API}/api/summary`),
      ])

      if (!picksRes.ok || !summaryRes.ok) {
        throw new Error('Error al conectar con la API')
      }

      const picksData   = await picksRes.json()
      const summaryData = await summaryRes.json()

      setPicks(picksData.picks ?? [])
      setSummary(summaryData)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { fetchAll() }, [fetchAll])

  return { picks, summary, loading, error, refetch: fetchAll }
}
