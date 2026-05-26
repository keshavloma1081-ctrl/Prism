import { useState, useEffect } from 'react'
import { getSessions, health } from '../api'
import Card from '../components/Card'
import Badge from '../components/Badge'
import Metric from '../components/Metric'

export default function Overview({ onSelect }) {
  const [sessions, setSessions] = useState([])
  const [status,   setStatus]   = useState(null)
  const [loading,  setLoading]  = useState(true)

  useEffect(() => {
    Promise.all([getSessions(), health()])
      .then(([s, h]) => { setSessions(s.sessions || []); setStatus(h) })
      .catch(() => {})
      .finally(() => setLoading(false))
  }, [])

  const gradeColor = (g) => ({
    EXCELLENT: 'green', GOOD: 'teal',
    MODERATE: 'amber', POOR: 'red'
  }[g] || 'gray')

  const statusColor = (s) => ({
    ACTIVE: 'green', COMPLETED: 'teal',
    PAUSED: 'amber', REPLAYING: 'purple'
  }[s] || 'gray')

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>

      {/* Header */}
      <div>
        <h1 style={{ fontSize: '24px', fontWeight: 700, color: '#f1f5f9' }}>
          PRISM Dashboard
        </h1>
        <p style={{ color: '#64748b', marginTop: '4px' }}>
          Epistemic Observability for Human-AI Workflows
        </p>
      </div>

      {/* Status bar */}
      {status && (
        <div style={{
          display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '16px'
        }}>
          {[
            { label: 'API Status',    value: status.status?.toUpperCase(), color: 'green' },
            { label: 'Version',       value: status.version,               color: 'purple' },
            { label: 'Total Sessions',value: status.sessions,              color: 'teal' },
            { label: 'Loaded',        value: sessions.length + ' sessions', color: 'white' },
          ].map(m => (
            <Card key={m.label}>
              <Metric label={m.label} value={m.value} color={m.color} size="sm" />
            </Card>
          ))}
        </div>
      )}

      {/* Sessions table */}
      <Card title="Workflow Sessions" subtitle="Click a session to drill into VERDICT, DECAY, ATLAS, GHOST">
        {loading ? (
          <div style={{ color: '#64748b', padding: '20px 0' }}>Loading sessions...</div>
        ) : sessions.length === 0 ? (
          <div style={{ color: '#64748b', padding: '20px 0' }}>
            No sessions yet. Run <code style={{ color: '#8b5cf6' }}>python demo.py</code> to create one.
          </div>
        ) : (
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead>
              <tr style={{ borderBottom: '1px solid #2d3748' }}>
                {['Session ID', 'Client', 'Workflow', 'Events', 'Status'].map(h => (
                  <th key={h} style={{ textAlign: 'left', padding: '8px 12px', fontSize: '11px', color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {sessions.map(s => (
                <tr
                  key={s.session_id}
                  onClick={() => onSelect(s.session_id)}
                  style={{ borderBottom: '1px solid #1f2937', cursor: 'pointer', transition: 'background 0.15s' }}
                  onMouseEnter={e => e.currentTarget.style.background = '#1f2937'}
                  onMouseLeave={e => e.currentTarget.style.background = 'transparent'}
                >
                  <td style={{ padding: '10px 12px', fontFamily: 'monospace', fontSize: '12px', color: '#8b5cf6' }}>
                    {s.session_id?.slice(0, 20)}...
                  </td>
                  <td style={{ padding: '10px 12px', color: '#f1f5f9' }}>{s.client_id}</td>
                  <td style={{ padding: '10px 12px', color: '#94a3b8' }}>{s.workflow_id}</td>
                  <td style={{ padding: '10px 12px', color: '#f1f5f9', fontVariantNumeric: 'tabular-nums' }}>
                    {s.total_events}
                  </td>
                  <td style={{ padding: '10px 12px' }}>
                    <Badge label={s.status} color={statusColor(s.status)} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </Card>
    </div>
  )
}