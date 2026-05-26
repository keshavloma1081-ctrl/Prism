import { useState } from 'react'
import Overview from './pages/Overview'
import SessionDetail from './pages/SessionDetail'

export default function App() {
  const [selectedSession, setSelectedSession] = useState(null)

  return (
    <div style={{
      minHeight:  '100vh',
      background: '#0a0e1a',
      display:    'flex',
      flexDirection: 'column'
    }}>
      {/* Top nav */}
      <nav style={{
        background:   '#111827',
        borderBottom: '1px solid #1f2937',
        padding:      '0 32px',
        height:       '56px',
        display:      'flex',
        alignItems:   'center',
        gap:          '16px',
        position:     'sticky',
        top:          0,
        zIndex:       100
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <div style={{
            width:        '28px',
            height:       '28px',
            background:   'linear-gradient(135deg, #8b5cf6, #14b8a6)',
            borderRadius: '6px',
            display:      'flex',
            alignItems:   'center',
            justifyContent: 'center',
            fontSize:     '14px',
            fontWeight:   800,
            color:        '#fff'
          }}>P</div>
          <span style={{ fontWeight: 700, fontSize: '16px', color: '#f1f5f9' }}>PRISM</span>
          <span style={{ fontSize: '11px', color: '#64748b', background: '#1f2937', padding: '2px 8px', borderRadius: '9999px' }}>
            Signal Tower
          </span>
        </div>

        <div style={{ marginLeft: 'auto', display: 'flex', gap: '8px', alignItems: 'center' }}>
          {selectedSession && (
            <button
              onClick={() => setSelectedSession(null)}
              style={{ background: 'transparent', border: '1px solid #2d3748', color: '#64748b', padding: '4px 12px', borderRadius: '6px', cursor: 'pointer', fontSize: '12px' }}
            >
              All Sessions
            </button>
          )}
          <div style={{ width: '8px', height: '8px', borderRadius: '50%', background: '#10b981' }} />
          <span style={{ fontSize: '12px', color: '#64748b' }}>API Connected</span>
        </div>
      </nav>

      {/* Main content */}
      <main style={{ flex: 1, padding: '32px', maxWidth: '1400px', margin: '0 auto', width: '100%' }}>
        {selectedSession ? (
          <SessionDetail
            sessionId={selectedSession}
            onBack={() => setSelectedSession(null)}
          />
        ) : (
          <Overview onSelect={setSelectedSession} />
        )}
      </main>
    </div>
  )
}