export default function ScoreBar({ label, value, max = 1, color = '#8b5cf6' }) {
  const pct = value != null ? Math.round((value / max) * 100) : null
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <span style={{ fontSize: '12px', color: '#94a3b8' }}>{label}</span>
        <span style={{ fontSize: '12px', fontWeight: 600, color: '#f1f5f9', fontVariantNumeric: 'tabular-nums' }}>
          {value != null ? value.toFixed(3) : '—'}
        </span>
      </div>
      <div style={{ height: '6px', background: '#1f2937', borderRadius: '9999px', overflow: 'hidden' }}>
        <div style={{
          height: '100%',
          width:  `${pct ?? 0}%`,
          background: color,
          borderRadius: '9999px',
          transition: 'width 0.6s ease'
        }} />
      </div>
    </div>
  )
}