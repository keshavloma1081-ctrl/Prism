export default function Metric({ label, value, unit, color, size = 'md' }) {
  const sizes = { sm: '20px', md: '28px', lg: '36px' }
  const colors = {
    purple: '#8b5cf6', teal: '#14b8a6',
    amber: '#f59e0b', green: '#10b981',
    red: '#ef4444', blue: '#3b82f6', white: '#f1f5f9'
  }
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
      <div style={{ fontSize: '11px', color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
        {label}
      </div>
      <div style={{ fontSize: sizes[size], fontWeight: 700, color: colors[color] || '#f1f5f9', fontVariantNumeric: 'tabular-nums' }}>
        {value ?? '—'}
        {unit && <span style={{ fontSize: '13px', color: '#64748b', marginLeft: '4px' }}>{unit}</span>}
      </div>
    </div>
  )
}