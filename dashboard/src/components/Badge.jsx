export default function Badge({ label, color = 'purple' }) {
  const colors = {
    purple:  { bg: '#4c1d95', text: '#c4b5fd' },
    green:   { bg: '#064e3b', text: '#6ee7b7' },
    amber:   { bg: '#78350f', text: '#fcd34d' },
    red:     { bg: '#7f1d1d', text: '#fca5a5' },
    teal:    { bg: '#134e4a', text: '#5eead4' },
    gray:    { bg: '#1f2937', text: '#9ca3af' },
  }
  const c = colors[color] || colors.gray
  return (
    <span style={{
      background:   c.bg,
      color:        c.text,
      padding:      '2px 8px',
      borderRadius: '9999px',
      fontSize:     '11px',
      fontWeight:   600,
      letterSpacing: '0.05em',
      textTransform: 'uppercase'
    }}>
      {label}
    </span>
  )
}