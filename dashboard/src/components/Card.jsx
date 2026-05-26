export default function Card({ title, subtitle, children, accent }) {
  return (
    <div style={{
      background:   '#111827',
      border:       `1px solid ${accent || '#2d3748'}`,
      borderRadius: '12px',
      padding:      '20px',
      display:      'flex',
      flexDirection:'column',
      gap:          '12px'
    }}>
      {title && (
        <div>
          <div style={{ fontWeight: 600, fontSize: '13px', color: '#f1f5f9' }}>
            {title}
          </div>
          {subtitle && (
            <div style={{ fontSize: '11px', color: '#64748b', marginTop: '2px' }}>
              {subtitle}
            </div>
          )}
        </div>
      )}
      {children}
    </div>
  )
}