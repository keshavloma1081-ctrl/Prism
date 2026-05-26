import { useState, useEffect, useCallback } from 'react'
import {
  RadarChart, Radar, PolarGrid, PolarAngleAxis,
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer,
  BarChart, Bar, Cell
} from 'recharts'
import { getVerdict, getDecay, getAtlas, getEvents, runGhost, getSession } from '../api'
import Card from '../components/Card'
import Badge from '../components/Badge'
import Metric from '../components/Metric'
import ScoreBar from '../components/ScoreBar'

export default function SessionDetail({ sessionId, onBack }) {
  const [session,  setSession]  = useState(null)
  const [verdict,  setVerdict]  = useState(null)
  const [decay,    setDecay]    = useState(null)
  const [atlas,    setAtlas]    = useState(null)
  const [events,   setEvents]   = useState([])
  const [ghost,    setGhost]    = useState(null)
  const [ghosting, setGhosting] = useState(false)
  const [tab,      setTab]      = useState('verdict')
  const [loading,  setLoading]  = useState(true)

  const load = useCallback(() => {
    Promise.all([
      getSession(sessionId),
      getVerdict(sessionId),
      getDecay(sessionId),
      getAtlas(sessionId),
      getEvents(sessionId, 50)
    ]).then(([s, v, d, a, e]) => {
      setSession(s)
      setVerdict(v)
      setDecay(d)
      setAtlas(a)
      setEvents(e.events || [])
    }).catch(console.error)
      .finally(() => setLoading(false))
  }, [sessionId])

  useEffect(() => { load() }, [load])

  // Auto-refresh every 5 seconds
  useEffect(() => {
    const interval = setInterval(load, 5000)
    return () => clearInterval(interval)
  }, [load])

  const handleGhost = async () => {
    setGhosting(true)
    try {
      const result = await runGhost(sessionId)
      setGhost(result)
    } catch (e) {
      console.error(e)
    } finally {
      setGhosting(false)
    }
  }

  const gradeColor = (g) => ({
    EXCELLENT: 'green', GOOD: 'teal',
    MODERATE: 'amber', POOR: 'red', NO_DATA: 'gray'
  }[g] || 'gray')

  const verdictData = verdict ? [
    { subject: 'Groundedness',   value: (verdict.mean_groundedness || 0) * 100 },
    { subject: 'Novelty',        value: (verdict.mean_novelty_delta || 0) * 100 },
    { subject: 'Influence',      value: (verdict.mean_influence_survival || 0) * 100 },
    { subject: 'Calibration',    value: (verdict.composite_score || 0) * 100 },
  ] : []

  const decayData = decay ? [
    { name: 'Engagement',  value: decay.current_engagement_rate || 0 },
    { name: 'Diversity',   value: decay.current_diversity_index || 0 },
    { name: 'Novelty',     value: decay.current_novelty_rate   || 0 },
  ] : []

  const tabs = ['verdict', 'decay', 'atlas', 'ghost', 'events']

  if (loading) return (
    <div style={{ color: '#64748b', padding: '40px' }}>Loading session...</div>
  )

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>

      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
        <button
          onClick={onBack}
          style={{ background: '#1f2937', border: '1px solid #2d3748', color: '#94a3b8', padding: '6px 12px', borderRadius: '8px', cursor: 'pointer' }}
        >
          ← Back
        </button>
        <div>
          <h2 style={{ fontSize: '18px', fontWeight: 700, color: '#f1f5f9' }}>
            {session?.workflow_id}
          </h2>
          <div style={{ fontSize: '12px', color: '#64748b', fontFamily: 'monospace' }}>
            {sessionId}
          </div>
        </div>
        <div style={{ marginLeft: 'auto', display: 'flex', gap: '8px', alignItems: 'center' }}>
          <Badge label={session?.status || 'ACTIVE'} color="teal" />
          <span style={{ fontSize: '12px', color: '#64748b' }}>
            {session?.total_events} events · {session?.human_agents} humans · {session?.ai_agents} AI agents
          </span>
        </div>
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: '4px', borderBottom: '1px solid #2d3748', paddingBottom: '1px' }}>
        {tabs.map(t => (
          <button
            key={t}
            onClick={() => setTab(t)}
            style={{
              background:   tab === t ? '#1f2937' : 'transparent',
              border:       tab === t ? '1px solid #2d3748' : '1px solid transparent',
              borderBottom: tab === t ? '1px solid #0a0e1a' : '1px solid transparent',
              color:        tab === t ? '#f1f5f9' : '#64748b',
              padding:      '8px 16px',
              borderRadius: '8px 8px 0 0',
              cursor:       'pointer',
              fontSize:     '13px',
              fontWeight:   tab === t ? 600 : 400,
              textTransform:'uppercase',
              letterSpacing:'0.05em',
              marginBottom: '-1px'
            }}
          >
            {t}
          </button>
        ))}
      </div>

      {/* VERDICT TAB */}
      {tab === 'verdict' && verdict && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '16px' }}>
            <Card title="Overall Grade" accent="#6d28d9">
              <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                <div style={{ fontSize: '48px', fontWeight: 800, color: '#8b5cf6' }}>
                  {verdict.verdict_grade?.[0] || '?'}
                </div>
                <div>
                  <Badge label={verdict.verdict_grade || 'N/A'} color={gradeColor(verdict.verdict_grade)} />
                  <div style={{ fontSize: '12px', color: '#64748b', marginTop: '4px' }}>
                    Composite: {verdict.composite_score?.toFixed(3) || '—'}
                  </div>
                </div>
              </div>
            </Card>
            <Card title="AI Events">
              <Metric label="Total AI Events" value={verdict.total_ai_events} color="purple" size="lg" />
            </Card>
            <Card title="Composite Score">
              <Metric label="Score" value={verdict.composite_score?.toFixed(3)} color="teal" size="lg" />
            </Card>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
            <Card title="Four Dimensions" subtitle="AI epistemic quality scores">
              <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                <ScoreBar label="Groundedness"       value={verdict.mean_groundedness}        color="#8b5cf6" />
                <ScoreBar label="Novelty Delta"      value={verdict.mean_novelty_delta}       color="#14b8a6" />
                <ScoreBar label="Influence Survival" value={verdict.mean_influence_survival}  color="#f59e0b" />
                <ScoreBar label="Composite"          value={verdict.composite_score}          color="#10b981" />
              </div>
            </Card>

            <Card title="Radar View" subtitle="Epistemic quality profile">
              <ResponsiveContainer width="100%" height={200}>
                <RadarChart data={verdictData}>
                  <PolarGrid stroke="#2d3748" />
                  <PolarAngleAxis dataKey="subject" tick={{ fill: '#64748b', fontSize: 11 }} />
                  <Radar dataKey="value" stroke="#8b5cf6" fill="#8b5cf6" fillOpacity={0.3} />
                </RadarChart>
              </ResponsiveContainer>
            </Card>
          </div>
        </div>
      )}

      {/* DECAY TAB */}
      {tab === 'decay' && decay && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '16px' }}>
            {[
              { label: 'Total Alerts',    value: decay.total_alerts,              color: decay.total_alerts > 0 ? 'red' : 'green' },
              { label: 'Engagement Rate', value: decay.current_engagement_rate?.toFixed(2), color: 'teal' },
              { label: 'Diversity Index', value: decay.current_diversity_index?.toFixed(3), color: decay.current_diversity_index < 0.2 ? 'red' : 'green' },
              { label: 'Novelty Rate',    value: decay.current_novelty_rate?.toFixed(2),    color: 'purple' },
            ].map(m => (
              <Card key={m.label}>
                <Metric label={m.label} value={m.value} color={m.color} />
              </Card>
            ))}
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
            <Card title="Signal Health">
              <ResponsiveContainer width="100%" height={180}>
                <BarChart data={decayData}>
                  <XAxis dataKey="name" tick={{ fill: '#64748b', fontSize: 11 }} />
                  <YAxis domain={[0, 1]} tick={{ fill: '#64748b', fontSize: 11 }} />
                  <Tooltip
                    contentStyle={{ background: '#1f2937', border: '1px solid #2d3748', borderRadius: '8px' }}
                    labelStyle={{ color: '#f1f5f9' }}
                  />
                  <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                    {decayData.map((entry, i) => (
                      <Cell key={i} fill={entry.value < 0.2 ? '#ef4444' : entry.value < 0.5 ? '#f59e0b' : '#10b981'} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </Card>

            <Card title="Alerts by Type">
              <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                {Object.entries(decay.alerts_by_type || {}).map(([type, count]) => (
                  <div key={type} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '6px 0', borderBottom: '1px solid #1f2937' }}>
                    <span style={{ fontSize: '12px', color: '#94a3b8' }}>{type}</span>
                    <Badge label={String(count)} color={count > 0 ? 'red' : 'gray'} />
                  </div>
                ))}
              </div>
            </Card>
          </div>

          {decay.critical_alerts?.length > 0 && (
            <Card title="Critical Alerts" accent="#ef4444">
              <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                {decay.critical_alerts.map((alert, i) => (
                  <div key={i} style={{ background: '#1f2937', borderRadius: '8px', padding: '12px', borderLeft: '3px solid #ef4444' }}>
                    <div style={{ display: 'flex', gap: '8px', marginBottom: '6px' }}>
                      <Badge label={alert.severity} color="red" />
                      <Badge label={alert.type} color="amber" />
                    </div>
                    <div style={{ fontSize: '12px', color: '#94a3b8', lineHeight: 1.6 }}>
                      {alert.recommendation}
                    </div>
                  </div>
                ))}
              </div>
            </Card>
          )}
        </div>
      )}

      {/* ATLAS TAB */}
      {tab === 'atlas' && atlas && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '16px' }}>
            {[
              { label: 'Nodes',          value: atlas.node_count,              color: 'purple' },
              { label: 'Edges',          value: atlas.edge_count,              color: 'teal' },
              { label: 'Coupling Index', value: atlas.coupling_index?.toFixed(3), color: 'amber' },
              { label: 'Discoveries',    value: atlas.discoveries?.length,    color: 'green' },
            ].map(m => (
              <Card key={m.label}>
                <Metric label={m.label} value={m.value} color={m.color} />
              </Card>
            ))}
          </div>

          <Card title="Causal Fingerprint Nodes" subtitle="Every epistemic event as a node">
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
              {(atlas.nodes || []).map(node => (
                <div key={node.id} style={{
                  background:   node.agent_type === 'HUMAN' ? '#1e1b4b' : '#134e4a',
                  border:       `1px solid ${node.agent_type === 'HUMAN' ? '#4c1d95' : '#134e4a'}`,
                  borderRadius: '8px',
                  padding:      '8px 12px',
                  fontSize:     '11px'
                }}>
                  <div style={{ color: node.agent_type === 'HUMAN' ? '#a78bfa' : '#5eead4', fontWeight: 600 }}>
                    {node.label}
                  </div>
                  <div style={{ color: '#64748b', marginTop: '2px' }}>
                    mag: {node.magnitude?.toFixed(2)} · t={node.t}
                  </div>
                </div>
              ))}
            </div>
          </Card>

          {atlas.discoveries?.length > 0 && (
            <Card title="Discoveries Traced" accent="#10b981">
              {atlas.discoveries.map((d, i) => (
                <div key={i} style={{ background: '#0d1f17', borderRadius: '8px', padding: '12px', borderLeft: '3px solid #10b981' }}>
                  <div style={{ display: 'flex', gap: '8px', marginBottom: '6px' }}>
                    <Badge label={`Discovery ${i + 1}`} color="green" />
                    <span style={{ fontSize: '11px', color: '#64748b' }}>novelty: {d.novelty_score?.toFixed(3)}</span>
                    <span style={{ fontSize: '11px', color: '#64748b' }}>chain: {d.causal_chain?.length} events</span>
                  </div>
                  <div style={{ fontSize: '12px', color: '#94a3b8' }}>{d.description}</div>
                </div>
              ))}
            </Card>
          )}
        </div>
      )}

      {/* GHOST TAB */}
      {tab === 'ghost' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
          {!ghost ? (
            <Card title="Ghost Runner — Counterfactual Analysis" subtitle="Isolates what humans and AI each contributed">
              <p style={{ color: '#64748b', fontSize: '13px', lineHeight: 1.6 }}>
                Ghost Runner replays this session three ways: full ensemble, humans only, AI only.
                The difference reveals what each side actually contributed.
              </p>
              <button
                onClick={handleGhost}
                disabled={ghosting}
                style={{
                  background:   ghosting ? '#1f2937' : '#6d28d9',
                  color:        '#f1f5f9',
                  border:       'none',
                  padding:      '10px 20px',
                  borderRadius: '8px',
                  cursor:       ghosting ? 'not-allowed' : 'pointer',
                  fontWeight:   600,
                  fontSize:     '13px',
                  width:        'fit-content'
                }}
              >
                {ghosting ? 'Running counterfactuals...' : '▶ Run Ghost Runner'}
              </button>
            </Card>
          ) : (
            <>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '16px' }}>
                {[
                  { label: 'Emergence Score',  value: ghost.emergence_score?.toFixed(3),   color: ghost.emergence_score > 0.5 ? 'green' : 'amber' },
                  { label: 'AI Value Score',   value: ghost.ai_value_score?.toFixed(3),    color: 'purple' },
                  { label: 'Human Value Score',value: ghost.human_value_score?.toFixed(3), color: 'teal' },
                ].map(m => (
                  <Card key={m.label}>
                    <Metric label={m.label} value={m.value} color={m.color} size="lg" />
                  </Card>
                ))}
              </div>

              <Card title="Verdict" accent="#6d28d9">
                <p style={{ color: '#94a3b8', fontSize: '13px', lineHeight: 1.7 }}>
                  {ghost.verdict}
                </p>
              </Card>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
                <Card title="AI Unique Concepts" subtitle="Concepts only AI contributed">
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
                    {(ghost.ai_unique || []).length === 0
                      ? <span style={{ color: '#64748b', fontSize: '12px' }}>None</span>
                      : (ghost.ai_unique || []).map(c => (
                          <Badge key={c} label={c} color="purple" />
                        ))
                    }
                  </div>
                </Card>
                <Card title="Human Unique Concepts" subtitle="Concepts only humans contributed">
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
                    {(ghost.human_unique || []).length === 0
                      ? <span style={{ color: '#64748b', fontSize: '12px' }}>None</span>
                      : (ghost.human_unique || []).map(c => (
                          <Badge key={c} label={c} color="teal" />
                        ))
                    }
                  </div>
                </Card>
              </div>

              <Card title="Recommendation" accent="#14b8a6">
                <p style={{ color: '#94a3b8', fontSize: '13px', lineHeight: 1.7 }}>
                  {ghost.recommendation}
                </p>
              </Card>

              <button
                onClick={() => setGhost(null)}
                style={{ background: '#1f2937', border: '1px solid #2d3748', color: '#64748b', padding: '8px 16px', borderRadius: '8px', cursor: 'pointer', width: 'fit-content' }}
              >
                Reset
              </button>
            </>
          )}
        </div>
      )}

      {/* EVENTS TAB */}
      {tab === 'events' && (
        <Card title="EAT Event Stream" subtitle="Last 50 epistemic events">
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead>
              <tr style={{ borderBottom: '1px solid #2d3748' }}>
                {['T', 'Agent', 'Type', 'Event', 'Magnitude', 'Timestamp'].map(h => (
                  <th key={h} style={{ textAlign: 'left', padding: '8px 12px', fontSize: '11px', color: '#64748b', textTransform: 'uppercase' }}>
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {events.map(e => (
                <tr key={e.event_id} style={{ borderBottom: '1px solid #1f2937' }}>
                  <td style={{ padding: '8px 12px', color: '#8b5cf6', fontVariantNumeric: 'tabular-nums' }}>{e.t}</td>
                  <td style={{ padding: '8px 12px', fontFamily: 'monospace', fontSize: '11px', color: '#94a3b8' }}>
                    {e.agent_id?.slice(0, 12)}
                  </td>
                  <td style={{ padding: '8px 12px' }}>
                    <Badge label={e.agent_type} color={e.agent_type === 'HUMAN' ? 'purple' : 'teal'} />
                  </td>
                  <td style={{ padding: '8px 12px' }}>
                    <Badge label={e.event_type} color="gray" />
                  </td>
                  <td style={{ padding: '8px 12px', fontVariantNumeric: 'tabular-nums', color: '#f1f5f9' }}>
                    {e.delta_magnitude?.toFixed(3)}
                  </td>
                  <td style={{ padding: '8px 12px', fontSize: '11px', color: '#64748b' }}>
                    {new Date(e.timestamp).toLocaleTimeString()}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>
      )}
    </div>
  )
}