import axios from 'axios'

const BASE = 'http://localhost:8000'

const api = axios.create({ baseURL: BASE })

export const getSessions = (clientId) =>
  api.get('/sessions/', { params: clientId ? { client_id: clientId } : {} })
    .then(r => r.data)

export const getSession = (id) =>
  api.get(`/sessions/${id}`).then(r => r.data)

export const getVerdict = (id) =>
  api.get(`/sessions/${id}/verdict`).then(r => r.data)

export const getDecay = (id) =>
  api.get(`/sessions/${id}/decay`).then(r => r.data)

export const getAtlas = (id) =>
  api.get(`/sessions/${id}/atlas`).then(r => r.data)

export const runGhost = (id) =>
  api.post(`/sessions/${id}/ghost`).then(r => r.data)

export const getReport = (id) =>
  api.get(`/sessions/${id}/report`).then(r => r.data)

export const getEvents = (id, limit = 20) =>
  api.get(`/sessions/${id}/events`, { params: { limit } }).then(r => r.data)

export const health = () =>
  api.get('/health').then(r => r.data)