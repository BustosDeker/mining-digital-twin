// Cliente REST tipado hacia el backend FastAPI. El frontend consume
// EXCLUSIVAMENTE esta API — ninguna lógica de negocio se duplica aquí
// (regla fija del proyecto).

import type {
  ActiveModelResponse,
  CreateSessionInput,
  DatasetInfo,
  EDASummary,
  ModelMetadata,
  PredictStressResponse,
  RoutingComparisonResponse,
  SessionHistoryRecord,
  SessionSummary,
  SimulationSnapshot,
} from "./types";

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000";

class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
    this.name = "ApiError";
  }
}

async function request<T>(
  path: string,
  options: RequestInit = {}
): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...options.headers,
    },
    cache: "no-store",
  });

  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = await response.json();
      detail = body.detail || detail;
    } catch {
      // el cuerpo no era JSON; se mantiene el statusText
    }
    throw new ApiError(response.status, detail);
  }

  if (response.status === 204) {
    return undefined as T;
  }
  return (await response.json()) as T;
}

export const api = {
  // -------------------- Simulación --------------------
  createSession: (payload: CreateSessionInput) =>
    request<SessionSummary>("/api/simulations", {
      method: "POST",
      body: JSON.stringify(payload),
    }),

  listSessions: () => request<SessionSummary[]>("/api/simulations"),

  getSessionState: (sessionId: string) =>
    request<SimulationSnapshot>(`/api/simulations/${sessionId}`),

  startSession: (sessionId: string) =>
    request<SessionSummary>(`/api/simulations/${sessionId}/start`, {
      method: "POST",
    }),

  pauseSession: (sessionId: string) =>
    request<SessionSummary>(`/api/simulations/${sessionId}/pause`, {
      method: "POST",
    }),

  stopSession: (sessionId: string) =>
    request<SessionSummary>(`/api/simulations/${sessionId}/stop`, {
      method: "POST",
    }),

  resetSession: (sessionId: string) =>
    request<SessionSummary>(`/api/simulations/${sessionId}/reset`, {
      method: "POST",
    }),

  stepSession: (sessionId: string) =>
    request<SessionSummary>(`/api/simulations/${sessionId}/step`, {
      method: "POST",
    }),

  // -------------------- Evaluación / Monte Carlo --------------------
  compareRoutingStrategies: (payload: Record<string, unknown>) =>
    request<RoutingComparisonResponse>("/api/evaluation/routing-comparison", {
      method: "POST",
      body: JSON.stringify(payload),
    }),

  // -------------------- Motor IA --------------------
  listDatasets: () => request<DatasetInfo[]>("/api/ml/datasets"),

  runEDA: (datasetName: string, subjectLimit?: number) =>
    request<EDASummary>(`/api/ml/datasets/${datasetName}/eda`, {
      method: "POST",
      body: JSON.stringify({ subject_limit: subjectLimit ?? null }),
    }),

  listEDAArtifacts: (datasetName: string) =>
    request<string[]>(`/api/ml/datasets/${datasetName}/eda/artifacts`),

  edaArtifactUrl: (datasetName: string, filename: string) =>
    `${API_BASE_URL}/api/ml/datasets/${datasetName}/eda/artifacts/${filename}`,

  listModels: (architectureName?: string) =>
    request<ModelMetadata[]>(
      `/api/ml/models${architectureName ? `?architecture_name=${architectureName}` : ""}`
    ),

  getActiveModel: () => request<ActiveModelResponse>("/api/ml/models/active"),

  activateModel: (architectureName: string, versionId: string) =>
    request(`/api/ml/models/${architectureName}/${versionId}/activate`, {
      method: "POST",
    }),

  predictStress: (features: Record<string, number>) =>
    request<PredictStressResponse>("/api/ml/predict", {
      method: "POST",
      body: JSON.stringify({ features }),
    }),

  listReports: () => request<string[]>("/api/ml/reports"),

  reportDownloadUrl: (filename: string) =>
    `${API_BASE_URL}/api/ml/reports/${filename}`,

  // -------------------- Historial de sesiones --------------------
  listSessionHistory: () =>
    request<SessionHistoryRecord[]>("/api/sessions"),

  getSessionRecord: (sessionId: string) =>
    request<Record<string, unknown>>(`/api/sessions/${sessionId}`),
};

export { ApiError, API_BASE_URL };
