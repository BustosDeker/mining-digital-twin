// Tipos que reflejan exactamente los esquemas Pydantic del backend
// (backend/api/schemas.py, backend/digital_twin/graph_models.py,
// backend/digital_twin/state.py). Mantener sincronizado manualmente: el
// backend es la única fuente de verdad del contrato.

export type NodeType =
  | "gallery"
  | "intersection"
  | "refuge_chamber"
  | "exit"
  | "risk_zone";

export type EdgeStatus = "clear" | "degraded" | "blocked";

export interface MineNode {
  node_id: string;
  node_type: NodeType;
  position: [number, number, number];
  level: number;
  label: string | null;
  capacity: number;
}

export interface MineEdge {
  edge_id: string;
  source: string;
  target: string;
  length_m: number;
  width_m: number;
  slope_pct: number;
  base_risk: number;
  status: EdgeStatus;
  current_risk: number;
}

export type AgentStatus =
  | "moving"
  | "waiting"
  | "sheltered"
  | "evacuated"
  | "lost";

export interface AgentSnapshot {
  agent_id: number;
  status: AgentStatus;
  node_id: string;
  next_node_id: string;
  edge_id: string | null;
  progress: number;
  panic_level: number;
  cumulative_distance_m: number;
}

export type HazardType = "fire" | "collapse" | "gas_leak";

export interface HazardEvent {
  event_id: string;
  hazard_type: HazardType;
  origin_node_id: string;
  started_at_step: number;
  intensity: number;
  affected_edges: string[];
}

export type SessionStatus =
  | "ready"
  | "running"
  | "paused"
  | "stopped"
  | "finished";

export interface SimulationSnapshot {
  session_id: string;
  status: SessionStatus;
  router_name: string;
  scenario_name: string;
  step: number;
  layout_id: string;
  nodes: MineNode[];
  edges: MineEdge[];
  active_hazards: HazardEvent[];
  agents: Record<string, AgentSnapshot>;
}

export interface SessionSummary {
  session_id: string;
  scenario_name: string;
  router_name: string;
  status: SessionStatus;
  step: number;
}

export interface LayoutConfigInput {
  n_levels: number;
  galleries_per_level: number;
  n_refuge_chambers: number;
  n_exits: number;
  n_risk_zones: number;
  random_seed: number;
}

export interface CreateSessionInput {
  scenario_name: string;
  n_agents?: number;
  router_name: "adaptive_astar" | "q_learning";
  hazard_type: HazardType | null;
  hazard_intensity: number;
  layout_config: LayoutConfigInput;
}

export interface DatasetInfo {
  dataset_name: string;
  is_available_locally: boolean;
  is_synthetic: boolean;
  n_subjects: number;
}

export interface EDAQualityReport {
  dataset_name: string;
  n_subjects: number;
  subjects: string[];
  channels_present: string[];
  valid_class_counts: Record<string, number>;
  class_balance_ratio: Record<string, number>;
  mean_flatline_pct_across_channels: number;
  is_synthetic: boolean;
}

export interface EDASummary {
  dataset_name: string;
  artifacts_dir: string;
  class_distribution: { aggregate_counts: Record<string, number>; n_subjects: number };
  quality_report: EDAQualityReport;
}

export interface ModelMetadata {
  architecture_name: string;
  version_id: string;
  dataset_name: string;
  hyperparams: Record<string, unknown>;
  cv_metrics: Record<string, number>;
  class_names: string[];
  created_at: string;
  has_scaler: boolean;
  extra: Record<string, unknown>;
}

export interface ActiveModelResponse {
  pointer: { architecture_name: string; version_id: string; activated_at: string };
  metadata: ModelMetadata;
}

export interface PredictStressResponse {
  predicted_class: string;
  probabilities: Record<string, number>;
  model_version_id: string;
}

export interface RoutingComparisonResponse {
  scenario_name: string;
  metric_name: string;
  per_router_summary: Record<
    string,
    {
      n_runs: number;
      mean_evacuation_rate: number;
      std_evacuation_rate: number;
      mean_not_evacuated_rate: number;
      mean_evacuation_time_steps: number;
      mean_bottleneck_max_waiting: number;
    }
  >;
  statistical_decision: {
    comparison_type: string;
    statistically_justified_best: string;
    [key: string]: unknown;
  };
}

export interface SessionHistoryRecord {
  session_id: string;
  scenario_name: string;
  router_name: string;
  status: string;
  step: number;
  persisted_at: string;
}
