// Los mismos tokens que tailwind.config.ts, en formato hexadecimal plano
// para uso directo en materiales de Three.js (que no entienden clases CSS).
export const SCENE_COLORS = {
  clear: "#33FFB2",
  degraded: "#FFB020",
  blocked: "#FF5C5C",
  exit: "#33FFB2",
  refuge: "#5FA8FF",
  riskZone: "#FFB020",
  intersection: "#7C8B88",
  gallery: "#4A5754",
  agentMoving: "#ECEDE9",
  agentWaiting: "#FFB020",
  agentSheltered: "#5FA8FF",
  agentEvacuated: "#33FFB2",
  agentLost: "#FF5C5C",
  agentPanicHalo: "#FF5C5C",
  background: "#0D1210",
  backgroundLight: "#ECEDE9",
} as const;
