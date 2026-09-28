"use client";

import { useMemo, useLayoutEffect, useEffect, useRef } from "react";
import * as THREE from "three";
import type { AgentSnapshot, AgentStatus, MineNode } from "@/lib/types";
import { backendToThreePosition, lerpPosition } from "./geometry";
import { SCENE_COLORS as C } from "./colors";

interface AgentsProps {
  nodes: MineNode[]; agents: Record<string, AgentSnapshot>; followedAgentId: string | null;
  onSelectAgent: (id: string) => void; performanceMode: boolean;
}
const STATUS: Record<AgentStatus, string> = { moving: C.agentMoving, waiting: C.agentWaiting,
  sheltered: C.agentSheltered, evacuated: C.agentEvacuated, lost: C.agentLost };
interface Worker { agent: AgentSnapshot; position: [number, number, number]; heading: number; followed: boolean }
interface Part {
  shape: "box" | "head" | "helmet" | "ring";
  position: [number, number, number]; scale: [number, number, number]; color: string;
  basic?: boolean; indicator?: "status" | "panic" | "follow";
}
const PARTS: Part[] = [
  { shape: "box", position: [0, 0.145, 0], scale: [0.09, 0.12, 0.055], color: "#d7a44e" },
  { shape: "box", position: [-0.025, 0.044, 0], scale: [0.031, 0.088, 0.041], color: "#283c4b" },
  { shape: "box", position: [0.025, 0.044, 0], scale: [0.031, 0.088, 0.041], color: "#283c4b" },
  { shape: "head", position: [0, 0.228, 0], scale: [0.033, 0.035, 0.032], color: "#caa98c" },
  { shape: "helmet", position: [0, 0.245, 0], scale: [0.045, 0.042, 0.043], color: "#f2c35f" },
  { shape: "box", position: [0, 0.247, 0.043], scale: [0.021, 0.015, 0.015], color: "#fff5d8", basic: true },
  { shape: "box", position: [0, 0.15, 0.031], scale: [0.095, 0.019, 0.008], color: "#eceee0", basic: true },
  { shape: "box", position: [-0.062, 0.14, 0], scale: [0.027, 0.115, 0.032], color: "#d7a44e" },
  { shape: "box", position: [0.062, 0.14, 0], scale: [0.027, 0.115, 0.032], color: "#d7a44e" },
  { shape: "box", position: [0, 0.145, -0.043], scale: [0.062, 0.078, 0.035], color: "#465563" },
  { shape: "box", position: [0, 0.196, 0.032], scale: [0.095, 0.016, 0.008], color: "#eceee0", basic: true },
  { shape: "box", position: [0, 0.298, 0], scale: [0.056, 0.018, 0.025], color: "#ffffff", basic: true, indicator: "status" },
  { shape: "ring", position: [0, 0.011, 0], scale: [0.16, 0.16, 0.16], color: C.agentPanicHalo, basic: true, indicator: "panic" },
  { shape: "ring", position: [0, 0.017, 0], scale: [0.21, 0.21, 0.21], color: "#c1e7f5", basic: true, indicator: "follow" },
];

function InstancedPart({ part, workers, geometry, onSelect }: { part: Part; workers: Worker[]; geometry: THREE.BufferGeometry; onSelect: (id: string) => void }) {
  const ref = useRef<THREE.InstancedMesh>(null);
  useLayoutEffect(() => {
    const mesh = ref.current;
    if (!mesh) return;
    const root = new THREE.Object3D(), local = new THREE.Object3D(), matrix = new THREE.Matrix4();
    const color = new THREE.Color();
    workers.forEach((worker, i) => {
      root.position.set(...worker.position); root.rotation.set(0, worker.heading, 0); root.updateMatrix();
      local.position.set(...part.position); local.scale.set(...part.scale);
      local.rotation.set(part.shape === "ring" ? -Math.PI / 2 : 0, 0, 0);
      if (part.indicator === "panic" && worker.agent.panic_level < 0.6 || part.indicator === "follow" && !worker.followed) local.scale.setScalar(0);
      local.updateMatrix();
      mesh.setMatrixAt(i, matrix.multiplyMatrices(root.matrix, local.matrix));
      mesh.setColorAt(i, color.set(part.indicator === "status" ? STATUS[worker.agent.status] : part.color));
    });
    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
    mesh.computeBoundingSphere();
  }, [workers, part, geometry]);
  return <instancedMesh key={workers.length} ref={ref} args={[geometry, undefined, workers.length]} onClick={e => {
    if (e.instanceId === undefined) return;
    e.stopPropagation(); onSelect(String(workers[e.instanceId].agent.agent_id));
  }}>
    {part.basic ? <meshBasicMaterial side={THREE.DoubleSide} /> : <meshStandardMaterial roughness={0.7} />}
  </instancedMesh>;
}

export function Agents({ nodes, agents, followedAgentId, onSelectAgent, performanceMode }: AgentsProps) {
  const positions = useMemo(() => new Map(nodes.map(n => [n.node_id, backendToThreePosition(n.position)])), [nodes]);
  const geometry = useMemo(() => ({ box: new THREE.BoxGeometry(1, 1, 1),
    head: new THREE.SphereGeometry(1, 8, 6), helmet: new THREE.SphereGeometry(1, 8, 4, 0, Math.PI * 2, 0, Math.PI / 2),
    ring: new THREE.RingGeometry(0.86, 1, performanceMode ? 12 : 24) }), [performanceMode]);
  useEffect(() => () => Object.values(geometry).forEach(g => g.dispose()), [geometry]);
  const workers = useMemo(() => Object.values(agents).flatMap(agent => {
    if (agent.status === "evacuated") return [];
    const from = positions.get(agent.node_id), to = positions.get(agent.next_node_id) ?? from;
    if (!from || !to) return [];
    const position = lerpPosition(from, to, agent.progress);
    return [{ agent, position, heading: Math.atan2(to[0] - from[0], to[2] - from[2]), followed: followedAgentId === String(agent.agent_id) }];
  }), [agents, positions, followedAgentId]);
  // A bounded number of draws for 20 or 100 workers. Performance mode keeps PPE,
  // status, panic and selection while omitting arms, backpack and extra trim.
  const parts = performanceMode ? PARTS.filter((_, i) => ![3, 7, 8, 9, 10].includes(i)) : PARTS;
  if (!workers.length) return null;
  return <group>{parts.map((part, i) => <InstancedPart key={i} part={part} workers={workers} geometry={geometry[part.shape]} onSelect={onSelectAgent} />)}</group>;
}
