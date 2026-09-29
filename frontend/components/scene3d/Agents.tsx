"use client";

import { useMemo, useLayoutEffect, useEffect, useRef } from "react";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";
import type { AgentSnapshot, AgentStatus, MineNode } from "@/lib/types";
import { backendToThreePosition } from "./geometry";
import { SCENE_COLORS as C } from "./colors";
import { AgentMotion } from "./agentMotion";
import { createWorkerRig, WORKER_PARTS } from "./MineWorker";

interface AgentsProps {
  nodes: MineNode[]; agents: Record<string, AgentSnapshot>; followedAgentId: string | null;
  onSelectAgent: (id: string) => void; performanceMode: boolean;
}
const STATUS: Record<AgentStatus, string> = { moving: C.agentMoving, waiting: C.agentWaiting,
  sheltered: C.agentSheltered, evacuated: C.agentEvacuated, lost: C.agentLost };

export function Agents({ nodes, agents, followedAgentId, onSelectAgent, performanceMode }: AgentsProps) {
  const positions = useMemo(() => new Map(nodes.map(n => [n.node_id, new THREE.Vector3(...backendToThreePosition(n.position))])), [nodes]);
  const motions = useRef(new Map<number, AgentMotion>());
  const oldPositions = useRef(positions);
  const meshes = useRef<(THREE.InstancedMesh | null)[]>([]);
  const rig = useMemo(createWorkerRig, []);
  const color = useMemo(() => new THREE.Color(), []);
  const workers = useMemo(() => Object.values(agents).filter(a => a.status !== "evacuated" && positions.has(a.node_id)), [agents, positions]);
  const parts = useMemo(() => WORKER_PARTS.filter(p => !performanceMode || !p.detail), [performanceMode]);
  const geometry = useMemo(() => ({ box: new THREE.BoxGeometry(1, 1, 1),
    round: new THREE.SphereGeometry(1, performanceMode ? 6 : 10, performanceMode ? 4 : 8),
    helmet: new THREE.SphereGeometry(1, performanceMode ? 6 : 10, 4, 0, Math.PI * 2, 0, Math.PI / 2),
    ring: new THREE.RingGeometry(0.86, 1, performanceMode ? 12 : 24) }), [performanceMode]);
  const materials = useMemo(() => ({ standard: new THREE.MeshStandardMaterial({ roughness: 0.78 }),
    basic: new THREE.MeshBasicMaterial({ side: THREE.DoubleSide }) }), []);
  useEffect(() => () => Object.values(geometry).forEach(g => g.dispose()), [geometry]);
  useEffect(() => () => Object.values(materials).forEach(m => m.dispose()), [materials]);
  useLayoutEffect(() => {
    const now = performance.now(), states = motions.current;
    // MineGraphScene is already keyed by session_id; remounts isolate sessions.
    // Stable nodes identity changes on layout changes. A distance rollback is
    // authoritative evidence of reset, including workers already at rest.
    const reset = oldPositions.current !== positions || workers.some(a => {
      const old = states.get(a.agent_id);
      return old && a.cumulative_distance_m + 0.05 < old.agent.cumulative_distance_m;
    });
    if (reset) states.clear();
    oldPositions.current = positions;
    const active = new Set(workers.map(a => a.agent_id));
    states.forEach((_, id) => { if (!active.has(id)) states.delete(id); });
    workers.forEach(a => {
      const motion = states.get(a.agent_id);
      if (!motion || !motion.receive(a, positions, now)) states.set(a.agent_id, new AgentMotion(a, positions, now));
    });
    parts.forEach((part, p) => {
      const mesh = meshes.current[p];
      if (!mesh) return;
      mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
      workers.forEach((a, i) => mesh.setColorAt(i, color.set(part.indicator === "status" ? STATUS[a.status] : part.color)));
      if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
    });
  }, [workers, positions, parts, color]);

  // Single frame callback; no per-worker React updates or per-frame math objects.
  useFrame((_, delta) => {
    workers.forEach((a, i) => {
      const motion = motions.current.get(a.agent_id);
      if (!motion) return;
      motion.advance(delta); rig.pose(motion, performanceMode);
      parts.forEach((part, p) => {
        const hidden = part.indicator === "panic" && a.panic_level < 0.6
          || part.indicator === "follow" && followedAgentId !== String(a.agent_id);
        meshes.current[p]?.setMatrixAt(i, rig.matrix(part, hidden));
      });
    });
    parts.forEach((_, p) => {
      const mesh = meshes.current[p];
      if (!mesh) return;
      mesh.instanceMatrix.needsUpdate = true;
      // Raycasting needs current bounds even with frustum culling disabled.
      mesh.computeBoundingSphere();
    });
  }, -1);

  if (!workers.length) return null;
  // R3F owns each InstancedMesh and releases its instance buffers on removal.
  // Geometry/material are constructor arguments, not declarative children:
  // their shared lifetime is managed by the effects above. Do not set
  // dispose={null}: it replaces the mesh's actual dispose method with null.
  return <group>{parts.map((part, p) => <instancedMesh key={`${WORKER_PARTS.indexOf(part)}:${workers.length}`}
    ref={mesh => { meshes.current[p] = mesh; }} args={[geometry[part.shape], part.basic ? materials.basic : materials.standard, workers.length]}
    frustumCulled={false} onClick={e => {
      if (e.instanceId === undefined || !workers[e.instanceId]) return;
      e.stopPropagation(); onSelectAgent(String(workers[e.instanceId].agent_id));
    }} />)}</group>;
}
