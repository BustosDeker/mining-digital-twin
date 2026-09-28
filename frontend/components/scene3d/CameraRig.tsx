"use client";

import { useRef, useEffect, useMemo } from "react";
import { useFrame, useThree } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import type { OrbitControls as OrbitControlsImpl } from "three-stdlib";
import * as THREE from "three";
import type { AgentSnapshot, MineNode, MineViewMode } from "@/lib/types";
import { backendToThreePosition } from "./geometry";

interface CameraRigProps {
  nodes: MineNode[];
  agents: Record<string, AgentSnapshot>;
  followedAgentId: string | null;
  panMode: boolean;
  view: MineViewMode;
  fitRequest: number;
  focusNodes: MineNode[];
  onCancelFollow: () => void;
}

export function CameraRig({ nodes, agents, followedAgentId, panMode, view, fitRequest, focusNodes, onCancelFollow }: CameraRigProps) {
  const controlsRef = useRef<OrbitControlsImpl>(null);
  const { camera, size } = useThree();
  const positions = useMemo(() => new Map(nodes.map(n => [n.node_id, backendToThreePosition(n.position)])), [nodes]);
  const scratch = useMemo(() => ({ target: new THREE.Vector3(), to: new THREE.Vector3(), delta: new THREE.Vector3() }), []);
  const focusSignature = focusNodes.map(n => `${n.node_id}:${n.position.join(",")}`).join("|");
  const focusRef = useRef(focusNodes);
  focusRef.current = focusNodes;

  useEffect(() => {
    const controls = controlsRef.current;
    if (!controls || !focusRef.current.length) return;
    const bounds = new THREE.Box3();
    focusRef.current.forEach(n => bounds.expandByPoint(new THREE.Vector3(...backendToThreePosition(n.position))));
    bounds.expandByScalar(0.7);
    const center = bounds.getCenter(new THREE.Vector3());
    const vertical = THREE.MathUtils.degToRad((camera as THREE.PerspectiveCamera).fov / 2);
    const horizontal = Math.atan(Math.tan(vertical) * size.width / Math.max(1, size.height));
    const direction = view === "top" ? new THREE.Vector3(0, 1, 0.001)
      : view === "lateral" ? new THREE.Vector3(0, 0.12, 1) : new THREE.Vector3(1, 0.72, 1);
    direction.normalize();
    const right = new THREE.Vector3(0, 1, 0).cross(direction).normalize();
    const up = direction.clone().cross(right).normalize();
    // Fit projected real points, instead of a bounding sphere that wastes screen space.
    const offset = new THREE.Vector3();
    let distance = view === "incidents" ? 4.5 : 3;
    for (const node of focusRef.current) {
      offset.set(...backendToThreePosition(node.position)).sub(center);
      distance = Math.max(distance, offset.dot(direction) + Math.max(
        (Math.abs(offset.dot(right)) + 0.7) / (Math.tan(horizontal) * 0.78),
        (Math.abs(offset.dot(up)) + 0.7) / (Math.tan(vertical) * 0.8)
      ));
    }
    controls.target.copy(center);
    camera.position.copy(center).addScaledVector(direction.normalize(), distance);
    controls.maxDistance = Math.max(220, distance * 3);
    camera.far = Math.max(1000, distance * 5);
    camera.updateProjectionMatrix();
    controls.update();
  }, [camera, view, fitRequest, focusSignature, size.width, size.height]);

  useEffect(() => {
    if (!followedAgentId || !controlsRef.current) return;
    const agent = agents[followedAgentId];
    const from = agent && positions.get(agent.node_id);
    const to = agent && positions.get(agent.next_node_id);
    if (!from) return;
    scratch.target.set(...from);
    if (to) scratch.target.lerp(scratch.to.set(...to), agent.progress);
    controlsRef.current.target.copy(scratch.target);
    camera.position.copy(scratch.target).add(scratch.delta.set(2.2, 1.8, 2.2));
    controlsRef.current.update();
    // Reposition only when following changes, not at every snapshot.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [followedAgentId]);

  useFrame((_, delta) => {
    const controls = controlsRef.current;
    const agent = followedAgentId ? agents[followedAgentId] : null;
    if (!controls || !agent || agent.status === "evacuated") return;
    const from = positions.get(agent.node_id), to = positions.get(agent.next_node_id) ?? from;
    if (!from || !to) return;
    scratch.target.set(...from).lerp(scratch.to.set(...to), agent.progress);
    scratch.delta.copy(scratch.target).sub(controls.target).multiplyScalar(1 - Math.exp(-6 * delta));
    controls.target.add(scratch.delta);
    camera.position.add(scratch.delta);
    controls.update();
  });

  return <OrbitControls ref={controlsRef} makeDefault enableDamping dampingFactor={0.08}
    minDistance={0.65} maxDistance={220} minPolarAngle={0.001} maxPolarAngle={Math.PI * 0.88}
    enablePan panSpeed={1} rotateSpeed={0.7} zoomSpeed={0.9} screenSpacePanning zoomToCursor
    onStart={() => { if (followedAgentId) onCancelFollow(); }}
    mouseButtons={panMode ? { LEFT: THREE.MOUSE.PAN, MIDDLE: THREE.MOUSE.DOLLY, RIGHT: THREE.MOUSE.ROTATE }
      : { LEFT: THREE.MOUSE.ROTATE, MIDDLE: THREE.MOUSE.DOLLY, RIGHT: THREE.MOUSE.PAN }} />;
}
