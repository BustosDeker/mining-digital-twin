import * as THREE from "three";
import type { AgentSnapshot } from "@/lib/types";

export type NodePositions = Map<string, THREE.Vector3>;
const EPS = 1e-7;
const UP = new THREE.Vector3(0, 1, 0);
const TAU = Math.PI * 2;

/** Only snapshot endpoints authorize travel; this module never searches a route. */
export class AgentMotion {
  readonly position = new THREE.Vector3();
  readonly quaternion = new THREE.Quaternion();
  private readonly facing = new THREE.Quaternion();
  private readonly direction = new THREE.Vector3();
  private path: THREE.Vector3[] = [];
  private segment = 1;
  private elapsed = 0;
  private duration = 1;
  private length = 0;
  private traveled = 0;
  private receivedAt = 0;
  private samples = 0;
  interval = 1;
  phase: number;
  weight = 0;
  idleTime = 0;
  slope = 0;
  agent: AgentSnapshot;

  constructor(agent: AgentSnapshot, positions: NodePositions, now: number) {
    this.agent = agent;
    this.phase = ((Math.imul(agent.agent_id + 1, 2654435761) >>> 0) / 4294967296) * TAU;
    this.idleTime = this.phase;
    const from = positions.get(agent.node_id)!;
    const to = positions.get(agent.next_node_id) ?? from;
    this.position.copy(from).lerp(to, THREE.MathUtils.clamp(agent.progress, 0, 1));
    this.direction.subVectors(to, from);
    this.setFacing();
    this.quaternion.copy(this.facing);
    this.receivedAt = now;
  }

  private setFacing() {
    const horizontal = Math.hypot(this.direction.x, this.direction.z);
    if (horizontal > EPS) {
      this.facing.setFromAxisAngle(UP, Math.atan2(this.direction.x, this.direction.z));
      this.slope = this.direction.y / horizontal;
    } else this.slope = 0;
  }

  /** false means discontinuity: caller replaces the entire visual state. */
  receive(agent: AgentSnapshot, positions: NodePositions, now: number): boolean {
    const old = this.agent;
    if (agent.cumulative_distance_m + 0.05 < old.cumulative_distance_m) return false;
    const from = positions.get(agent.node_id)!;
    const to = positions.get(agent.next_node_id) ?? from;
    const target = from.clone().lerp(to, THREE.MathUtils.clamp(agent.progress, 0, 1));
    const oldFrom = positions.get(old.node_id)!;
    const oldTo = positions.get(old.next_node_id) ?? oldFrom;
    const oldTarget = oldFrom.clone().lerp(oldTo, THREE.MathUtils.clamp(old.progress, 0, 1));
    const sameEdge = old.node_id === agent.node_id && old.next_node_id === agent.next_node_id
      && (old.edge_id === agent.edge_id || old.edge_id === null || agent.edge_id === null);
    this.agent = agent;
    if (sameEdge && target.distanceToSquared(oldTarget) < EPS * EPS) return true;
    // A missing intermediate snapshot is not permission to invent a connecting path.
    const forwardJoin = old.next_node_id === agent.node_id;
    const backToNode = old.node_id === agent.node_id && !sameEdge;
    if (!sameEdge && !forwardJoin && !backToNode) return false;
    if (sameEdge && agent.progress + 1e-5 < old.progress) return false;
    if (target.distanceTo(oldTarget) > 3) return false;

    const changedPosition = target.distanceToSquared(oldTarget) >= EPS * EPS;
    const observed = (now - this.receivedAt) / 1000;
    // Ignore pauses/stalls as cadence samples. First moving update bootstraps;
    // subsequent updates measure the real cadence, not the simulation timestep.
    if (changedPosition && this.samples > 0 && observed > 0.04 && observed < Math.max(3, this.interval * 3)) {
      this.interval = this.samples === 1 ? observed : this.interval * 0.75 + observed * 0.25;
    }
    if (changedPosition) { this.samples++; this.receivedAt = now; }
    const remaining = this.path.slice(this.segment, -1);
    if (!sameEdge) {
      // Keep outstanding confirmed corners; replace the previous endpoint with
      // the shared real node. Never connect the two galleries with a diagonal.
      const joint = forwardJoin ? oldTo : oldFrom;
      remaining.push(joint.clone());
    }
    this.path = [this.position.clone(), ...remaining, target];
    // Keep a repeated final point: it records a corner at progress=0. Removing
    // it would erase the corner when an early update extends the next edge.
    if (this.path.length > 8) return false;
    this.length = 0;
    for (let i = 1; i < this.path.length; i++) this.length += this.path[i - 1].distanceTo(this.path[i]);
    if (this.length > 3) return false;
    this.segment = 1;
    this.elapsed = this.traveled = 0;
    this.duration = THREE.MathUtils.clamp(this.interval, 0.08, 2.5);
    return true;
  }

  advance(delta: number) {
    // At most 50 ms of animation on resume; no giant limb/rotation update.
    const dt = Math.max(0, Math.min(delta, 0.05));
    this.idleTime = (this.idleTime + dt) % (TAU * 10);
    this.elapsed = Math.min(this.duration, this.elapsed + dt);
    // Linear time is deliberately monotone and has no per-snapshot easing stop.
    // The bounded polyline ends EXACTLY at the last authorized target.
    const distance = this.length * Math.min(1, this.elapsed / this.duration);
    let remaining = Math.max(0, distance - this.traveled);
    const moved = remaining;
    while (remaining > EPS && this.segment < this.path.length) {
      const end = this.path[this.segment];
      this.direction.subVectors(end, this.position);
      const available = this.direction.length();
      this.setFacing();
      if (available <= remaining + EPS) {
        this.position.copy(end);
        remaining = Math.max(0, remaining - available);
        this.segment++;
      } else {
        this.position.addScaledVector(this.direction, remaining / available);
        remaining = 0;
      }
    }
    this.traveled = distance;
    const walking = this.agent.status === "moving" && dt > 0 && moved / dt > 0.002;
    if (walking) this.phase = (this.phase + moved * TAU / 0.145) % TAU;
    this.weight += ((walking ? 1 : 0) - this.weight) * (1 - Math.exp(-12 * dt));
    this.quaternion.slerp(this.facing, 1 - Math.exp(-12 * dt));
  }
}
