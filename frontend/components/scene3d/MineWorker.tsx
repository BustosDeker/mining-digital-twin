import * as THREE from "three";
import type { AgentMotion } from "./agentMotion";
import { SCENE_COLORS as C } from "./colors";

type Shape = "box" | "round" | "helmet" | "ring";
type Joint = "root" | "body" | "torso" | "head" | "leftArm" | "rightArm" | "leftForearm" | "rightForearm"
  | "leftThigh" | "rightThigh" | "leftShin" | "rightShin" | "leftFoot" | "rightFoot";
export interface WorkerPart {
  joint: Joint; shape: Shape; position: [number, number, number]; scale: [number, number, number];
  color: string; basic?: boolean; detail?: boolean; indicator?: "status" | "panic" | "follow";
}
const CLOTH = "#d7a44e", PANTS = "#283c4b", TRIM = "#eceee0";
export const WORKER_PARTS: WorkerPart[] = [
  { joint: "body", shape: "round", position: [0, 0.104, 0], scale: [0.033, 0.019, 0.022], color: PANTS },
  { joint: "torso", shape: "round", position: [0, 0.025, 0], scale: [0.032, 0.048, 0.026], color: CLOTH },
  { joint: "torso", shape: "box", position: [0, 0.012, 0.024], scale: [0.054, 0.009, 0.007], color: TRIM, basic: true },
  ...([-1, 1] as const).map(side => ({ joint: "torso" as const, shape: "box" as const,
    position: [side * 0.017, 0.042, 0.021] as [number, number, number], scale: [0.008, 0.039, 0.007] as [number, number, number],
    color: TRIM, basic: true, detail: true })),
  { joint: "torso", shape: "box", position: [0, 0.027, -0.03], scale: [0.045, 0.057, 0.024], color: "#465563", detail: true },
  { joint: "head", shape: "round", position: [0, -0.022, 0], scale: [0.012, 0.016, 0.012], color: "#caa98c", detail: true },
  { joint: "head", shape: "round", position: [0, 0, 0], scale: [0.020, 0.024, 0.021], color: "#caa98c" },
  { joint: "head", shape: "helmet", position: [0, 0.012, 0], scale: [0.027, 0.023, 0.028], color: "#f2c35f" },
  { joint: "head", shape: "round", position: [0, 0.012, 0.003], scale: [0.030, 0.004, 0.032], color: "#f2c35f", detail: true },
  { joint: "head", shape: "box", position: [0, 0.023, 0.027], scale: [0.016, 0.012, 0.012], color: "#fff5d8", basic: true },
  ...(["left", "right"] as const).flatMap(side => [
    { joint: `${side}Arm` as Joint, shape: "round" as Shape, position: [0, -0.022, 0] as [number, number, number], scale: [0.013, 0.028, 0.014] as [number, number, number], color: CLOTH },
    { joint: `${side}Forearm` as Joint, shape: "round" as Shape, position: [0, -0.020, 0] as [number, number, number], scale: [0.011, 0.025, 0.012] as [number, number, number], color: CLOTH },
    { joint: `${side}Forearm` as Joint, shape: "round" as Shape, position: [0, -0.044, 0] as [number, number, number], scale: [0.010, 0.013, 0.010] as [number, number, number], color: "#caa98c", detail: true },
    { joint: `${side}Thigh` as Joint, shape: "round" as Shape, position: [0, -0.0215, 0] as [number, number, number], scale: [0.016, 0.027, 0.018] as [number, number, number], color: PANTS },
    { joint: `${side}Shin` as Joint, shape: "round" as Shape, position: [0, -0.0215, 0] as [number, number, number], scale: [0.013, 0.027, 0.015] as [number, number, number], color: PANTS },
    { joint: `${side}Shin` as Joint, shape: "box" as Shape, position: [0, -0.026, 0.014] as [number, number, number], scale: [0.025, 0.007, 0.004] as [number, number, number], color: TRIM, basic: true, detail: true },
    { joint: `${side}Foot` as Joint, shape: "box" as Shape, position: [0, -0.007, 0.008] as [number, number, number], scale: [0.03, 0.022, 0.047] as [number, number, number], color: "#17252e" },
  ]),
  { joint: "root", shape: "box", position: [0, 0.298, 0], scale: [0.056, 0.018, 0.025], color: "#ffffff", basic: true, indicator: "status" },
  { joint: "root", shape: "ring", position: [0, 0.011, 0], scale: [0.16, 0.16, 0.16], color: C.agentPanicHalo, basic: true, indicator: "panic" },
  { joint: "root", shape: "ring", position: [0, 0.017, 0], scale: [0.21, 0.21, 0.21], color: "#c1e7f5", basic: true, indicator: "follow" },
];

/** One reusable CPU hierarchy, evaluated into instance matrices for ALL workers. */
export function createWorkerRig() {
  const joints = Object.fromEntries((["root", "body", "torso", "head", "leftArm", "rightArm", "leftForearm", "rightForearm",
    "leftThigh", "rightThigh", "leftShin", "rightShin", "leftFoot", "rightFoot"] as Joint[]).map(key => [key, new THREE.Object3D()])) as Record<Joint, THREE.Object3D>;
  const attach = (child: Joint, parent: Joint, x: number, y: number, z = 0) => {
    joints[parent].add(joints[child]); joints[child].position.set(x, y, z);
  };
  attach("body", "root", 0, 0); attach("torso", "body", 0, 0.135); attach("head", "body", 0, 0.215);
  for (const side of ["left", "right"] as const) {
    const sign = side === "left" ? -1 : 1;
    attach(`${side}Arm`, "torso", sign * 0.034, 0.047);
    attach(`${side}Forearm`, `${side}Arm`, 0, -0.043);
    attach(`${side}Thigh`, "body", sign * 0.020, 0.104);
    attach(`${side}Shin`, `${side}Thigh`, 0, -0.043);
    attach(`${side}Foot`, `${side}Shin`, 0, -0.043);
  }
  const local = new THREE.Object3D(), matrix = new THREE.Matrix4();
  return {
    pose(motion: AgentMotion, simple: boolean) {
      const w = motion.weight, s = Math.sin(motion.phase);
      joints.root.position.copy(motion.position); joints.root.quaternion.copy(motion.quaternion);
      joints.body.position.y = 0;
      joints.torso.rotation.set(simple ? 0 : w * 0.035, simple ? 0 : s * w * 0.025, 0);
      joints.torso.scale.y = 1 + (simple ? 0 : Math.sin(motion.idleTime * 1.7) * 0.006 * (1 - w));
      let sole = Infinity;
      for (const side of ["left", "right"] as const) {
        const phase = motion.phase + (side === "left" ? 0 : Math.PI);
        const swing = Math.sin(phase), hip = -swing * 0.34 * w;
        const knee = Math.max(0, Math.cos(phase)) * 0.42 * w;
        joints[`${side}Thigh`].rotation.x = hip;
        joints[`${side}Shin`].rotation.x = knee;
        joints[`${side}Foot`].rotation.x = -hip - knee - Math.atan(motion.slope);
        joints[`${side}Arm`].rotation.x = swing * 0.27 * w;
        joints[`${side}Forearm`].rotation.x = -0.12 - (simple ? 0 : Math.max(0, -swing) * 0.14 * w);
        const footZ = -0.043 * (Math.sin(hip) + Math.sin(hip + knee));
        const footY = 0.104 - 0.043 * (Math.cos(hip) + Math.cos(hip + knee)) - 0.018 * Math.hypot(1, motion.slope);
        sole = Math.min(sole, footY - footZ * motion.slope);
      }
      // Tunnel floor surface is at the logical centerline (local y=0).
      // Keep the lowest sole on that plane; compensate only the local model.
      joints.body.position.y = -sole + (simple ? 0 : 0.0008 * w * (1 - Math.cos(motion.phase * 2)));
      joints.head.position.y = 0.215 - joints.body.position.y * 0.35;
      joints.root.updateMatrixWorld(true);
    },
    matrix(part: WorkerPart, hidden: boolean) {
      local.position.set(...part.position); local.scale.set(...part.scale);
      if (hidden) local.scale.setScalar(0);
      local.rotation.set(part.shape === "ring" ? -Math.PI / 2 : 0, 0, 0);
      local.updateMatrix();
      return matrix.multiplyMatrices(joints[part.joint].matrixWorld, local.matrix);
    },
  };
}
