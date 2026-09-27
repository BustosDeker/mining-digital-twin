"use client";

import { useState } from "react";
import { useI18n } from "@/contexts/I18nContext";
import type { CreateSessionInput, HazardType } from "@/lib/types";

interface ScenarioLauncherPanelProps {
  open: boolean;
  onClose: () => void;
  onLaunch: (payload: CreateSessionInput) => Promise<void>;
}

function NumberField({
  label,
  value,
  onChange,
  min,
  max,
  step = 1,
}: {
  label: string;
  value: number;
  onChange: (v: number) => void;
  min: number;
  max: number;
  step?: number;
}) {
  return (
    <label className="flex flex-col gap-1 text-[11px] text-steel">
      {label}
      <input
        type="number"
        value={value}
        min={min}
        max={max}
        step={step}
        onChange={(e) => onChange(Number(e.target.value))}
        className="rounded-sm border border-hairline bg-panel2 px-2 py-1 font-mono text-[12px] text-steel2 outline-none focus:border-signal/50"
      />
    </label>
  );
}

export function ScenarioLauncherPanel({
  open,
  onClose,
  onLaunch,
}: ScenarioLauncherPanelProps) {
  const { t } = useI18n();
  const [scenarioName, setScenarioName] = useState("scenario_1");
  const [nAgents, setNAgents] = useState(20);
  const [router, setRouter] = useState<"adaptive_astar" | "q_learning">("adaptive_astar");
  const [hazardType, setHazardType] = useState<HazardType | "none">("fire");
  const [hazardIntensity, setHazardIntensity] = useState(0.9);
  const [nLevels, setNLevels] = useState(3);
  const [galleriesPerLevel, setGalleriesPerLevel] = useState(8);
  const [nRefuges, setNRefuges] = useState(3);
  const [nExits, setNExits] = useState(2);
  const [nRiskZones, setNRiskZones] = useState(4);
  const [seed, setSeed] = useState(42);
  const [submitting, setSubmitting] = useState(false);

  if (!open) return null;

  const handleSubmit = async () => {
    setSubmitting(true);
    try {
      await onLaunch({
        scenario_name: scenarioName,
        n_agents: nAgents,
        router_name: router,
        hazard_type: hazardType === "none" ? null : hazardType,
        hazard_intensity: hazardIntensity,
        layout_config: {
          n_levels: nLevels,
          galleries_per_level: galleriesPerLevel,
          n_refuge_chambers: nRefuges,
          n_exits: nExits,
          n_risk_zones: nRiskZones,
          random_seed: seed,
        },
      });
      onClose();
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="absolute inset-0 z-20 flex items-center justify-center bg-void/70">
      <div className="w-[440px] rounded border border-hairline bg-panel p-5">
        <h2 className="mb-4 text-[13px] font-semibold uppercase tracking-wide text-steel2">
          {t.scenario.title}
        </h2>

        <div className="flex flex-col gap-3">
          <label className="flex flex-col gap-1 text-[11px] text-steel">
            Nombre del escenario
            <input
              value={scenarioName}
              onChange={(e) => setScenarioName(e.target.value)}
              className="rounded-sm border border-hairline bg-panel2 px-2 py-1 font-mono text-[12px] text-steel2 outline-none focus:border-signal/50"
            />
          </label>

          <NumberField label={t.scenario.nAgents} value={nAgents} onChange={setNAgents} min={1} max={100} />

          <label className="flex flex-col gap-1 text-[11px] text-steel">
            {t.scenario.router}
            <select
              value={router}
              onChange={(e) => setRouter(e.target.value as typeof router)}
              className="rounded-sm border border-hairline bg-panel2 px-2 py-1 text-[12px] text-steel2 outline-none focus:border-signal/50"
            >
              <option value="adaptive_astar">{t.scenario.routerAstar}</option>
              <option value="q_learning">{t.scenario.routerQLearning}</option>
            </select>
          </label>

          <div className="grid grid-cols-2 gap-3">
            <label className="flex flex-col gap-1 text-[11px] text-steel">
              {t.scenario.hazardType}
              <select
                value={hazardType}
                onChange={(e) => setHazardType(e.target.value as HazardType | "none")}
                className="rounded-sm border border-hairline bg-panel2 px-2 py-1 text-[12px] text-steel2 outline-none focus:border-signal/50"
              >
                <option value="fire">{t.scenario.hazardFire}</option>
                <option value="collapse">{t.scenario.hazardCollapse}</option>
                <option value="gas_leak">{t.scenario.hazardGasLeak}</option>
                <option value="none">{t.scenario.hazardNone}</option>
              </select>
            </label>
            <NumberField
              label={t.scenario.hazardIntensity}
              value={hazardIntensity}
              onChange={setHazardIntensity}
              min={0}
              max={1}
              step={0.05}
            />
          </div>

          <div className="grid grid-cols-3 gap-3">
            <NumberField label={t.scenario.levels} value={nLevels} onChange={setNLevels} min={1} max={6} />
            <NumberField
              label={t.scenario.galleries}
              value={galleriesPerLevel}
              onChange={setGalleriesPerLevel}
              min={1}
              max={30}
            />
            <NumberField label={t.scenario.seed} value={seed} onChange={setSeed} min={0} max={99999} />
          </div>

          <div className="grid grid-cols-3 gap-3">
            <NumberField label={t.scenario.refuges} value={nRefuges} onChange={setNRefuges} min={0} max={10} />
            <NumberField label={t.scenario.exits} value={nExits} onChange={setNExits} min={1} max={6} />
            <NumberField
              label={t.scenario.riskZones}
              value={nRiskZones}
              onChange={setNRiskZones}
              min={0}
              max={15}
            />
          </div>
        </div>

        <div className="mt-5 flex justify-end gap-2">
          <button
            onClick={onClose}
            className="rounded-sm border border-hairline px-3 py-1.5 text-[12px] text-steel hover:text-steel2"
          >
            {t.common.close}
          </button>
          <button
            onClick={handleSubmit}
            disabled={submitting}
            className="rounded-sm border border-signal/40 px-3 py-1.5 text-[12px] font-medium text-signal hover:bg-signal/10 disabled:opacity-50"
          >
            {submitting ? t.common.loading : t.scenario.launch}
          </button>
        </div>
      </div>
    </div>
  );
}
