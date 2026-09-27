"use client";

import { useI18n } from "@/contexts/I18nContext";
import type { HazardEvent } from "@/lib/types";

const HAZARD_LABEL_KEY: Record<HazardEvent["hazard_type"], "hazardFire" | "hazardCollapse" | "hazardGasLeak"> = {
  fire: "hazardFire",
  collapse: "hazardCollapse",
  gas_leak: "hazardGasLeak",
};

export function EventLogPanel({ hazards }: { hazards: HazardEvent[] }) {
  const { t } = useI18n();

  return (
    <div className="border-b border-hairline p-4">
      <h3 className="mb-2 text-[11px] font-semibold uppercase tracking-wide text-steel">
        {t.eventLog.title}
      </h3>
      {hazards.length === 0 ? (
        <p className="text-[12px] text-steel">{t.eventLog.empty}</p>
      ) : (
        <ul className="flex flex-col gap-2">
          {hazards.map((h) => (
            <li key={h.event_id} className="rounded-sm border border-amber/30 bg-amber/5 p-2 text-[11px]">
              <div className="flex items-center justify-between font-medium text-amber">
                <span>{t.scenario[HAZARD_LABEL_KEY[h.hazard_type]]}</span>
                <span className="font-mono">{Math.round(h.intensity * 100)}%</span>
              </div>
              <div className="mt-0.5 font-mono text-steel">
                {h.origin_node_id} · {t.eventLog.startedAtStep} {h.started_at_step}
              </div>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

const LEGEND_ITEMS: { key: keyof ReturnType<typeof useI18n>["t"]["legend"]; color: string }[] = [
  { key: "clear", color: "#33FFB2" },
  { key: "degraded", color: "#FFB020" },
  { key: "blocked", color: "#FF5C5C" },
  { key: "exit", color: "#33FFB2" },
  { key: "refuge", color: "#5FA8FF" },
  { key: "riskZone", color: "#FFB020" },
  { key: "agent", color: "#ECEDE9" },
];

export function LegendPanel() {
  const { t } = useI18n();
  return (
    <div className="p-4">
      <h3 className="mb-2 text-[11px] font-semibold uppercase tracking-wide text-steel">
        {t.legend.title}
      </h3>
      <ul className="grid grid-cols-2 gap-x-3 gap-y-1.5">
        {LEGEND_ITEMS.map((item) => (
          <li key={item.key} className="flex items-center gap-1.5 text-[11px] text-steel">
            <span
              className="h-2 w-2 shrink-0 rounded-full"
              style={{ backgroundColor: item.color }}
            />
            {t.legend[item.key]}
          </li>
        ))}
      </ul>
    </div>
  );
}
