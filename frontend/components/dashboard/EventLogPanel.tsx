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
            <li key={h.event_id} className="twin-event" data-hazard={h.hazard_type}>
              <div className="flex items-center justify-between font-medium text-amber">
                <span>{h.hazard_type === "fire" ? "♨" : h.hazard_type === "collapse" ? "▧" : "≋"} {t.scenario[HAZARD_LABEL_KEY[h.hazard_type]]}</span>
                <span className="font-mono">{Math.round(h.intensity * 100)}%</span>
              </div>
              <div className="mt-0.5 font-mono text-steel">
                {h.origin_node_id} · {t.eventLog.startedAtStep} {h.started_at_step}
              </div>
              <div className="mt-1 text-steel">{t.twin.affected}: {h.affected_edges.length}</div>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

const LEGEND_ITEMS: { key: keyof ReturnType<typeof useI18n>["t"]["legend"]; color: string }[] = [
  { key: "clear", color: "var(--twin-safe)" },
  { key: "degraded", color: "var(--twin-amber)" },
  { key: "blocked", color: "var(--twin-red)" },
  { key: "exit", color: "var(--twin-safe)" },
  { key: "refuge", color: "var(--twin-blue)" },
  { key: "riskZone", color: "var(--twin-amber)" },
  { key: "agent", color: "var(--twin-ink)" },
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
              className="twin-legend-symbol"
              style={{ color: item.color }}
            >{({clear: "━", degraded: "!", blocked: "×", exit: "⇥", refuge: "+", riskZone: "△", agent: "●"} as Record<string, string>)[item.key]}</span>
            {t.legend[item.key]}
          </li>
        ))}
      </ul>
    </div>
  );
}
