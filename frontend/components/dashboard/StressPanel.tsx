"use client";

import { useMemo } from "react";
import { useI18n } from "@/contexts/I18nContext";
import type { AgentSnapshot } from "@/lib/types";

interface StressPanelProps {
  agents: Record<string, AgentSnapshot>;
  step: number;
}

function StatRow({ label, value, color }: { label: string; value: number; color?: string }) {
  return (
    <div className="flex items-center justify-between py-1 text-[12px]">
      <span className="text-steel">{label}</span>
      <span className="font-mono font-medium" style={{ color }}>
        {value}
      </span>
    </div>
  );
}

export function StressPanel({ agents, step }: StressPanelProps) {
  const { t } = useI18n();

  const stats = useMemo(() => {
    const list = Object.values(agents);
    const total = list.length || 1;
    const avgPanic = list.reduce((sum, a) => sum + a.panic_level, 0) / total;
    const countBy = (status: AgentSnapshot["status"]) =>
      list.filter((a) => a.status === status).length;

    return {
      avgPanic,
      evacuated: countBy("evacuated"),
      sheltered: countBy("sheltered"),
      lost: countBy("lost"),
      moving: countBy("moving") + countBy("waiting"),
    };
  }, [agents]);

  const panicPct = Math.round(stats.avgPanic * 100);

  return (
    <div className="border-b border-hairline p-4">
      <h3 className="mb-2 text-[11px] font-semibold uppercase tracking-wide text-steel">
        {t.stressPanel.title}
      </h3>

      <div className="mb-3">
        <div className="mb-1 flex items-center justify-between text-[11px] text-steel">
          <span>{t.stressPanel.avgPanic}</span>
          <span className="font-mono text-steel2">{panicPct}%</span>
        </div>
        <div className="h-1.5 w-full rounded-full bg-panel2">
          <div
            className="h-1.5 rounded-full transition-all"
            style={{
              width: `${panicPct}%`,
              backgroundColor:
                panicPct >= 60 ? "#FF5C5C" : panicPct >= 30 ? "#FFB020" : "#33FFB2",
            }}
          />
        </div>
      </div>

      <StatRow label={t.stressPanel.moving} value={stats.moving} color="#ECEDE9" />
      <StatRow label={t.stressPanel.evacuated} value={stats.evacuated} color="#33FFB2" />
      <StatRow label={t.stressPanel.sheltered} value={stats.sheltered} color="#5FA8FF" />
      <StatRow label={t.stressPanel.lost} value={stats.lost} color="#FF5C5C" />

      <div className="mt-2 border-t border-hairline pt-2 text-[11px] text-steel">
        {t.stressPanel.step}: <span className="font-mono text-steel2">{step}</span>
      </div>
    </div>
  );
}
