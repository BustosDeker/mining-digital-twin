"use client";

import clsx from "clsx";
import { useI18n } from "@/contexts/I18nContext";
import type { SessionStatus } from "@/lib/types";

interface ControlBarProps {
  status: SessionStatus | null;
  onStart: () => void;
  onPause: () => void;
  onStop: () => void;
  onReset: () => void;
  onStep: () => void;
  onNewScenario: () => void;
  disabled: boolean;
  tunnelOpacity: number;
  onTunnelOpacityChange: (value: number) => void;
  showLabels: boolean;
  onShowLabelsChange: (value: boolean) => void;
  performanceMode: boolean;
  onPerformanceModeChange: (value: boolean) => void;
}

function ControlButton({
  label,
  onClick,
  disabled,
  variant = "default",
}: {
  label: string;
  onClick: () => void;
  disabled?: boolean;
  variant?: "default" | "primary";
}) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className={clsx(
        "rounded-sm border px-3 py-1.5 text-[12px] font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-40",
        variant === "primary"
          ? "border-signal/40 text-signal hover:bg-signal/10"
          : "border-hairline text-steel2 hover:bg-panel2"
      )}
    >
      {label}
    </button>
  );
}

export function ControlBar({
  status,
  onStart,
  onPause,
  onStop,
  onReset,
  onStep,
  onNewScenario,
  disabled,
  tunnelOpacity,
  onTunnelOpacityChange,
  showLabels,
  onShowLabelsChange,
  performanceMode,
  onPerformanceModeChange,
}: ControlBarProps) {
  const { t } = useI18n();
  const isRunning = status === "running";

  return (
    <div className="flex flex-wrap items-center gap-2 border-b border-hairline bg-panel px-4 py-2">
      <ControlButton
        label={`▶ ${t.controls.start}`}
        onClick={onStart}
        disabled={disabled || isRunning}
        variant="primary"
      />
      <ControlButton label={`⏸ ${t.controls.pause}`} onClick={onPause} disabled={disabled || !isRunning} />
      <ControlButton label={`■ ${t.controls.stop}`} onClick={onStop} disabled={disabled} />
      <ControlButton label={`↺ ${t.controls.reset}`} onClick={onReset} disabled={disabled} />
      <ControlButton label={`⏭ ${t.controls.step}`} onClick={onStep} disabled={disabled || isRunning} />

      <div className="mx-2 h-5 w-px bg-hairline" />

      <ControlButton label={`+ ${t.controls.newScenario}`} onClick={onNewScenario} />

      <div className="mx-2 h-5 w-px bg-hairline" />

      <label className="flex items-center gap-2 font-mono text-[11px] text-steel">
        {t.controls.opacity}
        <input
          type="range"
          min={0.1}
          max={1}
          step={0.05}
          value={tunnelOpacity}
          onChange={(e) => onTunnelOpacityChange(Number(e.target.value))}
          className="w-20 accent-signal"
        />
      </label>

      <label className="flex items-center gap-1.5 font-mono text-[11px] text-steel">
        <input
          type="checkbox"
          checked={showLabels}
          onChange={(e) => onShowLabelsChange(e.target.checked)}
          className="accent-signal"
        />
        {t.controls.labels}
      </label>

      <label className="flex items-center gap-1.5 font-mono text-[11px] text-steel">
        <input
          type="checkbox"
          checked={performanceMode}
          onChange={(e) => onPerformanceModeChange(e.target.checked)}
          className="accent-signal"
        />
        {t.controls.performance}
      </label>
    </div>
  );
}
