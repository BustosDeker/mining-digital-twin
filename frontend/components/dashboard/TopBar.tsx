"use client";

import clsx from "clsx";
import { useI18n } from "@/contexts/I18nContext";
import { useTheme } from "@/contexts/ThemeContext";

export type DashboardTab = "twin" | "ml" | "sessions" | "reports";

interface TopBarProps {
  activeTab: DashboardTab;
  onTabChange: (tab: DashboardTab) => void;
  connectionLabel: string | null;
  connected: boolean;
}

export function TopBar({
  activeTab,
  onTabChange,
  connectionLabel,
  connected,
}: TopBarProps) {
  const { t, locale, setLocale } = useI18n();
  const { theme, toggleTheme } = useTheme();

  const tabs: { id: DashboardTab; label: string }[] = [
    { id: "twin", label: t.nav.digitalTwin },
    { id: "ml", label: t.nav.mlResults },
    { id: "sessions", label: t.nav.sessions },
    { id: "reports", label: t.nav.reports },
  ];

  return (
    <header className="flex h-12 shrink-0 items-center justify-between border-b border-hairline bg-panel px-4">
      <div className="flex items-center gap-6">
        <h1 className="text-[13px] font-semibold uppercase tracking-wide text-steel2">
          {t.appTitle}
        </h1>
        <nav className="flex items-center gap-1">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => onTabChange(tab.id)}
              className={clsx(
                "rounded-sm px-3 py-1.5 text-[12px] font-medium transition-colors",
                activeTab === tab.id
                  ? "bg-panel2 text-signal"
                  : "text-steel hover:text-steel2"
              )}
            >
              {tab.label}
            </button>
          ))}
        </nav>
      </div>

      <div className="flex items-center gap-4">
        {connectionLabel && (
          <div className="flex items-center gap-1.5 font-mono text-[11px] text-steel">
            <span
              className={clsx(
                "h-1.5 w-1.5 rounded-full",
                connected ? "bg-signal animate-pulse-signal" : "bg-amber"
              )}
            />
            {connectionLabel}
          </div>
        )}

        <button
          onClick={() => setLocale(locale === "es" ? "en" : "es")}
          className="rounded-sm border border-hairline px-2 py-1 font-mono text-[11px] text-steel hover:text-steel2"
        >
          {locale.toUpperCase()}
        </button>

        <button
          onClick={toggleTheme}
          className="rounded-sm border border-hairline px-2 py-1 text-[11px] text-steel hover:text-steel2"
        >
          {theme === "dark" ? "☾" : "☀"}
        </button>
      </div>
    </header>
  );
}
