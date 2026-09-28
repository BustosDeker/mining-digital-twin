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
    <header className="flex h-16 shrink-0 items-center justify-between border-b border-hairline bg-panel px-6">
      <div className="flex items-center gap-8">
        <h1 className="text-[16px] font-semibold uppercase tracking-wide text-steel2">
          {t.appTitle}
        </h1>
        <nav className="flex items-center gap-2">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => onTabChange(tab.id)}
              className={clsx(
                "rounded-sm px-5 py-2.5 text-[14px] font-medium transition-colors",
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

      <div className="flex items-center gap-6">
        {connectionLabel && (
          <div className="flex items-center gap-2 font-mono text-[13px] text-steel">
            <span
              className={clsx(
                "h-2 w-2 rounded-full",
                connected ? "bg-signal animate-pulse-signal" : "bg-amber"
              )}
            />
            {connectionLabel}
          </div>
        )}

        <button
          onClick={() => setLocale(locale === "es" ? "en" : "es")}
          className="rounded-sm border border-hairline px-4 py-2 font-mono text-[13px] text-steel hover:text-steel2"
        >
          {locale.toUpperCase()}
        </button>

        <button
          onClick={toggleTheme}
          className="rounded-sm border border-hairline px-4 py-2 text-[13px] text-steel hover:text-steel2"
        >
          {theme === "dark" ? "☾" : "☀"}
        </button>
      </div>
    </header>
  );
}
