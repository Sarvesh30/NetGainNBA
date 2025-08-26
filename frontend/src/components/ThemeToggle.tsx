import React, { memo } from "react";
import { useTheme } from "../context/ThemeContext";
import "./ThemeToggle.css";

const ThemeToggle: React.FC = () => {
  const { theme, toggleTheme } = useTheme();

  const isLight = theme === "light";

  return (
    <button
      className="theme-toggle"
      onClick={toggleTheme}
      aria-label={isLight ? "Switch to Dark Mode" : "Switch to Light Mode"}
    >
      <span className="theme-toggle-icon">{isLight ? "🌙" : "☀️"}</span>
      <span className="theme-toggle-text">
        {isLight ? "Dark Mode" : "Light Mode"}
      </span>
    </button>
  );
};

// Memoize this component to avoid unnecessary re-renders
export default memo(ThemeToggle);
