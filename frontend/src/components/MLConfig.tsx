import React, { useState } from "react";
import "./MLConfig.css";

interface MLConfigProps {
  onRunSimulation: (runMultiple: boolean) => void;
  isDisabled?: boolean;
}

const MLConfig: React.FC<MLConfigProps> = ({
  onRunSimulation,
  isDisabled = false,
}) => {
  const [runMultipleSimulations, setRunMultipleSimulations] = useState(false);

  return (
    <div className="ml-config-container">
      <h2>ML Configuration</h2>

      <div className="config-options">
        <div className="config-option">
          <label className="checkbox-container">
            <input
              type="checkbox"
              checked={runMultipleSimulations}
              onChange={(e) => setRunMultipleSimulations(e.target.checked)}
            />
            <span className="checkbox-label">Run 1000 Simulations</span>
          </label>
        </div>
      </div>

      <div className="simulation-button-container">
        <button
          className="simulation-button"
          onClick={() => onRunSimulation(runMultipleSimulations)}
          disabled={isDisabled}
          title={
            isDisabled
              ? "Complete the bracket to run simulation"
              : "Run Simulation"
          }
        >
          {isDisabled ? "Complete Bracket First" : "Run Simulation"}
        </button>
      </div>
    </div>
  );
};

export default MLConfig;
