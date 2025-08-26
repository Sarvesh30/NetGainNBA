import React, { useState, useEffect } from "react";
import "./SeasonSeedModal.css";
import { SeedMap } from "../types";

interface SeasonSeedModalProps {
  onConfirm: (season: string, seed: number) => void;
  onClose: () => void;
  conference: "East" | "West";
  availableSeeds: SeedMap;
  initialSeason?: string;
  initialSeed?: number;
  opposingSeed?: number;
}

const SeasonSeedModal: React.FC<SeasonSeedModalProps> = ({
  onConfirm,
  onClose,
  conference,
  availableSeeds,
  initialSeason,
  initialSeed,
  opposingSeed,
}) => {
  //const currentYear = new Date().getFullYear();
  const currentYear = new Date().getFullYear(); // set to current year
  const startYear = 1996;
  const endYear = currentYear;

  const [season, setSeason] = useState<string>(
    initialSeason || `${currentYear - 1}-${currentYear}`
  );
  const [seed, setSeed] = useState<number | null>(initialSeed || null);

  // Generate array of season years
  const seasons = [];
  for (let year = startYear; year <= endYear; year++) {
    seasons.push(`${year}-${year + 1}`);
  }

  // Determine which seeds are available
  const seedOptions = [];
  for (let i = 1; i <= 8; i++) {
    // A seed is available if:
    // 1. It's not already used (availableSeeds[i] is true or undefined)
    // 2. OR it's the initially selected seed
    // 3. OR if opposingSeed exists, it's the complementary seed to opposingSeed (9-opposingSeed)
    if (
      availableSeeds[i] !== false ||
      i === initialSeed ||
      (opposingSeed && i === 9 - opposingSeed)
    ) {
      seedOptions.push(i);
    }
  }

  // Set default seed if none selected and only one option
  useEffect(() => {
    if (seed === null && seedOptions.length === 1) {
      setSeed(seedOptions[0]);
    }

    // If opposing seed exists and no seed selected, set complementary seed
    if (seed === null && opposingSeed) {
      setSeed(9 - opposingSeed);
    }
  }, [seed, seedOptions, opposingSeed]);

  const handleConfirm = () => {
    if (season && seed) {
      onConfirm(season, seed);
    }
  };

  return (
    <div className="modal-overlay">
      <div className="modal-content season-seed-modal">
        <h2>{conference} Conference Team</h2>

        <div className="modal-field">
          <label htmlFor="season">Season:</label>
          <select
            id="season"
            value={season}
            onChange={(e) => setSeason(e.target.value)}
          >
            {seasons.reverse().map((s) => (
              <option key={s} value={s}>
                {s}
              </option>
            ))}
          </select>
        </div>

        <div className="modal-field">
          <label htmlFor="seed">Seed:</label>
          <select
            id="seed"
            value={seed || ""}
            onChange={(e) => setSeed(Number(e.target.value))}
          >
            <option value="" disabled>
              Select Seed
            </option>
            {seedOptions.map((s) => (
              <option key={s} value={s}>
                {s}
              </option>
            ))}
          </select>
          {opposingSeed && (
            <div className="seed-info">
              Opposing team has seed {opposingSeed}
              {seed && (
                <span>
                  {" "}
                  (Matchup: {Math.min(seed, opposingSeed)} vs{" "}
                  {Math.max(seed, opposingSeed)})
                </span>
              )}
            </div>
          )}
        </div>

        <div className="modal-buttons">
          <button onClick={onClose} className="cancel-button">
            Cancel
          </button>
          <button
            onClick={handleConfirm}
            className="confirm-button green-button"
            disabled={!season || !seed}
          >
            Confirm
          </button>
        </div>
      </div>
    </div>
  );
};

export default SeasonSeedModal;
