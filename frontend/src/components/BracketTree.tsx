import React from "react";
import BracketSlot from "./BracketSlot";
import { BracketData, RoundKey } from "../types";
import "./BracketTree.css";

interface BracketTreeProps {
  bracket: BracketData;
  onDrop: (slotId: string, roundKey: string, teamId: number | null) => void;
  onSlotClick?: (slotId: string, roundKey: string) => void;
  onReset?: () => void;
  isMobile?: boolean;
}

const BracketTree: React.FC<BracketTreeProps> = ({
  bracket,
  onDrop,
  onSlotClick,
  onReset,
  isMobile = false,
}) => {
  // Placeholder function for save button
  const handleSave = () => {
    console.log("Save bracket functionality will be implemented later");
  };

  return (
    <>
      <div className={`bracket-container ${isMobile ? "mobile-view" : ""}`}>
        <div className="conference-headers">
          <div className="east-header">Eastern Conference</div>
          <div className="finals-header">NBA Finals</div>
          <div className="west-header">Western Conference</div>
        </div>

        <div className="bracket-rounds">
          <div className="round round-1">
            <h3>First Round</h3>
            <div className="matchups east-matchups">
              {bracket.round1.slice(0, 8).map((slot) => (
                <BracketSlot
                  key={slot.id}
                  slot={slot}
                  roundKey="round1"
                  onDrop={onDrop}
                  onSlotClick={onSlotClick}
                  conference="East"
                  isMobile={isMobile}
                />
              ))}
            </div>
          </div>

          <div className="round round-2">
            <h3>Conference Semifinals</h3>
            <div className="matchups east-matchups">
              {bracket.round2.slice(0, 4).map((slot) => (
                <BracketSlot
                  key={slot.id}
                  slot={slot}
                  roundKey="round2"
                  onDrop={onDrop}
                  onSlotClick={onSlotClick}
                  conference="East"
                  isMobile={isMobile}
                />
              ))}
            </div>
          </div>

          <div className="round round-3">
            <h3>Conference Finals</h3>
            <div className="matchups east-matchups">
              {bracket.round3.slice(0, 2).map((slot) => (
                <BracketSlot
                  key={slot.id}
                  slot={slot}
                  roundKey="round3"
                  onDrop={onDrop}
                  onSlotClick={onSlotClick}
                  conference="East"
                  isMobile={isMobile}
                />
              ))}
            </div>
          </div>

          <div className="round round-4">
            <h3>NBA Finals</h3>
            <div className="matchups finals-matchups">
              <BracketSlot
                slot={bracket.round4[0]}
                roundKey="round4"
                onDrop={onDrop}
                onSlotClick={onSlotClick}
                conference="Finals"
                isMobile={isMobile}
              />
            </div>
          </div>

          <div className="round champion-round">
            <h3>Champion</h3>
            <div className="matchups champion-matchup">
              <BracketSlot
                slot={bracket.champion[0]}
                roundKey="champion"
                onDrop={onDrop}
                onSlotClick={onSlotClick}
                conference="Champion"
                isMobile={isMobile}
              />
            </div>
          </div>

          <div className="round round-4">
            <h3>NBA Finals</h3>
            <div className="matchups finals-matchups">
              <BracketSlot
                slot={bracket.round4[1]}
                roundKey="round4"
                onDrop={onDrop}
                onSlotClick={onSlotClick}
                conference="Finals"
                isMobile={isMobile}
              />
            </div>
          </div>

          <div className="round round-3">
            <h3>Conference Finals</h3>
            <div className="matchups west-matchups">
              {bracket.round3.slice(2, 4).map((slot) => (
                <BracketSlot
                  key={slot.id}
                  slot={slot}
                  roundKey="round3"
                  onDrop={onDrop}
                  onSlotClick={onSlotClick}
                  conference="West"
                  isMobile={isMobile}
                />
              ))}
            </div>
          </div>

          <div className="round round-2">
            <h3>Conference Semifinals</h3>
            <div className="matchups west-matchups">
              {bracket.round2.slice(4, 8).map((slot) => (
                <BracketSlot
                  key={slot.id}
                  slot={slot}
                  roundKey="round2"
                  onDrop={onDrop}
                  onSlotClick={onSlotClick}
                  conference="West"
                  isMobile={isMobile}
                />
              ))}
            </div>
          </div>

          <div className="round round-1">
            <h3>First Round</h3>
            <div className="matchups west-matchups">
              {bracket.round1.slice(8, 16).map((slot) => (
                <BracketSlot
                  key={slot.id}
                  slot={slot}
                  roundKey="round1"
                  onDrop={onDrop}
                  onSlotClick={onSlotClick}
                  conference="West"
                  isMobile={isMobile}
                />
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Mobile action buttons outside the bracket container */}
      {isMobile && onReset && (
        <div className="mobile-bracket-actions">
          <button
            className="mobile-reset-button"
            onClick={onReset}
            type="button"
          >
            Reset Bracket
          </button>
          <button
            className="mobile-save-button"
            onClick={handleSave}
            type="button"
          >
            Save Bracket
          </button>
        </div>
      )}
    </>
  );
};

export default BracketTree;
