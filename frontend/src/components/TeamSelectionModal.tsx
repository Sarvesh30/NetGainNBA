import React from "react";
import { Team } from "../types";
import "./TeamSelectionModal.css";

interface TeamSelectionModalProps {
  teams: Team[];
  onSelect: (teamId: number) => void;
  onClose: () => void;
}

const TeamSelectionModal: React.FC<TeamSelectionModalProps> = ({
  teams,
  onSelect,
  onClose,
}) => {
  // Split teams by conference
  const eastTeams = teams.filter((team) => team.conference === "East");
  const westTeams = teams.filter((team) => team.conference === "West");

  // Handle background click to dismiss
  const handleBackgroundClick = (e: React.MouseEvent) => {
    if (e.target === e.currentTarget) {
      onClose();
    }
  };

  // Get only the last word of the team name
  const getShortTeamName = (teamName: string) => {
    return teamName.split(" ").pop() || teamName;
  };

  // Get the correct logo filename based on team name
  const getLogoPath = (teamName: string) => {
    // Convert to the format like "jazz logo.png"
    const shortName = teamName.split(" ").pop()?.toLowerCase() || ""; // Use last word of team name
    try {
      // For dynamic imports in Vite, we need to use this format
      return new URL(`../logos/${shortName} logo.png`, import.meta.url).href;
    } catch (error) {
      console.error(`Could not load logo for ${teamName}:`, error);
      return ""; // Return empty string if logo can't be found
    }
  };

  return (
    <div
      className="team-selection-modal-backdrop"
      onClick={handleBackgroundClick}
    >
      <div className="team-selection-modal">
        <div className="modal-header">
          <h3>Select a Team</h3>
          <button className="close-button" onClick={onClose}>
            &times;
          </button>
        </div>

        <div className="modal-body">
          {teams.length === 0 ? (
            <p className="no-teams-message">No teams available</p>
          ) : (
            <>
              {eastTeams.length > 0 && (
                <div className="conference-section">
                  <h4 className="east-conference">Eastern Conference</h4>
                  <div className="team-grid">
                    {eastTeams.map((team) => (
                      <div
                        key={team.id}
                        className="team-select-card"
                        style={{
                          backgroundColor: team.primaryColor,
                          color: team.secondaryColor,
                        }}
                        onClick={() => onSelect(team.id)}
                      >
                        <div className="team-select-card-content">
                          <span>{getShortTeamName(team.name)}</span>
                          <img
                            src={getLogoPath(team.name)}
                            alt={`${team.name} logo`}
                            className="team-select-logo"
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {westTeams.length > 0 && (
                <div className="conference-section">
                  <h4 className="west-conference">Western Conference</h4>
                  <div className="team-grid">
                    {westTeams.map((team) => (
                      <div
                        key={team.id}
                        className="team-select-card"
                        style={{
                          backgroundColor: team.primaryColor,
                          color: team.secondaryColor,
                        }}
                        onClick={() => onSelect(team.id)}
                      >
                        <div className="team-select-card-content">
                          <span>{getShortTeamName(team.name)}</span>
                          <img
                            src={getLogoPath(team.name)}
                            alt={`${team.name} logo`}
                            className="team-select-logo"
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </>
          )}
        </div>

        <div className="modal-footer">
          <button className="cancel-button" onClick={onClose}>
            Cancel
          </button>
        </div>
      </div>
    </div>
  );
};

export default TeamSelectionModal;
