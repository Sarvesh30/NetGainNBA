import React from "react";
import TeamCard from "./TeamCard";
import { Team } from "../types";
import "./TeamList.css";

interface TeamListProps {
  teams: Team[];
  onReset: () => void;
}

const TeamList: React.FC<TeamListProps> = ({ teams, onReset }) => {
  const eastTeams = teams.filter((team) => team.conference === "East");
  const westTeams = teams.filter((team) => team.conference === "West");

  return (
    <div className="team-list-container">
      <h2>Available Teams</h2>

      <div className="conference-lists">
        {teams.length === 0 ? (
          <p className="no-teams">All teams have been placed in the bracket</p>
        ) : (
          <>
            <div className="conference-section">
              <h3>Eastern Conference</h3>
              <div className="team-grid">
                {eastTeams.map((team) => (
                  <TeamCard key={team.id} team={team} isDragging={true} />
                ))}
              </div>
            </div>

            <div className="conference-section">
              <h3>Western Conference</h3>
              <div className="team-grid">
                {westTeams.map((team) => (
                  <TeamCard key={team.id} team={team} isDragging={true} />
                ))}
              </div>
            </div>
          </>
        )}
      </div>

      <div className="reset-button-container">
        <button className="reset-button" onClick={onReset}>
          Reset Bracket
        </button>
      </div>
    </div>
  );
};

export default TeamList;
