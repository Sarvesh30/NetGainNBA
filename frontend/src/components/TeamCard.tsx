import React, { useRef, useEffect, memo, useState } from "react";
import { useDrag } from "react-dnd";
import { Team } from "../types";
import "./TeamCard.css";

interface TeamCardProps {
  team: Team;
  isDragging: boolean;
}

const TeamCard: React.FC<TeamCardProps> = ({ team, isDragging }) => {
  const cardRef = useRef<HTMLDivElement>(null);
  const [textSizeClass, setTextSizeClass] = useState("text-size-md");
  const [useAbbrev, setUseAbbrev] = useState(false);

  const [{ opacity }, drag] = useDrag({
    type: "team",
    item: () => ({ id: team.id }),
    collect: (monitor) => ({
      opacity: monitor.isDragging() ? 0.4 : 1,
    }),
  });

  // Connect the drag ref for draggable cards
  useEffect(() => {
    if (isDragging && cardRef.current) {
      drag(cardRef.current);
    }
  }, [drag, isDragging]);

  // Get only the last word of the team name
  const getShortTeamName = () => {
    return team.name.split(" ").pop() || team.name;
  };

  // Simple function to determine text size based on team name length
  const updateTextSize = () => {
    const shortName = getShortTeamName();
    const nameLength = shortName.length;
    const inBracketSlot = cardRef.current?.closest(".bracket-slot") !== null;
    const screenWidth = window.innerWidth;

    // Always use abbreviations on small screens in bracket slots
    if (inBracketSlot && screenWidth <= 900) {
      setUseAbbrev(true);
      return;
    }

    // For team list, always use standard size
    if (!inBracketSlot) {
      setTextSizeClass("text-size-md");
      setUseAbbrev(false);
      return;
    }

    // For bracket slots, determine size based on name length and screen width
    if (nameLength > 12) {
      // Longer names like "Timberwolves"
      if (screenWidth <= 1400) {
        setUseAbbrev(true); // Use abbreviation on most screens
      } else {
        setTextSizeClass("text-size-xxs");
        setUseAbbrev(false);
      }
    } else if (nameLength > 9) {
      // Medium: "Mavericks", "Cavaliers"
      if (screenWidth <= 1200) {
        setUseAbbrev(true);
      } else {
        setTextSizeClass("text-size-xs");
        setUseAbbrev(false);
      }
    } else if (nameLength > 6) {
      // Short-Medium: "Celtics", "Raptors"
      if (screenWidth <= 1000) {
        setUseAbbrev(true);
      } else {
        setTextSizeClass("text-size-sm");
        setUseAbbrev(false);
      }
    } else {
      // Short: "Heat", "Bulls"
      setTextSizeClass("text-size-md");
      setUseAbbrev(false);
    }
  };

  // Update text size when component mounts or team changes
  useEffect(() => {
    updateTextSize();

    // Add resize listener to handle window size changes
    const handleResize = () => updateTextSize();
    window.addEventListener("resize", handleResize);

    // Custom event handler for team updates
    const handleTeamUpdate = () => updateTextSize();
    window.addEventListener("team-update", handleTeamUpdate);

    return () => {
      window.removeEventListener("resize", handleResize);
      window.removeEventListener("team-update", handleTeamUpdate);
    };
  }, [team.name]);

  // Get the correct logo filename based on team abbreviation
  const getLogoPath = () => {
    // Convert to the format like "jazz logo.png"
    const teamName = team.name.split(" ").pop()?.toLowerCase() || ""; // Use last word of team name
    try {
      // For dynamic imports in Vite, we need to use this format
      return new URL(`../logos/${teamName} logo.png`, import.meta.url).href;
    } catch (error) {
      console.error(`Could not load logo for ${team.name}:`, error);
      return ""; // Return empty string if logo can't be found
    }
  };

  return (
    <div
      ref={cardRef}
      className={`team-card ${useAbbrev ? "use-abbrev" : ""}`}
      style={{
        opacity,
        backgroundColor: team.primaryColor,
        color: team.secondaryColor,
        cursor: isDragging ? "grab" : "default",
      }}
      title={team.name}
    >
      {useAbbrev ? (
        <span className="team-abbrev">{team.abbrev}</span>
      ) : (
        <div className="team-name-with-logo">
          <span className={`team-full-name ${textSizeClass}`}>
            {getShortTeamName()}
          </span>
          <img
            src={getLogoPath()}
            alt={`${team.name} logo`}
            className="team-logo"
          />
        </div>
      )}
    </div>
  );
};

// Memoize to prevent unnecessary re-renders
export default memo(TeamCard);
