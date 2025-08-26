import React, { useRef, useEffect, useState } from "react";
import { useDrop } from "react-dnd";
import TeamCard from "./TeamCard";
import { BracketSlot as BracketSlotType, ConferenceType } from "../types";
import "./BracketSlot.css";

interface BracketSlotProps {
  slot: BracketSlotType;
  roundKey: string;
  onDrop: (slotId: string, roundKey: string, teamId: number | null) => void;
  onSlotClick?: (slotId: string, roundKey: string) => void;
  conference: ConferenceType;
  isMobile?: boolean;
}

const BracketSlot: React.FC<BracketSlotProps> = ({
  slot,
  roundKey,
  onDrop,
  onSlotClick,
  conference,
  isMobile = false,
}) => {
  const slotRef = useRef<HTMLDivElement>(null);
  const isFirstRound = roundKey === "round1";
  const [isSmallScreen, setIsSmallScreen] = useState<boolean>(
    window.innerWidth <= 900
  );
  const [isTouching, setIsTouching] = useState<boolean>(false);

  // Update screen size state
  useEffect(() => {
    const handleResize = () => {
      setIsSmallScreen(window.innerWidth <= 900);
    };

    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  // Only enable drop for first round or to remove teams from any round
  const [{ isOver }, drop] = useDrop({
    accept: "team",
    canDrop: () => isFirstRound, // Only allow dropping in first round
    drop: (item: { id: number }) => {
      onDrop(slot.id, roundKey, item.id);
      // Simple trigger for text overflow check after drop
      setTimeout(() => {
        window.dispatchEvent(new Event("resize"));
      }, 100);
      return undefined;
    },
    collect: (monitor) => ({
      isOver: !!monitor.isOver(),
    }),
  });

  // Connect the drop ref to our div using useEffect
  useEffect(() => {
    if (slotRef.current) {
      drop(slotRef.current);
    }
  }, [drop]);

  // Handle removing a team from the slot
  const handleRemoveTeam = (e?: React.MouseEvent) => {
    if (e) {
      e.stopPropagation(); // Prevent event bubbling
    }

    if (slot.team) {
      // Pass null as teamId to indicate removal
      onDrop(slot.id, roundKey, null);
    }
  };

  // Handle slot click for team selection or removal
  const handleSlotClick = () => {
    if (isFirstRound) {
      if (slot.team) {
        // If there's already a team, remove it (works on both mobile and desktop)
        handleRemoveTeam();
      } else if (isMobile && onSlotClick) {
        // Only show team selection modal on mobile
        onSlotClick(slot.id, roundKey);
      }
    }
  };

  // Trigger resizing when slot content changes
  useEffect(() => {
    // Simple trigger for text overflow check after team changes
    if (slot.team) {
      setTimeout(() => {
        window.dispatchEvent(new Event("resize"));
      }, 100);
    }
  }, [slot.team]);

  // Touch event handlers for better mobile experience
  const handleTouchStart = () => {
    setIsTouching(true);
  };

  const handleTouchEnd = () => {
    setIsTouching(false);
  };

  // Render the bracket slot
  return (
    <div
      className={`team-slot-container ${isFirstRound ? "droppable-slot" : ""} ${
        isMobile ? "mobile-slot" : ""
      }`}
    >
      <div
        ref={slotRef}
        className={`bracket-slot ${isOver ? "is-over" : ""} ${
          isTouching ? "is-touching" : ""
        } ${!slot.team && isFirstRound && isMobile ? "mobile-select" : ""}`}
        onClick={handleSlotClick}
        onTouchStart={handleTouchStart}
        onTouchEnd={handleTouchEnd}
      >
        {slot.team ? (
          <div className="team-with-metadata">
            <TeamCard team={slot.team} />
            {isFirstRound && slot.seed && (
              <div className="seed-indicator">
                <span className="seed-number">{slot.seed}</span>
                {slot.season && (
                  <span className="season-year">
                    {slot.season.substring(0, 4)}
                  </span>
                )}
              </div>
            )}
          </div>
        ) : (
          <div className="empty-slot">
            {isFirstRound
              ? isMobile
                ? "Tap to Add Team"
                : "Drop Team Here"
              : ""}
          </div>
        )}
      </div>

      {slot.team && (
        <div
          className={`remove-hint ${isTouching ? "active" : ""}`}
          onClick={handleRemoveTeam}
        >
          {isMobile ? "Tap to Remove" : "Click to Remove"}
        </div>
      )}
    </div>
  );
};

export default BracketSlot;
