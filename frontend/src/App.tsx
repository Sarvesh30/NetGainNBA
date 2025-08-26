import { useState, useEffect } from "react";
import { DndProvider } from "react-dnd";
import { HTML5Backend } from "react-dnd-html5-backend";
import { SignInButton, SignOutButton, useUser } from "@clerk/clerk-react";
import BracketTree from "./components/BracketTree";
import TeamList from "./components/TeamList";
import MLConfig from "./components/MLConfig";
import ThemeToggle from "./components/ThemeToggle";
import TeamSelectionModal from "./components/TeamSelectionModal";
import SeasonSeedModal from "./components/SeasonSeedModal";
import { ThemeProvider } from "./context/ThemeContext";
import {
  BracketData,
  BracketSlot,
  RoundKey,
  ConferenceSeedMap,
  MatchupPair,
} from "./types";
import { nbaTeams } from "./data/teams";
import "./App.css";

const API_BASE_URL = "http://localhost:8080"; // change when deploying

// Create initial empty bracket structure
const createEmptyBracket = (): BracketData => {
  const bracketData: BracketData = {
    round1: [],
    round2: [],
    round3: [],
    round4: [],
    champion: [{ id: "champion", team: null }],
  };

  // First round - 16 slots (8 East, 8 West)
  for (let i = 0; i < 16; i++) {
    const conference = i < 8 ? "East" : "West";
    bracketData.round1.push({
      id: `r1m${i + 1}`,
      team: null,
      conference: conference,
    });
  }

  // Second round - 8 slots (4 East, 4 West)
  for (let i = 0; i < 8; i++) {
    const conference = i < 4 ? "East" : "West";
    bracketData.round2.push({
      id: `r2m${i + 1}`,
      team: null,
      conference: conference,
    });
  }

  // Conference Finals - 4 slots (2 East, 2 West)
  for (let i = 0; i < 4; i++) {
    const conference = i < 2 ? "East" : "West";
    bracketData.round3.push({
      id: `r3m${i + 1}`,
      team: null,
      conference: conference,
    });
  }

  // Finals - 2 slots (1 East, 1 West)
  for (let i = 0; i < 2; i++) {
    const conference = i === 0 ? "East" : "West";
    bracketData.round4.push({
      id: `r4m${i + 1}`,
      team: null,
      conference: conference,
    });
  }

  return bracketData;
};

// Define bracket slot matchups (which slots compete against each other)
const createBracketMatchups = (): MatchupPair[] => {
  const matchups: MatchupPair[] = [];
  // Round 1 East matchups (8 teams, 4 pairs)
  matchups.push({ slotA: "r1m1", slotB: "r1m2" });
  matchups.push({ slotA: "r1m3", slotB: "r1m4" });
  matchups.push({ slotA: "r1m5", slotB: "r1m6" });
  matchups.push({ slotA: "r1m7", slotB: "r1m8" });
  // Round 1 West matchups (8 teams, 4 pairs)
  matchups.push({ slotA: "r1m9", slotB: "r1m10" });
  matchups.push({ slotA: "r1m11", slotB: "r1m12" });
  matchups.push({ slotA: "r1m13", slotB: "r1m14" });
  matchups.push({ slotA: "r1m15", slotB: "r1m16" });
  return matchups;
};

// Helper function to create initial seed map
const createInitialSeedMap = (): ConferenceSeedMap => {
  return {
    East: {
      1: true,
      2: true,
      3: true,
      4: true,
      5: true,
      6: true,
      7: true,
      8: true,
    },
    West: {
      1: true,
      2: true,
      3: true,
      4: true,
      5: true,
      6: true,
      7: true,
      8: true,
    },
  };
};

// Function to check if bracket is fully filled (only round1 matters for simulation)
const isBracketFilled = (bracket: BracketData): boolean => {
  return bracket.round1.every(
    (slot) => slot.team !== null && slot.seed !== undefined
  );
};

function App() {
  const { isSignedIn, user } = useUser();
  const [bracket, setBracket] = useState<BracketData>(createEmptyBracket());
  const [availableTeams, setAvailableTeams] = useState([...nbaTeams]);
  const [activeTab, setActiveTab] = useState<"bracket" | "config">("bracket");
  const [isMobile, setIsMobile] = useState<boolean>(window.innerWidth < 768);

  // Team selection modal state
  const [showTeamModal, setShowTeamModal] = useState<boolean>(false);
  const [targetSlot, setTargetSlot] = useState<{
    slotId: string;
    roundKey: string;
  } | null>(null);

  // Season/Seed modal state
  const [showSeasonSeedModal, setShowSeasonSeedModal] =
    useState<boolean>(false);
  const [seasonSeedTarget, setSeasonSeedTarget] = useState<{
    slotId: string;
    teamId: number;
    conference: "East" | "West";
    opposingSlotId?: string;
    opposingSeed?: number;
  } | null>(null);

  // Track available seeds by conference
  const [availableSeeds, setAvailableSeeds] = useState<ConferenceSeedMap>(
    createInitialSeedMap()
  );

  // Re-sync available seeds whenever bracket.round1 changes.
  useEffect(() => {
    const newSeedState: ConferenceSeedMap = createInitialSeedMap();
    bracket.round1.forEach((slot) => {
      if (slot.seed && slot.conference) {
        newSeedState[slot.conference][slot.seed] = false;
      }
    });
    setAvailableSeeds(newSeedState);
  }, [bracket.round1]);

  // Store matchup pairs
  const [matchupPairs] = useState<MatchupPair[]>(createBracketMatchups());

  // Custom alert modal state for simulation
  const [showSimAlert, setShowSimAlert] = useState<boolean>(false);
  const [simAlertMessage, setSimAlertMessage] = useState<string>("");

  // Detect mobile devices
  useEffect(() => {
    const handleResize = () => {
      setIsMobile(window.innerWidth < 768);
    };
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  // Helper function to trigger a resize event for abbreviation checks
  const triggerResizeForAbbreviations = () => {
    window.dispatchEvent(new CustomEvent("team-update"));
    window.dispatchEvent(new Event("resize"));
  };

  // Find opposing slot ID for a given slot
  const findOpposingSlotId = (slotId: string): string | undefined => {
    const matchup = matchupPairs.find(
      (pair) => pair.slotA === slotId || pair.slotB === slotId
    );
    if (matchup) {
      return matchup.slotA === slotId ? matchup.slotB : matchup.slotA;
    }
    return undefined;
  };

  // Find opposing slot's seed for a given slot
  const findOpposingSeed = (slotId: string): number | undefined => {
    const opposingSlotId = findOpposingSlotId(slotId);
    if (!opposingSlotId) return undefined;
    const slotWithSeed = bracket.round1.find(
      (slot) => slot.id === opposingSlotId && slot.team && slot.seed
    );
    return slotWithSeed?.seed;
  };

  // Update seed for a slot and handle complementary seeding.
  // Note: Available seeds are now updated solely via useEffect.
  const updateSeedForSlot = (slotId: string, newSeed: number) => {
    const updatedBracket = { ...bracket };
    const slotIndex = updatedBracket.round1.findIndex(
      (slot) => slot.id === slotId
    );
    if (slotIndex === -1) return;

    // Update this slot's seed.
    updatedBracket.round1[slotIndex].seed = newSeed;

    // For complementary seeding on the opposing slot:
    const opposingSlotId = findOpposingSlotId(slotId);
    let complementarySeed = 9 - newSeed;
    if (opposingSlotId) {
      const opposingIndex = updatedBracket.round1.findIndex(
        (slot) => slot.id === opposingSlotId
      );
      if (opposingIndex !== -1 && updatedBracket.round1[opposingIndex].team) {
        // Set the complementary seed for the opposing slot while preserving its season.
        updatedBracket.round1[opposingIndex].seed = complementarySeed;
      }
    }

    setBracket(updatedBracket);
  };

  // Handle season and seed confirmation
  const handleSeasonSeedConfirm = (season: string, seed: number) => {
    if (!seasonSeedTarget) return;
    const { slotId, conference } = seasonSeedTarget;
    const updatedBracket = { ...bracket };
    const slotIndex = updatedBracket.round1.findIndex(
      (slot) => slot.id === slotId
    );
    if (slotIndex === -1) {
      setShowSeasonSeedModal(false);
      setSeasonSeedTarget(null);
      return;
    }
    // Update season and then update the seed with complementary logic.
    updatedBracket.round1[slotIndex].season = season;
    updateSeedForSlot(slotId, conference, seed);
    updateSeedForSlot(slotId, seed);
    setSeasonSeedTarget(null);
  };

  // Handle dropping a team into a bracket slot
  const handleDrop = (
    slotId: string,
    roundKey: string,
    teamId: number | null
  ) => {
    // Only allow dropping in round1 (or removal in any round)
    if (roundKey !== "round1" && teamId !== null) {
      console.log("Teams can only be placed in the first round!");
      return;
    }

    const slotIndex = bracket[roundKey as RoundKey].findIndex(
      (slot: BracketSlot) => slot.id === slotId
    );

    // Removing a team from a slot
    if (teamId === null && slotIndex !== -1) {
      const updatedBracket = { ...bracket };
      const currentTeam = updatedBracket[roundKey as RoundKey][slotIndex].team;
      if (currentTeam) {
        // Remove team and clear seed & season only from the current slot.
        updatedBracket[roundKey as RoundKey][slotIndex].team = null;
        updatedBracket[roundKey as RoundKey][slotIndex].seed = undefined;
        updatedBracket[roundKey as RoundKey][slotIndex].season = undefined;
        // Add the removed team back to available teams.
        setAvailableTeams((prev) => [...prev, currentTeam]);
        setBracket(updatedBracket);
      }
      return;
    }

    // If teamId is not null, handle adding or moving a team.
    if (teamId === null) return; // precaution

    // Check if the team exists in availableTeams
    const teamIndex = availableTeams.findIndex((team) => team.id === teamId);
    if (teamIndex === -1) {
      // The team is already in the bracket; find and remove it from its current slot.
      let existingRound = "";
      let existingSlotIndex = -1;
      Object.entries(bracket).forEach(([round, slots]) => {
        const idx = slots.findIndex(
          (slot: BracketSlot) => slot.team && slot.team.id === teamId
        );
        if (idx !== -1) {
          existingRound = round;
          existingSlotIndex = idx;
        }
      });
      if (existingRound && existingSlotIndex !== -1) {
        const updatedBracket = { ...bracket };
        const prevTeam =
          updatedBracket[existingRound as RoundKey][existingSlotIndex].team;
        if (existingRound === "round1") {
          // Remove seed from old slot; availableSeeds is re-synced via useEffect.
          updatedBracket[existingRound as RoundKey][existingSlotIndex].seed =
            undefined;
          updatedBracket[existingRound as RoundKey][existingSlotIndex].season =
            undefined;
        }
        updatedBracket[existingRound as RoundKey][existingSlotIndex].team =
          null;
        setAvailableTeams((prev) => [...prev, prevTeam!]);
        // Now, proceed to add the team into the new slot
        const newSlotIndex = bracket[roundKey as RoundKey].findIndex(
          (slot: BracketSlot) => slot.id === slotId
        );
        if (newSlotIndex !== -1) {
          // Temporarily store team to be moved
          const teamToMove = prevTeam;
          // Clear new slot if already occupied
          if (updatedBracket[roundKey as RoundKey][newSlotIndex].team) {
            const currentTeam =
              updatedBracket[roundKey as RoundKey][newSlotIndex].team;
            if (roundKey === "round1") {
              updatedBracket[roundKey as RoundKey][newSlotIndex].seed =
                undefined;
              updatedBracket[roundKey as RoundKey][newSlotIndex].season =
                undefined;
            }
            setAvailableTeams((prev) => [...prev, currentTeam]);
          }
          updatedBracket[roundKey as RoundKey][newSlotIndex].team = teamToMove;
          // For round1, open season/seed modal
          if (roundKey === "round1" && teamToMove) {
            const conference = updatedBracket.round1[newSlotIndex]
              .conference as "East" | "West";
            const opposingSlotId = findOpposingSlotId(slotId);
            const opposingSeed = opposingSlotId
              ? findOpposingSeed(opposingSlotId)
              : undefined;
            // Clear seed and season temporarily
            updatedBracket.round1[newSlotIndex].seed = undefined;
            updatedBracket.round1[newSlotIndex].season = undefined;
            setBracket(updatedBracket);
            setSeasonSeedTarget({
              slotId,
              teamId: teamToMove.id,
              conference,
              opposingSlotId,
              opposingSeed,
            });
            setShowSeasonSeedModal(true);
            triggerResizeForAbbreviations();
            return;
          }
          setBracket(updatedBracket);
        }
      }
    } else {
      // Team is in availableTeams; add it to the target slot.
      const team = availableTeams[teamIndex];
      const updatedBracket = { ...bracket };
      const slotIndex = bracket[roundKey as RoundKey].findIndex(
        (slot: BracketSlot) => slot.id === slotId
      );
      if (slotIndex !== -1) {
        // If there's an existing team in that slot, return it to availableTeams.
        const currentTeam =
          updatedBracket[roundKey as RoundKey][slotIndex].team;
        updatedBracket[roundKey as RoundKey][slotIndex].team = team;
        if (currentTeam) {
          if (roundKey === "round1") {
            updatedBracket[roundKey as RoundKey][slotIndex].seed = undefined;
            updatedBracket[roundKey as RoundKey][slotIndex].season = undefined;
          }
          setAvailableTeams((prev) => [
            ...prev.filter((t) => t.id !== team.id),
            currentTeam,
          ]);
        } else {
          setAvailableTeams((prev) => prev.filter((t) => t.id !== team.id));
        }
        // For round1, open the season/seed modal
        if (roundKey === "round1") {
          setBracket(updatedBracket);
          const conference = updatedBracket.round1[slotIndex].conference as
            | "East"
            | "West";
          const opposingSlotId = findOpposingSlotId(slotId);
          const opposingSeed = opposingSlotId
            ? findOpposingSeed(opposingSlotId)
            : undefined;
          setSeasonSeedTarget({
            slotId,
            teamId: team.id,
            conference,
            opposingSlotId,
            opposingSeed,
          });
          setShowSeasonSeedModal(true);
          triggerResizeForAbbreviations();
          return;
        }
        setBracket(updatedBracket);
      }
    }
    // Trigger a resize event after state updates.
    triggerResizeForAbbreviations();
  };

  // Function to reset the bracket
  const resetBracket = () => {
    setBracket(createEmptyBracket());
    setAvailableTeams([...nbaTeams]);
    setAvailableSeeds(createInitialSeedMap());
    triggerResizeForAbbreviations();
  };

  // Function to run ML simulation
  const runSimulation = async (runMultiple = false) => {
    console.log(`Running ML simulation... (Multiple: ${runMultiple})`);
    let message = "Running Simulation";
    if (isSignedIn && user) {
      message = `Running ${
        user.firstName || user.username || "your"
      }'s Simulation`;
    }

    if (runMultiple) {
      message += " (1000 simulations)";
    }

    setSimAlertMessage(message);
    setShowSimAlert(true);

    try {
      // Round 1 matchups - always collect matchup stats first
      for (let i = 0; i < bracket.round1.length; i += 2) {
        if (bracket.round1[i].team && bracket.round1[i + 1].team) {
          const teamA = bracket.round1[i].team;
          const teamB = bracket.round1[i + 1].team;
          const teamAseason = bracket.round1[i].season || "2022-2023";
          const teamBseason = bracket.round1[i + 1].season || "2022-2023";
          if (teamA && teamB)
            // incase of null
            await fetchMatchupStats(
              teamA.id,
              teamAseason,
              teamB.id,
              teamBseason
            );
        }
      }
      console.log("All matchup stats collected");

      if (runMultiple) {
        // Now call simulate_multiple with simulate_1000=true
        const response = await fetch(`${API_BASE_URL}/simulate_multiple`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            simulate_1000: true,
          }),
        });

        const data = await response.json();

        if (data.status === "success") {
          setSimAlertMessage(
            `Successfully completed 1000 simulations! ${data.message}`
          );
        } else {
          setSimAlertMessage(`Error: ${data.message}`);
        }
        return;
      }

      // Original simulation code for single simulation
      // After all matchup stats are collected, call /determine_winners:
      const winnersResponse = await fetch(`${API_BASE_URL}/determine_winners`);
      if (!winnersResponse.ok) {
        console.error("Failed to determine winners:", winnersResponse.status);
      } else {
        const winnersData = await winnersResponse.json();
        console.log("Determined Winners:", winnersData);
        // can update state or display the winner info here if needed
      }
    } catch (error) {
      console.error("Error in simulation:", error);
      setSimAlertMessage(`Error: ${error.message}`);
    }
  };

  // Function to fetch matchup stats from backend
  const fetchMatchupStats = async (
    teamAId: number,
    teamASeason: string,
    teamBId: number,
    teamBSeason: string
  ) => {
    try {
      const response = await fetch(`${API_BASE_URL}/get_matchup_stats`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          team_a_id: teamAId.toString(),
          team_a_szn: teamASeason,
          team_b_id: teamBId.toString(),
          team_b_szn: teamBSeason,
        }),
      });
      if (!response.ok)
        throw new Error(`HTTP error! Status: ${response.status}`);
      const data = await response.json();
      console.log(
        `Matchup stats collected for teams ${teamAId} and ${teamBId}:`,
        data
      );
      return data;
    } catch (error) {
      console.error("Error fetching matchup stats:", error);
      throw error;
    }
  };

  // Function to close simulation alert and handle post-alert actions
  const closeSimAlert = () => {
    setShowSimAlert(false);
    if (isMobile) {
      setActiveTab("bracket");
    }
  };

  // New function to handle opening team selection modal on mobile
  const handleSlotClick = (slotId: string, roundKey: string) => {
    if (isMobile && roundKey === "round1") {
      setTargetSlot({ slotId, roundKey });
      setShowTeamModal(true);
    }
  };

  // Handle team selection from modal
  const handleTeamSelect = (teamId: number) => {
    if (targetSlot) {
      handleDrop(targetSlot.slotId, targetSlot.roundKey, teamId);
      setShowTeamModal(false);
      setTargetSlot(null);
      triggerResizeForAbbreviations();
    }
  };

  // Configure drag-and-drop options for mobile support
  const dndOptions = { enableTouchEvents: true, enableMouseEvents: true };

  return (
    <ThemeProvider>
      <DndProvider backend={HTML5Backend} options={dndOptions}>
        <div className="app-container">
          <header>
            <div className="header-left">
              <ThemeToggle />
            </div>
            <h1>NBA Playoff Bracket Simulator</h1>
            <div className="header-right">
              {isSignedIn ? (
                <SignOutButton>
                  <button className="login-button">Sign Out</button>
                </SignOutButton>
              ) : (
                <SignInButton mode="modal">
                  <button className="login-button">Log In</button>
                </SignInButton>
              )}
            </div>
          </header>

          {isMobile && (
            <div className="mobile-tabs">
              <button
                className={`tab-button ${
                  activeTab === "bracket" ? "active" : ""
                }`}
                onClick={() => setActiveTab("bracket")}
              >
                Bracket
              </button>
              <button
                className={`tab-button ${
                  activeTab === "config" ? "active" : ""
                }`}
                onClick={() => setActiveTab("config")}
              >
                Simulate
              </button>
            </div>
          )}

          <div className={`main-content ${isMobile ? "mobile-layout" : ""}`}>
            {!isMobile && (
              <div className="panel">
                <TeamList teams={availableTeams} onReset={resetBracket} />
              </div>
            )}

            <div
              className={`panel ${
                isMobile ? (activeTab === "bracket" ? "visible" : "hidden") : ""
              }`}
            >
              <div className="bracket-panel-wrapper">
                <BracketTree
                  bracket={bracket}
                  onDrop={handleDrop}
                  isMobile={isMobile}
                  onSlotClick={handleSlotClick}
                  onReset={resetBracket}
                />
                {isMobile && activeTab === "bracket" && (
                  <div className="guaranteed-reset">
                    <button
                      className="guaranteed-reset-button"
                      onClick={resetBracket}
                    >
                      Reset Bracket
                    </button>
                  </div>
                )}
              </div>
            </div>

            {!isMobile && (
              <div className="panel">
                <MLConfig
                  onRunSimulation={runSimulation}
                  isDisabled={!isBracketFilled(bracket)}
                />
              </div>
            )}

            {isMobile && (
              <div
                className={`panel ${
                  activeTab === "config" ? "visible" : "hidden"
                }`}
              >
                <MLConfig
                  onRunSimulation={runSimulation}
                  isDisabled={!isBracketFilled(bracket)}
                />
              </div>
            )}

            {isMobile && activeTab === "config" && (
              <div className="mobile-actions">
                <button
                  className="mobile-action-button simulation-button"
                  onClick={() => runSimulation(false)}
                  disabled={!isBracketFilled(bracket)}
                >
                  {!isBracketFilled(bracket)
                    ? "Complete Bracket First"
                    : "Run Simulation"}
                </button>
                <button
                  className="mobile-action-button simulation-button"
                  onClick={() => runSimulation(true)}
                  disabled={!isBracketFilled(bracket)}
                >
                  {!isBracketFilled(bracket)
                    ? "Complete Bracket First"
                    : "Run 1000 Simulations"}
                </button>
              </div>
            )}
          </div>

          {showSimAlert && (
            <div className="modal-overlay">
              <div className="modal-content alert-modal">
                <p>{simAlertMessage}</p>
                <button onClick={closeSimAlert}>OK</button>
              </div>
            </div>
          )}

          {showTeamModal && (
            <TeamSelectionModal
              teams={availableTeams}
              onSelect={handleTeamSelect}
              onClose={() => setShowTeamModal(false)}
            />
          )}

          {showSeasonSeedModal && seasonSeedTarget && (
            <SeasonSeedModal
              onConfirm={handleSeasonSeedConfirm}
              onClose={() => {
                // If the user cancels seed selection, remove the team that was added.
                if (seasonSeedTarget.slotId) {
                  const slotIndex = bracket.round1.findIndex(
                    (slot) => slot.id === seasonSeedTarget.slotId
                  );
                  if (slotIndex !== -1 && bracket.round1[slotIndex].team) {
                    const updatedBracket = { ...bracket };
                    const teamToReturn = updatedBracket.round1[slotIndex].team;
                    updatedBracket.round1[slotIndex].team = null;
                    updatedBracket.round1[slotIndex].seed = undefined;
                    updatedBracket.round1[slotIndex].season = undefined;
                    if (teamToReturn) {
                      setAvailableTeams((prev) => [...prev, teamToReturn]);
                    }
                    setBracket(updatedBracket);
                  }
                }
                setShowSeasonSeedModal(false);
                setSeasonSeedTarget(null);
              }}
              conference={seasonSeedTarget.conference}
              availableSeeds={availableSeeds[seasonSeedTarget.conference]}
              opposingSeed={seasonSeedTarget.opposingSeed}
            />
          )}
        </div>
      </DndProvider>
    </ThemeProvider>
  );
}

export default App;
