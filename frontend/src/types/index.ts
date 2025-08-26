// Shared types for use throughout the application

export interface Team {
  id: number;
  name: string;
  abbrev: string;
  conference: "East" | "West";
  primaryColor: string;
  secondaryColor: string;
}

export interface BracketSlot {
  id: string;
  team: Team | null;
  conference?: "East" | "West";
  season?: string; // Season year (e.g., "1996-1997")
  seed?: number;   // Team seed (1-8)
}

export interface BracketData {
  round1: BracketSlot[];
  round2: BracketSlot[];
  round3: BracketSlot[];
  round4: BracketSlot[];
  champion: BracketSlot[];
}

export type RoundKey = "round1" | "round2" | "round3" | "round4" | "champion";

export type ConferenceType = "East" | "West" | "Finals" | "Champion";

// New types for seed tracking
export type SeedMap = {[key: number]: boolean};
export type ConferenceSeedMap = {
  East: SeedMap;
  West: SeedMap;
};

// New type for matchup tracking
export interface MatchupPair {
  slotA: string;
  slotB: string;
}
