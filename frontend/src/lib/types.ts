import { LocalAudioTrack, LocalVideoTrack } from "livekit-client";

export interface SessionProps {
  roomName: string;
  identity: string;
  audioTrack?: LocalAudioTrack;
  videoTrack?: LocalVideoTrack;
  region?: string;
  turnServer?: RTCIceServer;
  forceRelay?: boolean;
}

export interface TokenResult {
  identity: string;
  accessToken: string;
}

export interface TranscriptEntry {
  id: string;
  text: string;
  speaker: "user" | "agent";
  timestamp: Date;
}

export interface Fact {
  id: string;
  text: string;
  created_at: string; // ISO String
  tags: string[];
}