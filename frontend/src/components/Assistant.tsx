"use client";

import { LoadingSVG } from "@/components/button/LoadingSVG";
import { Header } from "@/components/Header";
import { Tile } from "@/components/Tile";
import { AgentMultibandAudioVisualizer } from "@/components/visualization/AgentMultibandAudioVisualizer";
import { useMultibandTrackVolume } from "@/hooks/useTrackVolume";
import { useWindowResize } from "@/hooks/useWindowResize";
import { Fact, TranscriptEntry } from "@/lib/types";
import {
  useConnectionState,
  useLocalParticipant,
  useTracks,
  useVoiceAssistant,
  useTrackTranscription,
} from "@livekit/components-react";
import { AnimatePresence, motion } from "framer-motion";
import { ConnectionState, LocalParticipant, Track } from "livekit-client";
import { ReactNode, useCallback, useEffect, useMemo, useState } from "react";
import { Button } from "./button/Button";
import { MicrophoneButton } from "./MicrophoneButton";
import { MenuSVG } from "./ui/icons";
import VoEPanel from "./VoEPanel";

export interface AssistantProps {
  title?: string;
  logo?: ReactNode;
  onConnect: (connect: boolean, opts?: { token: string; url: string }) => void;
}

const headerHeight = 56;
const mobileWindowWidth = 768;
const desktopBarWidth = 72;
const desktopMaxBarHeight = 280;
const desktopMinBarHeight = 60;
const mobileMaxBarHeight = 140;
const mobileMinBarHeight = 48;
const mobileBarWidth = 48;
const barCount = 5;
const defaultVolumes = Array.from({ length: barCount }, () => [0.0]);

export default function Assistant({ onConnect }: AssistantProps) {
  const [transcript, setTranscript] = useState<TranscriptEntry[]>([]);
  const { localParticipant } = useLocalParticipant();
  const [showVoEPanel, setShowVoEPanel] = useState(true);
  const windowSize = useWindowResize();
  const {
    agent: agentParticipant,
    state: agentState,
    audioTrack: agentAudioTrack,
    agentAttributes,
    agentTranscriptions,
  } = useVoiceAssistant();
  const [voeFacts, setVoeFacts] = useState<Fact[]>([]);
  const [isMobile, setIsMobile] = useState(false);
  const isAgentConnected = agentParticipant !== undefined;

  const roomState = useConnectionState();
  const tracks = useTracks();

  useEffect(() => {
    setShowVoEPanel(windowSize.width >= mobileWindowWidth);
    setIsMobile(windowSize.width < mobileWindowWidth);
  }, [windowSize]);

  useEffect(() => {
    if (agentParticipant?.attributes?.voe_facts) {
        try {
            const facts = JSON.parse(agentParticipant.attributes.voe_facts);
            setVoeFacts(facts);
        } catch (e) {
            console.error("Failed to parse voe_facts", e);
        }
    }
  }, [agentParticipant?.attributes]);

  useEffect(() => {
    if (roomState === ConnectionState.Connected) {
      localParticipant.setMicrophoneEnabled(true);
    }
  }, [localParticipant, roomState]);

  const subscribedVolumes = useMultibandTrackVolume(
    agentAudioTrack?.publication.track,
    barCount
  );

  const localTracks = tracks.filter(
    ({ participant }) => participant instanceof LocalParticipant
  );
  const localMicTrack = localTracks.find(
    ({ source }) => source === Track.Source.Microphone
  );

  const localMultibandVolume = useMultibandTrackVolume(
    localMicTrack?.publication.track,
    9
  );

  // Get user transcriptions from microphone track
  const { segments: userTranscriptions } = useTrackTranscription({
    participant: localParticipant,
    source: Track.Source.Microphone,
  });

  // Handle agent transcriptions
  useEffect(() => {
    if (agentTranscriptions && agentTranscriptions.length > 0) {
      const newEntries = agentTranscriptions.map((transcription, index) => ({
        id: `agent-${Date.now()}-${index}`,
        text: transcription.text,
        speaker: "agent" as const,
        timestamp: new Date(),
      }));
      setTranscript(prev => [...prev, ...newEntries]);
    }
  }, [agentTranscriptions]);

  // Handle user transcriptions
  useEffect(() => {
    if (userTranscriptions && userTranscriptions.length > 0) {
      const newEntries = userTranscriptions
        .filter(segment => segment.final) // Only show final transcriptions
        .map((transcription, index) => ({
          id: `user-${transcription.id || Date.now()}-${index}`,
          text: transcription.text,
          speaker: "user" as const,
          timestamp: new Date(),
        }));
      
      if (newEntries.length > 0) {
        setTranscript(prev => [...prev, ...newEntries]);
      }
    }
  }, [userTranscriptions]);

  const audioTileContent = useMemo(() => {
    const conversationToolbar = (
      <div className="fixed z-50 md:absolute left-1/2 bottom-4 md:bottom-auto md:bottom-8 -translate-x-1/2">
        <motion.div
          className="flex gap-3"
          initial={{
            opacity: 0,
            y: 25,
          }}
          animate={{
            opacity: 1,
            y: 0,
          }}
          exit={{
            opacity: 0,
            y: 25,
          }}
          transition={{
            type: "spring",
            stiffness: 260,
            damping: 20,
          }}
        >
          <Button
            state="destructive"
            className=""
            size="medium"
            onClick={() => {
              onConnect(roomState === ConnectionState.Disconnected);
            }}
          >
            Disconnect
          </Button>
          <MicrophoneButton localMultibandVolume={localMultibandVolume} />
          <Button
            state="secondary"
            size="medium"
            onClick={() => {
              setShowVoEPanel(!showVoEPanel);
            }}
          >
            <MenuSVG />
          </Button>
        </motion.div>
      </div>
    );

    const isLoading =
      roomState === ConnectionState.Connecting ||
      (!agentAudioTrack && roomState === ConnectionState.Connected);

    const startConversationButton = (
      <div className="fixed bottom-2 md:bottom-auto md:absolute left-1/2 md:bottom-8 -translate-x-1/2 w-11/12 md:w-auto text-center">
        <motion.div
          className="flex gap-3"
          initial={{
            opacity: 0,
            y: 50,
          }}
          animate={{
            opacity: 1,
            y: 0,
          }}
          exit={{
            opacity: 0,
            y: 50,
          }}
          transition={{
            type: "spring",
            stiffness: 260,
            damping: 20,
          }}
        >
          <Button
            state="primary"
            size="large"
            className="relative w-full text-sm md:text-base"
            onClick={() => {
              onConnect(roomState === ConnectionState.Disconnected);
            }}
          >
            <div
              className={`w-full ${isLoading ? "opacity-0" : "opacity-100"}`}
            >
              Start a conversation
            </div>
            <div
              className={`absolute left-1/2 top-1/2 -translate-y-1/2 -translate-x-1/2 ${
                isLoading ? "opacity-100" : "opacity-0"
              }`}
            >
              <LoadingSVG diameter={24} strokeWidth={4} />
            </div>
          </Button>
        </motion.div>
      </div>
    );

    // Show conversation history with gradual fade-out
    const maxVisibleMessages = 3;
    const visibleMessages = transcript.slice(-maxVisibleMessages);
    
    // Get interim user transcription (what's currently being spoken)
    const interimText = userTranscriptions && userTranscriptions.length > 0 
      ? userTranscriptions[userTranscriptions.length - 1]?.text || ""
      : "";
    const hasInterimText = interimText && interimText.length > 0 && !userTranscriptions[userTranscriptions.length - 1]?.final;

    const visualizerContent = (
      <div className="flex flex-col items-center justify-center h-full w-full relative">
        {/* Text Display with left fade animation */}
        <div className="absolute inset-0 flex flex-col justify-center px-8 pb-48">
          {/* Assistant label at top */}
          <div className="text-center mb-8">
            <motion.div
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8, ease: "easeOut" }}
            >
              <p className="text-sm text-white/60 font-mono tracking-wider uppercase">
                Assistant
              </p>
            </motion.div>
          </div>

          {/* Main content area */}
          <div className="flex-1 flex items-center justify-center">
            {!agentAudioTrack ? (
              <motion.div
                className="text-center"
                initial={{ opacity: 0, x: -30 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.8, ease: "easeOut" }}
              >
                <h2 className="text-2xl md:text-3xl font-light text-white font-spline">
                  How can I help you?
                </h2>
              </motion.div>
            ) : (
              <div className="w-full max-w-4xl">
                {/* Conversation with left-fade effect */}
                <div className="relative">
                  {/* Gradient mask for left fade */}
                  <div className="absolute inset-0 bg-gradient-to-r from-transparent via-transparent to-background/20 pointer-events-none z-10"></div>
                  
                  <AnimatePresence mode="popLayout">
                    {visibleMessages.map((entry, index) => {
                      const isLatest = index === visibleMessages.length - 1;
                      const opacity = isLatest ? 1 : index === visibleMessages.length - 2 ? 0.6 : 0.3;
                      
                      return (
                        <motion.div
                          key={entry.id}
                          className="mb-4"
                          initial={{ opacity: 0, x: -50 }}
                          animate={{ 
                            opacity, 
                            x: isLatest ? 0 : -20,
                            transition: { duration: 0.8, ease: "easeOut" }
                          }}
                          exit={{ 
                            opacity: 0, 
                            x: -100,
                            transition: { duration: 0.5 }
                          }}
                          layout
                        >
                          <p className="text-lg md:text-xl font-light leading-relaxed font-spline text-white">
                            {entry.text}
                          </p>
                        </motion.div>
                      );
                    })}
                  </AnimatePresence>
                  
                  {/* Live interim text */}
                  <AnimatePresence>
                    {hasInterimText && (
                      <motion.div
                        key="interim"
                        className="mb-4"
                        initial={{ opacity: 0, x: -30 }}
                        animate={{ opacity: 0.8, x: 0 }}
                        exit={{ opacity: 0, x: -50 }}
                        transition={{ duration: 0.4 }}
                      >
                        <p className="text-lg md:text-xl font-light leading-relaxed font-spline text-blue-200">
                          {interimText}
                        </p>
                      </motion.div>
                    )}
                  </AnimatePresence>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Audio Visualizer - moved to bottom */}
        <div className="absolute bottom-32 left-1/2 transform -translate-x-1/2">
          <AgentMultibandAudioVisualizer
            state={agentState}
            barWidth={isMobile ? mobileBarWidth : desktopBarWidth}
            minBarHeight={isMobile ? mobileMinBarHeight : desktopMinBarHeight}
            maxBarHeight={isMobile ? mobileMaxBarHeight : desktopMaxBarHeight}
            frequencies={!agentAudioTrack ? defaultVolumes : subscribedVolumes}
            gap={16}
          />
        </div>

        {/* Controls at bottom */}
        <AnimatePresence>
          {!agentAudioTrack ? startConversationButton : null}
        </AnimatePresence>
        <AnimatePresence>
          {agentAudioTrack ? conversationToolbar : null}
        </AnimatePresence>
      </div>
    );

    return visualizerContent;
  }, [localMultibandVolume, roomState, agentAudioTrack, transcript, userTranscriptions, agentState, isMobile, subscribedVolumes, onConnect, showVoEPanel]);

  return (
    <>
      {/* <Header
        height={headerHeight}
        onConnectClicked={() =>
          onConnect(roomState === ConnectionState.Disconnected)
        }
      /> */}
      <div
        className={`flex grow w-full selection:bg-cyan-900`}
        style={{ height: `calc(100% - ${headerHeight}px)` }}
      >
        <div className="flex-col grow basis-1/2 gap-4 h-full md:flex">
          <Tile
            title="ASSISTANT"
            className="w-full h-full grow"
            childrenClassName="justify-center"
          >
            {audioTileContent}
          </Tile>
        </div>
        <Tile
          padding={false}
          className={`h-full w-full basis-1/4 items-start overflow-hidden hidden max-w-[480px] border-l border-white/20 ${
            showVoEPanel ? "md:flex" : "md:hidden"
          }`}
          childrenClassName="h-full grow items-start"
        >
          <VoEPanel facts={voeFacts} />
        </Tile>
        <div
          className={`bg-white/80 backdrop-blur-lg absolute w-full items-start transition-all duration-100 md:hidden ${
            showVoEPanel ? "translate-x-0" : "translate-x-full"
          }`}
          style={{ height: `calc(100% - ${headerHeight}px)` }}
        >
          <div className="overflow-y-scroll h-full w-full">
            <div className="pb-32">
              <VoEPanel facts={voeFacts} />
            </div>
          </div>
          <div className="pointer-events-none absolute z-10 bottom-0 w-full h-64 bg-gradient-to-t from-white to-transparent"></div>
        </div>
      </div>
    </>
  );
}
