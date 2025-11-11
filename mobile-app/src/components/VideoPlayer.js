// VideoPlayer.js - Prediction bar only, info displayed externally
import React, { useState, useEffect, useRef } from "react";
import {
  View,
  StyleSheet,
  Dimensions,
  Alert,
  Text,
  ActivityIndicator,
} from "react-native";
import { VideoView, useVideoPlayer } from "expo-video";
import apiService from "../services/api";

const { width: screenWidth } = Dimensions.get("window");

const VideoPlayer = ({
  videoName,
  onViolationDetected,
  onDetectionUpdate, // NEW: Callback to send detection data to parent
  showOverlays = true,
  style = {},
}) => {
  const [detectionData, setDetectionData] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [persistentAlerts, setPersistentAlerts] = useState([]);
  const [processingTime, setProcessingTime] = useState(0);
  const [connectionStatus, setConnectionStatus] = useState("disconnected");
  const detectionInterval = useRef(null);
  const lastProcessedTime = useRef(0);
  const hasInitialized = useRef(false);

  // Get video URL
  const videoUrl = videoName ? apiService.getVideoUrl(videoName) : null;

  // Create video player instance with autoplay
  const player = useVideoPlayer(videoUrl, (player) => {
    player.loop = true;
    player.muted = false;
    player.play(); // Auto-play when video is ready
  });

  // Process current video frame via backend
  const processCurrentFrame = async () => {
    if (isProcessing || !videoName || !player) return;

    try {
      setIsProcessing(true);
      const startTime = Date.now();

      // Get current video timestamp
      const currentTime = player.currentTime || 0;

      // Skip if we just processed this timestamp
      if (Math.abs(currentTime - lastProcessedTime.current) < 1) {
        setIsProcessing(false);
        return;
      }

      lastProcessedTime.current = currentTime;

      // Request backend to process video at this timestamp
      const response = await apiService.processVideoFrame({
        video_name: videoName,
        timestamp: currentTime,
        alert_threshold: 5.0,
      });

      const endTime = Date.now();
      setProcessingTime((endTime - startTime) / 1000);
      setConnectionStatus("connected");

      if (response.data) {
        // Update detection data
        setDetectionData(response.data);

        // Update persistent alerts
        if (response.data.alerts && response.data.alerts.length > 0) {
          setPersistentAlerts(response.data.alerts);
        } else if (response.data.alerts) {
          setPersistentAlerts([]);
        }

        // Send detection data to parent component
        if (onDetectionUpdate) {
          onDetectionUpdate({
            ...response.data,
            processingTime: (endTime - startTime) / 1000,
            connectionStatus: "connected",
            persistentAlerts: response.data.alerts || [],
          });
        }

        // Handle new violations
        if (response.data.new_alerts && response.data.new_alerts.length > 0) {
          response.data.new_alerts.forEach((alert) => {
            const violation = {
              id: Date.now() + Math.random(),
              vehicleId: alert.vehicle_id,
              location: videoName,
              duration: alert.duration || 0,
              timestamp: new Date(),
              message:
                alert.text || `Vehicle #${alert.vehicle_id} violation detected`,
            };

            // Trigger callback
            if (onViolationDetected) {
              onViolationDetected(violation);
            }
          });
        }
      }
    } catch (error) {
      console.error("Error processing frame:", error);
      setConnectionStatus("error");

      // Send error status to parent
      if (onDetectionUpdate) {
        onDetectionUpdate({
          connectionStatus: "error",
          error: error.message,
        });
      }

      // Only show alert for critical errors
      if (error.response?.status === 404) {
        Alert.alert(
          "Video Not Found",
          "The selected video is not available on the server."
        );
      }
    } finally {
      setIsProcessing(false);
    }
  };

  // Start/stop detection loop
  useEffect(() => {
    if (player && player.status === "readyToPlay" && videoName) {
      // Initial connection check
      setConnectionStatus("connecting");

      // Process immediately
      setTimeout(processCurrentFrame, 500);

      // Then process every 3 seconds
      detectionInterval.current = setInterval(processCurrentFrame, 3000);
    } else {
      // Stop detection
      if (detectionInterval.current) {
        clearInterval(detectionInterval.current);
        detectionInterval.current = null;
      }

      // Clear data
      setDetectionData(null);
      setPersistentAlerts([]);
      setConnectionStatus("disconnected");

      // Notify parent of disconnection
      if (onDetectionUpdate) {
        onDetectionUpdate({
          connectionStatus: "disconnected",
          prediction: null,
          vehicles: [],
          alerts: [],
        });
      }
    }

    return () => {
      if (detectionInterval.current) {
        clearInterval(detectionInterval.current);
      }
    };
  }, [player?.status, videoName]);

  if (!videoName || !videoUrl) {
    return (
      <View style={[styles.container, styles.centered, style]}>
        <Text style={styles.placeholderText}>No video selected</Text>
      </View>
    );
  }

  return (
    <View style={[styles.container, style]}>
      {/* Video Player */}
      <VideoView
        style={styles.video}
        player={player}
        allowsFullscreen
        allowsPictureInPicture
        nativeControls
        contentFit="contain"
      />

      {/* Only Prediction Bar Overlay */}
      {showOverlays && detectionData && (
        <View style={styles.overlay}>
          {/* Zone Indicator Bar - ONLY THIS STAYS */}
          <View
            style={[
              styles.zoneIndicator,
              {
                backgroundColor:
                  detectionData.prediction === "Parking Zone"
                    ? "rgba(34, 197, 94, 0.85)"
                    : "rgba(239, 68, 68, 0.85)",
              },
            ]}
          >
            <View style={styles.zoneLeft}>
              <Text style={styles.zoneText}>
                {detectionData.prediction || "Unknown"}
              </Text>
              <Text style={styles.zoneSubtext}>
                {detectionData.is_no_parking_zone
                  ? "⚠️ No Parking"
                  : "✓ Parking Allowed"}
              </Text>
            </View>
            <View style={styles.zoneRight}>
              <Text style={styles.processingText}>
                {processingTime.toFixed(2)}s
              </Text>
              <View
                style={[
                  styles.statusDot,
                  {
                    backgroundColor:
                      connectionStatus === "connected"
                        ? "#22c55e"
                        : connectionStatus === "error"
                        ? "#ef4444"
                        : "#fbbf24",
                  },
                ]}
              />
            </View>
          </View>
        </View>
      )}

      {/* Processing Indicator */}
      {isProcessing && (
        <View style={styles.processingIndicator}>
          <ActivityIndicator size="small" color="#fff" />
        </View>
      )}

      {/* Connection Status (bottom right) */}
      <View style={styles.connectionStatus}>
        <View
          style={[
            styles.connectionDot,
            {
              backgroundColor:
                connectionStatus === "connected"
                  ? "#22c55e"
                  : connectionStatus === "connecting"
                  ? "#fbbf24"
                  : connectionStatus === "error"
                  ? "#ef4444"
                  : "#6b7280",
            },
          ]}
        />
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    backgroundColor: "#000",
    borderRadius: 8,
    overflow: "hidden",
    aspectRatio: 16 / 9,
    position: "relative",
  },
  centered: {
    justifyContent: "center",
    alignItems: "center",
  },
  video: {
    flex: 1,
    width: "100%",
  },
  placeholderText: {
    color: "#9ca3af",
    fontSize: 16,
  },
  overlay: {
    position: "absolute",
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    pointerEvents: "none",
  },
  // Zone Indicator - Only overlay element
  zoneIndicator: {
    position: "absolute",
    top: 0,
    left: 0,
    right: 0,
    paddingVertical: 12,
    paddingHorizontal: 16,
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.25,
    shadowRadius: 3.84,
    elevation: 5,
  },
  zoneLeft: {
    flex: 1,
  },
  zoneText: {
    color: "white",
    fontSize: 18,
    fontWeight: "bold",
  },
  zoneSubtext: {
    color: "rgba(255, 255, 255, 0.9)",
    fontSize: 12,
    marginTop: 2,
  },
  zoneRight: {
    alignItems: "flex-end",
  },
  processingText: {
    color: "rgba(255, 255, 255, 0.9)",
    fontSize: 11,
    marginBottom: 4,
  },
  statusDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
  },
  // Processing Indicator
  processingIndicator: {
    position: "absolute",
    top: 12,
    right: 12,
    backgroundColor: "rgba(251, 146, 60, 0.9)",
    paddingVertical: 6,
    paddingHorizontal: 10,
    borderRadius: 16,
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.25,
    shadowRadius: 3.84,
    elevation: 5,
  },
  // Connection Status
  connectionStatus: {
    position: "absolute",
    bottom: 8,
    right: 8,
    backgroundColor: "rgba(0, 0, 0, 0.6)",
    padding: 6,
    borderRadius: 12,
  },
  connectionDot: {
    width: 10,
    height: 10,
    borderRadius: 5,
  },
});

export default VideoPlayer;
