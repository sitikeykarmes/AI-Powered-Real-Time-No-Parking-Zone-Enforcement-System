// VideoFeedScreen.js - With external vehicle info and alerts display
import React, { useState, useEffect } from "react";
import {
  View,
  Text,
  ScrollView,
  StyleSheet,
  Alert,
  TouchableOpacity,
  Dimensions,
  RefreshControl,
} from "react-native";
import { Picker } from "@react-native-picker/picker";
import { Ionicons } from "@expo/vector-icons";
import VideoPlayer from "../components/VideoPlayer";
import apiService from "../services/api";

const { width } = Dimensions.get("window");

const VideoFeedScreen = ({ route, navigation }) => {
  const [availableVideos, setAvailableVideos] = useState({});
  const [selectedVideo, setSelectedVideo] = useState("");
  const [isLoading, setIsLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [violations, setViolations] = useState([]);
  const [highlightedAlert, setHighlightedAlert] = useState(null);
  const [systemStats, setSystemStats] = useState(null);

  // NEW: States for detection data from VideoPlayer
  const [currentDetection, setCurrentDetection] = useState(null);
  const [trackedVehicles, setTrackedVehicles] = useState([]);
  const [persistentAlerts, setPersistentAlerts] = useState([]);

  // Cache detection data per video location
  const [videoDetectionCache, setVideoDetectionCache] = useState({});

  // Fetch videos from backend
  const fetchVideos = async () => {
    try {
      setRefreshing(true);
      const response = await apiService.getVideos();

      if (response.data && response.data.videos) {
        setAvailableVideos(response.data.videos);

        // Set first available video as default
        const videoNames = Object.keys(response.data.videos);
        if (!selectedVideo && videoNames.length > 0) {
          setSelectedVideo(videoNames[0]);
        }
      }
    } catch (error) {
      console.error("Error fetching videos:", error);
      Alert.alert(
        "Connection Error",
        "Failed to connect to backend. Please check if the server is running."
      );
    } finally {
      setRefreshing(false);
      setIsLoading(false);
    }
  };

  // Fetch system statistics
  const fetchStatistics = async () => {
    try {
      const response = await apiService.getStatistics();
      if (response.data && response.data.statistics) {
        setSystemStats(response.data.statistics);
      }
    } catch (error) {
      console.error("Error fetching statistics:", error);
    }
  };

  // NEW: Handle detection updates from VideoPlayer
  const handleDetectionUpdate = (detectionData) => {
    setCurrentDetection(detectionData);

    if (detectionData.vehicles) {
      setTrackedVehicles(detectionData.vehicles);
    }

    if (detectionData.persistentAlerts) {
      setPersistentAlerts(detectionData.persistentAlerts);
    }
  };

  // Handle violation detected from VideoPlayer
  const handleViolationDetected = async (violation) => {
    try {
      // Add to local violations list (keep last 10)
      setViolations((prev) => [violation, ...prev.slice(0, 9)]);

      // Log to backend database
      await apiService.logViolation({
        vehicle_id: violation.vehicleId,
        location: violation.location,
        duration: violation.duration,
        violation_type: "no_parking_zone",
      });

      // Show notification
      Alert.alert(
        "⚠️ Parking Violation Detected",
        `Vehicle #${
          violation.vehicleId
        } has been parked in a no-parking zone at ${
          violation.location
        } for ${violation.duration.toFixed(1)} seconds.`,
        [
          { text: "Dismiss", style: "cancel" },
          {
            text: "View",
            onPress: () => {
              // Scroll to top to see video player
            },
          },
        ]
      );

      // Update statistics
      fetchStatistics();
    } catch (error) {
      console.error("Error logging violation:", error);
    }
  };

  // Reset all alerts
  const resetAlerts = async () => {
    Alert.alert(
      "Reset Alerts",
      "Are you sure you want to reset all violation alerts?",
      [
        { text: "Cancel", style: "cancel" },
        {
          text: "Reset",
          style: "destructive",
          onPress: async () => {
            try {
              await apiService.resetAlerts();
              setViolations([]);
              setPersistentAlerts([]);
              fetchStatistics();
              Alert.alert("Success", "All alerts have been reset");
            } catch (error) {
              console.error("Error resetting alerts:", error);
              Alert.alert("Error", "Failed to reset alerts");
            }
          },
        },
      ]
    );
  };

  // Handle navigation from alerts screen
  useEffect(() => {
    if (route.params?.selectedVideo) {
      setSelectedVideo(route.params.selectedVideo);

      if (route.params?.fromAlert && route.params?.alertData) {
        setHighlightedAlert(route.params.alertData);

        Alert.alert(
          "Alert Location",
          `Showing video feed for: ${route.params.selectedVideo}\n\nVehicle #${
            route.params.alertData.vehicle_id
          } violation detected at ${new Date(
            route.params.alertData.timestamp
          ).toLocaleString()}`,
          [{ text: "OK" }]
        );

        // Clear highlight after 5 seconds
        setTimeout(() => {
          setHighlightedAlert(null);
        }, 5000);
      }
    }
  }, [route.params]);

  // Initial load
  useEffect(() => {
    fetchVideos();
    fetchStatistics();

    // Refresh statistics every 10 seconds
    const interval = setInterval(fetchStatistics, 10000);
    return () => clearInterval(interval);
  }, []);

  if (isLoading) {
    return (
      <View style={styles.centered}>
        <Ionicons name="videocam" size={48} color="#007AFF" />
        <Text style={styles.loadingText}>Loading video feeds...</Text>
      </View>
    );
  }

  return (
    <ScrollView
      style={styles.container}
      refreshControl={
        <RefreshControl refreshing={refreshing} onRefresh={fetchVideos} />
      }
    >
      {/* Alert Info Banner */}
      {highlightedAlert && (
        <View style={styles.alertBanner}>
          <View style={styles.bannerContent}>
            <Ionicons name="information-circle" size={24} color="#007AFF" />
            <View style={styles.bannerText}>
              <Text style={styles.bannerTitle}>Viewing Alert Location</Text>
              <Text style={styles.bannerSubtitle}>
                Vehicle #{highlightedAlert.vehicle_id} -{" "}
                {highlightedAlert.location}
              </Text>
            </View>
          </View>
          <TouchableOpacity
            style={styles.bannerClose}
            onPress={() => setHighlightedAlert(null)}
          >
            <Ionicons name="close" size={20} color="#48484a" />
          </TouchableOpacity>
        </View>
      )}

      {/* Video Selection */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Select CCTV Location</Text>
        <View style={styles.pickerContainer}>
          <Picker
            selectedValue={selectedVideo}
            style={styles.picker}
            onValueChange={(itemValue) => setSelectedVideo(itemValue)}
          >
            <Picker.Item label="Select a location..." value="" />
            {Object.keys(availableVideos).map((videoName) => (
              <Picker.Item
                key={videoName}
                label={`${videoName}${
                  !availableVideos[videoName].exists ? " (Missing)" : ""
                }`}
                value={videoName}
                enabled={availableVideos[videoName].exists}
              />
            ))}
          </Picker>
        </View>
      </View>

      {/* Video Player */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Live Detection Feed</Text>
        <View
          style={[
            styles.videoContainer,
            highlightedAlert && styles.videoContainerHighlighted,
          ]}
        >
          {selectedVideo && availableVideos[selectedVideo]?.exists ? (
            <VideoPlayer
              videoName={selectedVideo}
              onViolationDetected={handleViolationDetected}
              onDetectionUpdate={handleDetectionUpdate}
              showOverlays={true}
              style={styles.videoPlayer}
            />
          ) : (
            <View style={styles.placeholder}>
              <Ionicons name="videocam-off" size={48} color="#8e8e93" />
              <Text style={styles.placeholderText}>
                {selectedVideo
                  ? "Video file not available on server"
                  : "Select a video location to start monitoring"}
              </Text>
            </View>
          )}
        </View>
      </View>

      {/* Tracked Vehicles - OUTSIDE VIDEO PLAYER */}
      {trackedVehicles.length > 0 && (
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>
            🚗 Detected Vehicles ({trackedVehicles.length})
          </Text>
          <View style={styles.vehiclesCard}>
            {trackedVehicles.map((vehicle) => (
              <View
                key={vehicle.id}
                style={[
                  styles.vehicleRow,
                  {
                    borderLeftColor:
                      vehicle.status === "violation"
                        ? "#ef4444"
                        : vehicle.status === "warning"
                        ? "#f59e0b"
                        : "#22c55e",
                  },
                ]}
              >
                <View style={styles.vehicleLeft}>
                  <View style={styles.vehicleIdBadge}>
                    <Text style={styles.vehicleIdText}>#{vehicle.id}</Text>
                  </View>
                  <View style={styles.vehicleInfo}>
                    <Text style={styles.vehicleClass}>{vehicle.class}</Text>
                    <Text style={styles.vehicleConfidence}>
                      {(vehicle.confidence * 100).toFixed(0)}% confidence
                    </Text>
                  </View>
                </View>
                <View style={styles.vehicleRight}>
                  <Text
                    style={[
                      styles.vehicleDuration,
                      {
                        color:
                          vehicle.status === "violation"
                            ? "#ef4444"
                            : "#1c1c1e",
                      },
                    ]}
                  >
                    {vehicle.duration.toFixed(1)}s
                  </Text>
                  {vehicle.status === "violation" && (
                    <View style={styles.violationBadge}>
                      <Text style={styles.violationBadgeText}>
                        ⚠️ VIOLATION
                      </Text>
                    </View>
                  )}
                  {vehicle.status === "warning" && (
                    <View style={styles.warningBadge}>
                      <Text style={styles.warningBadgeText}>⚠ WARNING</Text>
                    </View>
                  )}
                </View>
              </View>
            ))}
          </View>
        </View>
      )}

      {/* Persistent Alerts - OUTSIDE VIDEO PLAYER */}
      {persistentAlerts.length > 0 && (
        <View style={styles.section}>
          <View style={styles.sectionHeader}>
            <Text style={styles.sectionTitle}>
              ⚠️ Active Violations ({persistentAlerts.length})
            </Text>
            <TouchableOpacity style={styles.resetButton} onPress={resetAlerts}>
              <Ionicons name="close-circle" size={20} color="#FF3B30" />
              <Text style={styles.resetButtonText}>Clear</Text>
            </TouchableOpacity>
          </View>
          <View style={styles.alertsCard}>
            {persistentAlerts.map((alert, idx) => (
              <View key={idx} style={styles.alertRow}>
                <View style={styles.alertIcon}>
                  <Ionicons name="warning" size={24} color="#FF3B30" />
                </View>
                <View style={styles.alertContent}>
                  <Text style={styles.alertTitle}>
                    Vehicle #{alert.vehicle_id}
                  </Text>
                  <Text style={styles.alertMessage}>
                    Parked in no-parking zone
                  </Text>
                  <Text style={styles.alertTimestamp}>
                    {new Date(alert.timestamp * 1000).toLocaleTimeString()}
                  </Text>
                </View>
                <View style={styles.alertDurationBox}>
                  <Text style={styles.alertDurationLabel}>Duration</Text>
                  <Text style={styles.alertDurationValue}>
                    {alert.duration?.toFixed(1) || 0}s
                  </Text>
                </View>
              </View>
            ))}
          </View>
        </View>
      )}

      {/* System Statistics */}
      {systemStats && (
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>System Statistics</Text>
          <View style={styles.statsCard}>
            <View style={styles.statItem}>
              <Ionicons name="car" size={24} color="#007AFF" />
              <View style={styles.statContent}>
                <Text style={styles.statLabel}>Total Vehicles Tracked</Text>
                <Text style={styles.statValue}>
                  {systemStats.total_vehicles_tracked || 0}
                </Text>
              </View>
            </View>
            <View style={styles.statItem}>
              <Ionicons name="warning" size={24} color="#FF3B30" />
              <View style={styles.statContent}>
                <Text style={styles.statLabel}>Active Violations</Text>
                <Text style={[styles.statValue, { color: "#FF3B30" }]}>
                  {systemStats.active_violations || 0}
                </Text>
              </View>
            </View>
            <View style={styles.statItem}>
              <Ionicons name="eye" size={24} color="#34C759" />
              <View style={styles.statContent}>
                <Text style={styles.statLabel}>Currently Tracking</Text>
                <Text style={styles.statValue}>
                  {systemStats.tracked_vehicles_count || 0}
                </Text>
              </View>
            </View>
          </View>
        </View>
      )}

      {/* Detection Status */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Detection Status</Text>
        <View style={styles.statusCard}>
          <View style={styles.statusRow}>
            <Text style={styles.statusLabel}>Current Location:</Text>
            <Text style={styles.statusValue}>
              {selectedVideo || "None selected"}
            </Text>
          </View>
          <View style={styles.statusRow}>
            <Text style={styles.statusLabel}>Detection Engine:</Text>
            <View style={styles.statusBadge}>
              <View
                style={[
                  styles.statusDot,
                  {
                    backgroundColor:
                      currentDetection?.connectionStatus === "connected"
                        ? "#34C759"
                        : currentDetection?.connectionStatus === "error"
                        ? "#FF3B30"
                        : "#fbbf24",
                  },
                ]}
              />
              <Text
                style={[
                  styles.statusValue,
                  {
                    color:
                      currentDetection?.connectionStatus === "connected"
                        ? "#34C759"
                        : currentDetection?.connectionStatus === "error"
                        ? "#FF3B30"
                        : "#fbbf24",
                  },
                ]}
              >
                {currentDetection?.connectionStatus === "connected"
                  ? "Active"
                  : currentDetection?.connectionStatus === "error"
                  ? "Error"
                  : "Connecting"}
              </Text>
            </View>
          </View>
          <View style={styles.statusRow}>
            <Text style={styles.statusLabel}>Processing Model:</Text>
            <Text style={styles.statusValue}>YOLO + CNN + RF</Text>
          </View>
          <View style={styles.statusRow}>
            <Text style={styles.statusLabel}>Alert Threshold:</Text>
            <Text style={styles.statusValue}>5.0 seconds</Text>
          </View>
          {currentDetection?.processingTime && (
            <View style={styles.statusRow}>
              <Text style={styles.statusLabel}>Processing Time:</Text>
              <Text style={styles.statusValue}>
                {currentDetection.processingTime.toFixed(2)}s
              </Text>
            </View>
          )}
        </View>
      </View>

      {/* Recent Violations */}
      <View style={styles.section}>
        <View style={styles.sectionHeader}>
          <Text style={styles.sectionTitle}>Recent Violations</Text>
          <TouchableOpacity style={styles.resetButton} onPress={resetAlerts}>
            <Ionicons name="refresh" size={20} color="#007AFF" />
            <Text style={styles.resetButtonText}>Reset</Text>
          </TouchableOpacity>
        </View>

        {violations.length === 0 ? (
          <View style={styles.noViolations}>
            <Ionicons name="checkmark-circle" size={48} color="#34C759" />
            <Text style={styles.noViolationsText}>No violations detected</Text>
            <Text style={styles.noViolationsSubtext}>
              All monitored areas are clear
            </Text>
          </View>
        ) : (
          <View style={styles.violationsList}>
            {violations.map((violation, index) => (
              <View key={index} style={styles.violationItem}>
                <View style={styles.violationIcon}>
                  <Ionicons name="warning" size={20} color="#FF3B30" />
                </View>
                <View style={styles.violationContent}>
                  <Text style={styles.violationTitle}>
                    Vehicle #{violation.vehicleId}
                  </Text>
                  <Text style={styles.violationLocation}>
                    {violation.location}
                  </Text>
                  <Text style={styles.violationTime}>
                    {new Date(violation.timestamp).toLocaleTimeString()}
                  </Text>
                </View>
                <View style={styles.violationRight}>
                  <Text style={styles.violationDuration}>
                    {violation.duration.toFixed(1)}s
                  </Text>
                </View>
              </View>
            ))}
          </View>
        )}
      </View>

      {/* Controls */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Controls</Text>
        <View style={styles.controlsCard}>
          <TouchableOpacity style={styles.controlButton} onPress={resetAlerts}>
            <Ionicons name="refresh-circle" size={24} color="#FF3B30" />
            <Text style={styles.controlButtonText}>Reset All Alerts</Text>
          </TouchableOpacity>

          <TouchableOpacity
            style={styles.controlButton}
            onPress={() => {
              fetchVideos();
              fetchStatistics();
            }}
          >
            <Ionicons name="reload-circle" size={24} color="#34C759" />
            <Text style={styles.controlButtonText}>Refresh Feed</Text>
          </TouchableOpacity>

          <TouchableOpacity
            style={styles.controlButton}
            onPress={() => navigation.navigate("Alerts")}
          >
            <Ionicons name="list-circle" size={24} color="#007AFF" />
            <Text style={styles.controlButtonText}>View All Alerts</Text>
          </TouchableOpacity>
        </View>
      </View>
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#f8f9fa",
  },
  centered: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
    backgroundColor: "#f8f9fa",
  },
  loadingText: {
    marginTop: 12,
    fontSize: 16,
    color: "#48484a",
  },
  alertBanner: {
    backgroundColor: "#E3F2FD",
    marginHorizontal: 16,
    marginTop: 16,
    marginBottom: 8,
    padding: 16,
    borderRadius: 12,
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    borderLeftWidth: 4,
    borderLeftColor: "#007AFF",
  },
  bannerContent: {
    flexDirection: "row",
    alignItems: "center",
    flex: 1,
  },
  bannerText: {
    marginLeft: 12,
    flex: 1,
  },
  bannerTitle: {
    fontSize: 16,
    fontWeight: "600",
    color: "#1c1c1e",
    marginBottom: 2,
  },
  bannerSubtitle: {
    fontSize: 14,
    color: "#48484a",
  },
  bannerClose: {
    padding: 4,
  },
  section: {
    margin: 16,
    marginBottom: 8,
  },
  sectionHeader: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: 12,
  },
  sectionTitle: {
    fontSize: 20,
    fontWeight: "bold",
    color: "#1c1c1e",
    marginBottom: 12,
  },
  pickerContainer: {
    backgroundColor: "white",
    borderRadius: 12,
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  picker: {
    height: 50,
  },
  videoContainer: {
    backgroundColor: "white",
    borderRadius: 12,
    padding: 16,
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  videoContainerHighlighted: {
    borderWidth: 3,
    borderColor: "#007AFF",
    shadowColor: "#007AFF",
    shadowOpacity: 0.3,
  },
  videoPlayer: {
    width: "100%",
    height: ((width - 64) * 9) / 16,
  },
  placeholder: {
    height: ((width - 64) * 9) / 16,
    justifyContent: "center",
    alignItems: "center",
    backgroundColor: "#f2f2f7",
    borderRadius: 8,
  },
  placeholderText: {
    fontSize: 16,
    color: "#8e8e93",
    marginTop: 12,
    textAlign: "center",
    paddingHorizontal: 20,
  },
  // Vehicles Card - NEW STYLES
  vehiclesCard: {
    backgroundColor: "white",
    borderRadius: 12,
    overflow: "hidden",
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  vehicleRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    padding: 16,
    borderLeftWidth: 4,
    borderBottomWidth: 1,
    borderBottomColor: "#f2f2f7",
  },
  vehicleLeft: {
    flexDirection: "row",
    alignItems: "center",
    flex: 1,
  },
  vehicleIdBadge: {
    backgroundColor: "#007AFF",
    paddingHorizontal: 10,
    paddingVertical: 6,
    borderRadius: 6,
    marginRight: 12,
  },
  vehicleIdText: {
    color: "white",
    fontSize: 14,
    fontWeight: "bold",
  },
  vehicleInfo: {
    flex: 1,
  },
  vehicleClass: {
    fontSize: 16,
    fontWeight: "600",
    color: "#1c1c1e",
    textTransform: "capitalize",
  },
  vehicleConfidence: {
    fontSize: 12,
    color: "#8e8e93",
    marginTop: 2,
  },
  vehicleRight: {
    alignItems: "flex-end",
  },
  vehicleDuration: {
    fontSize: 18,
    fontWeight: "bold",
    marginBottom: 4,
  },
  violationBadge: {
    backgroundColor: "#FF3B30",
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 4,
  },
  violationBadgeText: {
    color: "white",
    fontSize: 10,
    fontWeight: "bold",
  },
  warningBadge: {
    backgroundColor: "#f59e0b",
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 4,
  },
  warningBadgeText: {
    color: "white",
    fontSize: 10,
    fontWeight: "bold",
  },
  // Alerts Card - NEW STYLES
  alertsCard: {
    backgroundColor: "white",
    borderRadius: 12,
    overflow: "hidden",
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  alertRow: {
    flexDirection: "row",
    alignItems: "center",
    padding: 16,
    borderBottomWidth: 1,
    borderBottomColor: "#f2f2f7",
    backgroundColor: "#FFF5F5",
  },
  alertIcon: {
    marginRight: 12,
  },
  alertContent: {
    flex: 1,
  },
  alertTitle: {
    fontSize: 16,
    fontWeight: "600",
    color: "#1c1c1e",
  },
  alertMessage: {
    fontSize: 14,
    color: "#FF3B30",
    marginTop: 2,
  },
  alertTimestamp: {
    fontSize: 12,
    color: "#8e8e93",
    marginTop: 4,
  },
  alertDurationBox: {
    backgroundColor: "#FF3B30",
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: 8,
    alignItems: "center",
  },
  alertDurationLabel: {
    fontSize: 10,
    color: "rgba(255, 255, 255, 0.8)",
    marginBottom: 2,
  },
  alertDurationValue: {
    fontSize: 18,
    fontWeight: "bold",
    color: "white",
  },
  // Stats Card
  statsCard: {
    backgroundColor: "white",
    borderRadius: 12,
    padding: 16,
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  statItem: {
    flexDirection: "row",
    alignItems: "center",
    paddingVertical: 12,
    borderBottomWidth: 1,
    borderBottomColor: "#f2f2f7",
  },
  statContent: {
    marginLeft: 12,
    flex: 1,
  },
  statLabel: {
    fontSize: 14,
    color: "#8e8e93",
    marginBottom: 4,
  },
  statValue: {
    fontSize: 20,
    fontWeight: "bold",
    color: "#1c1c1e",
  },
  statusCard: {
    backgroundColor: "white",
    borderRadius: 12,
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  statusRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    padding: 16,
    borderBottomWidth: 1,
    borderBottomColor: "#f2f2f7",
  },
  statusLabel: {
    fontSize: 16,
    color: "#1c1c1e",
  },
  statusValue: {
    fontSize: 16,
    color: "#48484a",
    fontWeight: "600",
  },
  statusBadge: {
    flexDirection: "row",
    alignItems: "center",
    gap: 6,
  },
  statusDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
  },
  resetButton: {
    flexDirection: "row",
    alignItems: "center",
    backgroundColor: "#f2f2f7",
    padding: 8,
    borderRadius: 8,
  },
  resetButtonText: {
    color: "#007AFF",
    marginLeft: 4,
    fontWeight: "600",
  },
  violationsList: {
    backgroundColor: "white",
    borderRadius: 12,
    overflow: "hidden",
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  violationItem: {
    flexDirection: "row",
    alignItems: "center",
    padding: 16,
    borderBottomWidth: 1,
    borderBottomColor: "#f2f2f7",
  },
  violationIcon: {
    marginRight: 12,
  },
  violationContent: {
    flex: 1,
  },
  violationTitle: {
    fontSize: 16,
    fontWeight: "600",
    color: "#1c1c1e",
  },
  violationLocation: {
    fontSize: 14,
    color: "#48484a",
    marginTop: 2,
  },
  violationTime: {
    fontSize: 12,
    color: "#8e8e93",
    marginTop: 2,
  },
  violationRight: {
    alignItems: "flex-end",
  },
  violationDuration: {
    fontSize: 16,
    fontWeight: "bold",
    color: "#FF3B30",
  },
  noViolations: {
    backgroundColor: "white",
    padding: 32,
    borderRadius: 12,
    alignItems: "center",
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  noViolationsText: {
    fontSize: 16,
    fontWeight: "600",
    color: "#1c1c1e",
    marginTop: 12,
  },
  noViolationsSubtext: {
    fontSize: 14,
    color: "#8e8e93",
    marginTop: 4,
  },
  controlsCard: {
    backgroundColor: "white",
    borderRadius: 12,
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  controlButton: {
    flexDirection: "row",
    alignItems: "center",
    padding: 16,
    borderBottomWidth: 1,
    borderBottomColor: "#f2f2f7",
  },
  controlButtonText: {
    fontSize: 16,
    color: "#1c1c1e",
    marginLeft: 12,
  },
});

export default VideoFeedScreen;
