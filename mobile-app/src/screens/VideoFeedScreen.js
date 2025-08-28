import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  ScrollView,
  StyleSheet,
  Alert,
  TouchableOpacity,
  Dimensions,
  RefreshControl,
} from 'react-native';
import { Picker } from '@react-native-picker/picker';
import { Ionicons } from '@expo/vector-icons';
import VideoPlayer from '../components/VideoPlayer';
import apiService from '../services/api';

const { width } = Dimensions.get('window');

const VideoFeedScreen = () => {
  const [availableVideos, setAvailableVideos] = useState({});
  const [selectedVideo, setSelectedVideo] = useState('');
  const [detectionInfo, setDetectionInfo] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [violations, setViolations] = useState([]);

  const fetchVideos = async () => {
    try {
      setRefreshing(true);
      const response = await apiService.getVideos();
      setAvailableVideos(response.data.videos);
      
      // Set first video as default if none selected
      if (!selectedVideo && Object.keys(response.data.videos).length > 0) {
        setSelectedVideo(Object.keys(response.data.videos)[0]);
      }
    } catch (error) {
      console.error('Error fetching videos:', error);
      Alert.alert('Error', 'Failed to load video list');
    } finally {
      setRefreshing(false);
      setIsLoading(false);
    }
  };

  const handleViolationDetected = async (violation) => {
    try {
      // Add to local violations list
      setViolations(prev => [violation, ...prev.slice(0, 9)]);
      
      // Log to backend
      await apiService.logViolation({
        vehicle_id: violation.vehicleId,
        location: violation.location,
        duration: violation.duration,
        violation_type: 'no_parking_zone'
      });

      // Show local notification (simplified)
      Alert.alert(
        'Parking Violation Detected!',
        `Vehicle #${violation.vehicleId} detected in no-parking zone at ${violation.location}`,
        [{ text: 'OK' }]
      );
    } catch (error) {
      console.error('Error logging violation:', error);
    }
  };

  const resetAlerts = async () => {
    try {
      await apiService.resetAlerts();
      setViolations([]);
      Alert.alert('Success', 'All alerts have been reset');
    } catch (error) {
      console.error('Error resetting alerts:', error);
      Alert.alert('Error', 'Failed to reset alerts');
    }
  };

  useEffect(() => {
    fetchVideos();
  }, []);

  if (isLoading) {
    return (
      <View style={styles.centered}>
        <Text>Loading video feed...</Text>
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
                label={videoName}
                value={videoName}
              />
            ))}
          </Picker>
        </View>
      </View>

      {/* Video Player */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Live Detection Feed</Text>
        <View style={styles.videoContainer}>
          {selectedVideo ? (
            <VideoPlayer
              videoName={selectedVideo}
              onViolationDetected={handleViolationDetected}
              showOverlays={true}
              style={styles.videoPlayer}
            />
          ) : (
            <View style={styles.placeholder}>
              <Ionicons name="videocam-off" size={48} color="#8e8e93" />
              <Text style={styles.placeholderText}>
                Select a video location to start monitoring
              </Text>
            </View>
          )}
        </View>
      </View>

      {/* Detection Status */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Detection Status</Text>
        <View style={styles.statusCard}>
          <View style={styles.statusRow}>
            <Text style={styles.statusLabel}>Current Location:</Text>
            <Text style={styles.statusValue}>
              {selectedVideo || 'None selected'}
            </Text>
          </View>
          <View style={styles.statusRow}>
            <Text style={styles.statusLabel}>Detection Engine:</Text>
            <Text style={[styles.statusValue, { color: '#34C759' }]}>
              Active
            </Text>
          </View>
          <View style={styles.statusRow}>
            <Text style={styles.statusLabel}>Processing Model:</Text>
            <Text style={styles.statusValue}>YOLO + CNN + RF</Text>
          </View>
          <View style={styles.statusRow}>
            <Text style={styles.statusLabel}>Alert Threshold:</Text>
            <Text style={styles.statusValue}>5 seconds</Text>
          </View>
        </View>
      </View>

      {/* Recent Detections */}
      <View style={styles.section}>
        <View style={styles.sectionHeader}>
          <Text style={styles.sectionTitle}>Recent Violations</Text>
          <TouchableOpacity
            style={styles.resetButton}
            onPress={resetAlerts}
          >
            <Ionicons name="refresh" size={20} color="#007AFF" />
            <Text style={styles.resetButtonText}>Reset</Text>
          </TouchableOpacity>
        </View>
        
        {violations.length === 0 ? (
          <View style={styles.noViolations}>
            <Ionicons name="checkmark-circle" size={48} color="#34C759" />
            <Text style={styles.noViolationsText}>
              No violations detected
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
                <Text style={styles.violationDuration}>
                  {violation.duration.toFixed(1)}s
                </Text>
              </View>
            ))}
          </View>
        )}
      </View>

      {/* Controls */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Controls</Text>
        <View style={styles.controlsCard}>
          <TouchableOpacity
            style={styles.controlButton}
            onPress={resetAlerts}
          >
            <Ionicons name="refresh-circle" size={24} color="#007AFF" />
            <Text style={styles.controlButtonText}>Reset All Alerts</Text>
          </TouchableOpacity>
          
          <TouchableOpacity
            style={styles.controlButton}
            onPress={fetchVideos}
          >
            <Ionicons name="reload-circle" size={24} color="#34C759" />
            <Text style={styles.controlButtonText}>Refresh Feed</Text>
          </TouchableOpacity>
        </View>
      </View>
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f8f9fa',
  },
  centered: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  section: {
    margin: 16,
    marginBottom: 24,
  },
  sectionHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 12,
  },
  sectionTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#1c1c1e',
    marginBottom: 12,
  },
  pickerContainer: {
    backgroundColor: 'white',
    borderRadius: 12,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  picker: {
    height: 50,
  },
  videoContainer: {
    backgroundColor: 'white',
    borderRadius: 12,
    padding: 16,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  videoPlayer: {
    width: '100%',
    height: (width - 64) * 9 / 16, // 16:9 aspect ratio
  },
  placeholder: {
    height: (width - 64) * 9 / 16,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#f2f2f7',
    borderRadius: 8,
  },
  placeholderText: {
    fontSize: 16,
    color: '#8e8e93',
    marginTop: 12,
    textAlign: 'center',
  },
  statusCard: {
    backgroundColor: 'white',
    borderRadius: 12,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  statusRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 16,
    borderBottomWidth: 1,
    borderBottomColor: '#f2f2f7',
  },
  statusLabel: {
    fontSize: 16,
    color: '#1c1c1e',
  },
  statusValue: {
    fontSize: 16,
    color: '#48484a',
    fontWeight: '600',
  },
  resetButton: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#f2f2f7',
    padding: 8,
    borderRadius: 8,
  },
  resetButtonText: {
    color: '#007AFF',
    marginLeft: 4,
    fontWeight: '600',
  },
  violationsList: {
    backgroundColor: 'white',
    borderRadius: 12,
    overflow: 'hidden',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  violationItem: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 16,
    borderBottomWidth: 1,
    borderBottomColor: '#f2f2f7',
  },
  violationIcon: {
    marginRight: 12,
  },
  violationContent: {
    flex: 1,
  },
  violationTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#1c1c1e',
  },
  violationLocation: {
    fontSize: 14,
    color: '#48484a',
    marginTop: 2,
  },
  violationTime: {
    fontSize: 12,
    color: '#8e8e93',
    marginTop: 2,
  },
  violationDuration: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#FF3B30',
  },
  noViolations: {
    backgroundColor: 'white',
    padding: 32,
    borderRadius: 12,
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  noViolationsText: {
    fontSize: 16,
    color: '#48484a',
    marginTop: 12,
  },
  controlsCard: {
    backgroundColor: 'white',
    borderRadius: 12,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  controlButton: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 16,
    borderBottomWidth: 1,
    borderBottomColor: '#f2f2f7',
  },
  controlButtonText: {
    fontSize: 16,
    color: '#1c1c1e',
    marginLeft: 12,
  },
});

export default VideoFeedScreen;