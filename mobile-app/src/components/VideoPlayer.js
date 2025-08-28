import React, { useState, useEffect, useRef } from 'react';
import { View, StyleSheet, Dimensions, Alert } from 'react-native';
import { VideoView, useVideoPlayer } from 'expo-video';
import apiService from '../services/api';

const { width: screenWidth, height: screenHeight } = Dimensions.get('window');

const VideoPlayer = ({ 
  videoName, 
  onViolationDetected, 
  showOverlays = true,
  style = {} 
}) => {
  const [detectionData, setDetectionData] = useState(null);
  const detectionInterval = useRef(null);
  
  // Get video URL
  const videoUrl = videoName ? apiService.getVideoUrl(videoName) : null;
  
  // Create video player instance
  const player = useVideoPlayer(videoUrl, (player) => {
    player.loop = true;
    player.muted = false;
  });

  // Mock detection data generator (similar to web app)
  const generateMockDetection = () => {
    const mockData = {
      prediction: Math.random() > 0.5 ? 'No Parking Zone' : 'Parking Zone',
      vehicles: [
        {
          id: Math.floor(Math.random() * 100),
          bbox: [
            Math.random() * 200,
            Math.random() * 150,
            Math.random() * 100 + 200,
            Math.random() * 80 + 150
          ],
          class: 'car',
          confidence: 0.9 + Math.random() * 0.1,
          duration: Math.random() * 15,
          status: Math.random() > 0.8 ? 'violation' : 'normal'
        }
      ],
      timestamp: Date.now()
    };

    setDetectionData(mockData);

    // Check for violations
    const violations = mockData.vehicles.filter(v => v.status === 'violation');
    if (violations.length > 0 && mockData.prediction === 'No Parking Zone') {
      const violation = {
        id: Date.now(),
        vehicleId: violations[0].id,
        location: videoName,
        duration: violations[0].duration,
        timestamp: new Date(),
        message: `Vehicle #${violations[0].id} parked in no-parking zone for ${violations[0].duration.toFixed(1)}s`
      };

      // Trigger violation callback
      if (onViolationDetected) {
        onViolationDetected(violation);
      }
    }
  };

  // Start/stop detection simulation
  useEffect(() => {
    if (player && player.status === 'readyToPlay') {
      // Start mock detection every 2 seconds
      detectionInterval.current = setInterval(generateMockDetection, 2000);
    } else {
      // Stop detection
      if (detectionInterval.current) {
        clearInterval(detectionInterval.current);
        detectionInterval.current = null;
      }
    }

    return () => {
      if (detectionInterval.current) {
        clearInterval(detectionInterval.current);
      }
    };
  }, [player?.status, videoName]);

  // Handle video load error
  const handleLoadError = (error) => {
    console.error('Video load error:', error);
    Alert.alert('Video Error', 'Failed to load video. Please try again.');
  };

  if (!videoName || !videoUrl) {
    return <View style={[styles.container, style]} />;
  }

  return (
    <View style={[styles.container, style]}>
      <VideoView
        style={styles.video}
        player={player}
        allowsFullscreen
        allowsPictureInPicture
        nativeControls
        contentFit="contain"
      />
      
      {/* Overlay for detection data - This would be implemented with SVG or Canvas */}
      {showOverlays && detectionData && (
        <View style={styles.overlay}>
          {/* This is a simplified overlay - in production you'd use react-native-svg for drawing */}
          <View style={[
            styles.zoneIndicator,
            { backgroundColor: detectionData.prediction === 'Parking Zone' ? 'rgba(0, 255, 0, 0.3)' : 'rgba(255, 0, 0, 0.3)' }
          ]}>
            {/* Zone and detection info would be rendered here */}
          </View>
        </View>
      )}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    backgroundColor: '#000',
    borderRadius: 8,
    overflow: 'hidden',
    aspectRatio: 16 / 9,
  },
  video: {
    flex: 1,
    width: '100%',
  },
  overlay: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    pointerEvents: 'none',
  },
  zoneIndicator: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    height: 40,
  },
});

export default VideoPlayer;