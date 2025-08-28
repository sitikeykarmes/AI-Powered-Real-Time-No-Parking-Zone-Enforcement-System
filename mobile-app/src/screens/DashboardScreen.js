import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  ScrollView,
  StyleSheet,
  RefreshControl,
  Alert,
  Dimensions,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import apiService from '../services/api';

const { width } = Dimensions.get('window');

const StatCard = ({ title, value, icon, color = '#007AFF', subtitle = '' }) => (
  <View style={[styles.statCard, { borderLeftColor: color }]}>
    <View style={styles.statHeader}>
      <Ionicons name={icon} size={24} color={color} />
      <Text style={styles.statTitle}>{title}</Text>
    </View>
    <Text style={styles.statValue}>{value}</Text>
    {subtitle ? <Text style={styles.statSubtitle}>{subtitle}</Text> : null}
  </View>
);

const AlertItem = ({ alert }) => (
  <View style={styles.alertItem}>
    <View style={styles.alertIcon}>
      <Ionicons name="warning" size={20} color="#FF3B30" />
    </View>
    <View style={styles.alertContent}>
      <Text style={styles.alertTitle}>Vehicle #{alert.vehicle_id}</Text>
      <Text style={styles.alertLocation}>{alert.location}</Text>
      <Text style={styles.alertTime}>
        {new Date(alert.timestamp).toLocaleTimeString()}
      </Text>
    </View>
    <Text style={styles.alertDuration}>{alert.duration.toFixed(1)}s</Text>
  </View>
);

const DashboardScreen = () => {
  const [stats, setStats] = useState({
    totalVideos: 0,
    activeAlerts: 0,
    totalViolations: 0,
    systemStatus: 'Loading...'
  });
  const [recentAlerts, setRecentAlerts] = useState([]);
  const [refreshing, setRefreshing] = useState(false);
  const [loading, setLoading] = useState(true);

  const fetchDashboardData = async () => {
    try {
      setRefreshing(true);
      
      // Fetch videos
      const videosResponse = await apiService.getVideos();
      const totalVideos = Object.keys(videosResponse.data.videos).length;

      // Fetch violations
      const violationsResponse = await apiService.getViolations();
      const violations = violationsResponse.data;
      
      // Calculate recent alerts (last 24 hours)
      const now = new Date();
      const oneDayAgo = new Date(now.getTime() - 24 * 60 * 60 * 1000);
      const recentViolations = violations.filter(v => 
        new Date(v.timestamp) > oneDayAgo
      );

      setStats({
        totalVideos,
        activeAlerts: recentViolations.length,
        totalViolations: violations.length,
        systemStatus: 'Online'
      });

      // Set recent alerts (limit to 5 most recent)
      setRecentAlerts(recentViolations.slice(0, 5));

    } catch (error) {
      console.error('Error fetching dashboard data:', error);
      Alert.alert('Error', 'Failed to load dashboard data');
      setStats(prev => ({ ...prev, systemStatus: 'Offline' }));
    } finally {
      setRefreshing(false);
      setLoading(false);
    }
  };

  const onRefresh = () => {
    fetchDashboardData();
  };

  useEffect(() => {
    fetchDashboardData();
    
    // Set up auto-refresh every 30 seconds
    const interval = setInterval(fetchDashboardData, 30000);
    
    return () => clearInterval(interval);
  }, []);

  if (loading) {
    return (
      <View style={styles.centered}>
        <Text>Loading dashboard...</Text>
      </View>
    );
  }

  return (
    <ScrollView
      style={styles.container}
      refreshControl={
        <RefreshControl refreshing={refreshing} onRefresh={onRefresh} />
      }
    >
      {/* System Status */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>System Status</Text>
        <View style={styles.statusContainer}>
          <View style={styles.statusIndicator}>
            <Ionicons 
              name={stats.systemStatus === 'Online' ? 'checkmark-circle' : 'alert-circle'} 
              size={24} 
              color={stats.systemStatus === 'Online' ? '#34C759' : '#FF3B30'} 
            />
            <Text style={[
              styles.statusText,
              { color: stats.systemStatus === 'Online' ? '#34C759' : '#FF3B30' }
            ]}>
              {stats.systemStatus}
            </Text>
          </View>
          <Text style={styles.lastUpdated}>
            Last updated: {new Date().toLocaleTimeString()}
          </Text>
        </View>
      </View>

      {/* Statistics Cards */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Overview</Text>
        <View style={styles.statsGrid}>
          <StatCard
            title="CCTV Locations"
            value={stats.totalVideos}
            icon="videocam"
            color="#007AFF"
            subtitle="Active cameras"
          />
          <StatCard
            title="Active Alerts"
            value={stats.activeAlerts}
            icon="warning"
            color="#FF9500"
            subtitle="Last 24 hours"
          />
        </View>
        <View style={styles.statsGrid}>
          <StatCard
            title="Total Violations"
            value={stats.totalViolations}
            icon="ban"
            color="#FF3B30"
            subtitle="All time"
          />
          <StatCard
            title="Detection Model"
            value="Active"
            icon="eye"
            color="#34C759"
            subtitle="YOLO + CNN + RF"
          />
        </View>
      </View>

      {/* Recent Alerts */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Recent Alerts</Text>
        {recentAlerts.length === 0 ? (
          <View style={styles.noAlerts}>
            <Ionicons name="checkmark-circle" size={48} color="#34C759" />
            <Text style={styles.noAlertsText}>No recent violations detected</Text>
          </View>
        ) : (
          <View style={styles.alertsList}>
            {recentAlerts.map((alert, index) => (
              <AlertItem key={index} alert={alert} />
            ))}
          </View>
        )}
      </View>

      {/* Detection Settings */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Detection Settings</Text>
        <View style={styles.settingsCard}>
          <View style={styles.settingRow}>
            <Text style={styles.settingLabel}>Alert Threshold</Text>
            <Text style={styles.settingValue}>5 seconds</Text>
          </View>
          <View style={styles.settingRow}>
            <Text style={styles.settingLabel}>Processing Mode</Text>
            <Text style={styles.settingValue}>Real-time</Text>
          </View>
          <View style={styles.settingRow}>
            <Text style={styles.settingLabel}>Notification</Text>
            <Text style={styles.settingValue}>Enabled</Text>
          </View>
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
  sectionTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#1c1c1e',
    marginBottom: 12,
  },
  statusContainer: {
    backgroundColor: 'white',
    padding: 16,
    borderRadius: 12,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  statusIndicator: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 8,
  },
  statusText: {
    fontSize: 18,
    fontWeight: '600',
    marginLeft: 8,
  },
  lastUpdated: {
    fontSize: 12,
    color: '#8e8e93',
  },
  statsGrid: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 12,
  },
  statCard: {
    backgroundColor: 'white',
    padding: 16,
    borderRadius: 12,
    borderLeftWidth: 4,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
    width: (width - 48) / 2,
  },
  statHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 8,
  },
  statTitle: {
    fontSize: 14,
    fontWeight: '600',
    color: '#48484a',
    marginLeft: 8,
  },
  statValue: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#1c1c1e',
  },
  statSubtitle: {
    fontSize: 12,
    color: '#8e8e93',
    marginTop: 4,
  },
  alertsList: {
    backgroundColor: 'white',
    borderRadius: 12,
    overflow: 'hidden',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  alertItem: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 16,
    borderBottomWidth: 1,
    borderBottomColor: '#f2f2f7',
  },
  alertIcon: {
    marginRight: 12,
  },
  alertContent: {
    flex: 1,
  },
  alertTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#1c1c1e',
  },
  alertLocation: {
    fontSize: 14,
    color: '#48484a',
    marginTop: 2,
  },
  alertTime: {
    fontSize: 12,
    color: '#8e8e93',
    marginTop: 2,
  },
  alertDuration: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#FF3B30',
  },
  noAlerts: {
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
  noAlertsText: {
    fontSize: 16,
    color: '#48484a',
    marginTop: 12,
  },
  settingsCard: {
    backgroundColor: 'white',
    borderRadius: 12,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  settingRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 16,
    borderBottomWidth: 1,
    borderBottomColor: '#f2f2f7',
  },
  settingLabel: {
    fontSize: 16,
    color: '#1c1c1e',
  },
  settingValue: {
    fontSize: 16,
    color: '#007AFF',
    fontWeight: '600',
  },
});

export default DashboardScreen;