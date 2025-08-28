import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  ScrollView,
  StyleSheet,
  RefreshControl,
  Alert,
  TouchableOpacity,
  FlatList,
  Switch,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import * as Notifications from 'expo-notifications';
import apiService from '../services/api';

// Configure notifications
Notifications.setNotificationHandler({
  handleNotification: async () => ({
    shouldShowAlert: true,
    shouldPlaySound: true,
    shouldSetBadge: true,
  }),
});

const AlertItem = ({ alert, onDismiss }) => (
  <View style={styles.alertItem}>
    <View style={styles.alertIcon}>
      <Ionicons 
        name="warning" 
        size={24} 
        color={alert.violation_type === 'no_parking_zone' ? '#FF3B30' : '#FF9500'} 
      />
    </View>
    <View style={styles.alertContent}>
      <Text style={styles.alertTitle}>
        Vehicle #{alert.vehicle_id} - Violation
      </Text>
      <Text style={styles.alertLocation}>
        Location: {alert.location}
      </Text>
      <Text style={styles.alertDescription}>
        Parked in no-parking zone for {alert.duration?.toFixed(1) || 0}s
      </Text>
      <Text style={styles.alertTime}>
        {new Date(alert.timestamp).toLocaleString()}
      </Text>
    </View>
    <View style={styles.alertActions}>
      <Text style={styles.alertDuration}>
        {alert.duration?.toFixed(1) || 0}s
      </Text>
      <TouchableOpacity
        style={styles.dismissButton}
        onPress={() => onDismiss(alert.id)}
      >
        <Ionicons name="close-circle" size={20} color="#8e8e93" />
      </TouchableOpacity>
    </View>
  </View>
);

const FilterButton = ({ title, active, onPress, color = '#007AFF' }) => (
  <TouchableOpacity
    style={[
      styles.filterButton,
      { backgroundColor: active ? color : '#f2f2f7' }
    ]}
    onPress={onPress}
  >
    <Text style={[
      styles.filterButtonText,
      { color: active ? 'white' : '#1c1c1e' }
    ]}>
      {title}
    </Text>
  </TouchableOpacity>
);

const AlertsScreen = () => {
  const [alerts, setAlerts] = useState([]);
  const [filteredAlerts, setFilteredAlerts] = useState([]);
  const [refreshing, setRefreshing] = useState(false);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState('all'); // 'all', 'today', 'recent'
  const [notificationsEnabled, setNotificationsEnabled] = useState(true);
  const [sortBy, setSortBy] = useState('newest'); // 'newest', 'oldest', 'duration'

  // Request notification permissions
  const requestNotificationPermissions = async () => {
    const { status } = await Notifications.requestPermissionsAsync();
    if (status !== 'granted') {
      Alert.alert(
        'Notification Permission',
        'Please enable notifications to receive parking violation alerts.'
      );
    }
  };

  // Send local notification for new violation
  const sendNotification = async (violation) => {
    if (!notificationsEnabled) return;

    await Notifications.scheduleNotificationAsync({
      content: {
        title: 'Parking Violation Detected!',
        body: `Vehicle #${violation.vehicle_id} in no-parking zone at ${violation.location}`,
        data: { violationId: violation.id },
      },
      trigger: null, // Show immediately
    });
  };

  const fetchAlerts = async () => {
    try {
      setRefreshing(true);
      const response = await apiService.getViolations();
      const alertsData = response.data.map(violation => ({
        ...violation,
        id: violation.id || `${violation.vehicle_id}-${violation.timestamp}`,
      }));
      
      setAlerts(alertsData);
      applyFilter(alertsData, filter);
    } catch (error) {
      console.error('Error fetching alerts:', error);
      Alert.alert('Error', 'Failed to load alerts');
    } finally {
      setRefreshing(false);
      setLoading(false);
    }
  };

  const applyFilter = (alertsData, filterType) => {
    let filtered = [...alertsData];
    const now = new Date();

    switch (filterType) {
      case 'today':
        const today = new Date(now.getFullYear(), now.getMonth(), now.getDate());
        filtered = alertsData.filter(alert => 
          new Date(alert.timestamp) >= today
        );
        break;
      case 'recent':
        const oneDayAgo = new Date(now.getTime() - 24 * 60 * 60 * 1000);
        filtered = alertsData.filter(alert => 
          new Date(alert.timestamp) >= oneDayAgo
        );
        break;
      default: // 'all'
        break;
    }

    // Apply sorting
    switch (sortBy) {
      case 'oldest':
        filtered.sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp));
        break;
      case 'duration':
        filtered.sort((a, b) => (b.duration || 0) - (a.duration || 0));
        break;
      default: // 'newest'
        filtered.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));
        break;
    }

    setFilteredAlerts(filtered);
  };

  const dismissAlert = (alertId) => {
    Alert.alert(
      'Dismiss Alert',
      'Are you sure you want to dismiss this alert?',
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Dismiss',
          style: 'destructive',
          onPress: () => {
            const updatedAlerts = alerts.filter(alert => alert.id !== alertId);
            setAlerts(updatedAlerts);
            applyFilter(updatedAlerts, filter);
          },
        },
      ]
    );
  };

  const clearAllAlerts = async () => {
    Alert.alert(
      'Clear All Alerts',
      'This will remove all alerts from the list. Are you sure?',
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Clear All',
          style: 'destructive',
          onPress: async () => {
            try {
              await apiService.resetAlerts();
              setAlerts([]);
              setFilteredAlerts([]);
            } catch (error) {
              console.error('Error clearing alerts:', error);
              Alert.alert('Error', 'Failed to clear alerts');
            }
          },
        },
      ]
    );
  };

  const handleFilterChange = (newFilter) => {
    setFilter(newFilter);
    applyFilter(alerts, newFilter);
  };

  const handleSortChange = (newSort) => {
    setSortBy(newSort);
    applyFilter(alerts, filter);
  };

  useEffect(() => {
    requestNotificationPermissions();
    fetchAlerts();

    // Set up auto-refresh every 30 seconds
    const interval = setInterval(fetchAlerts, 30000);
    
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    applyFilter(alerts, filter);
  }, [sortBy]);

  if (loading) {
    return (
      <View style={styles.centered}>
        <Text>Loading alerts...</Text>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      {/* Header Stats */}
      <View style={styles.headerStats}>
        <View style={styles.statItem}>
          <Text style={styles.statNumber}>{alerts.length}</Text>
          <Text style={styles.statLabel}>Total Alerts</Text>
        </View>
        <View style={styles.statItem}>
          <Text style={styles.statNumber}>{filteredAlerts.length}</Text>
          <Text style={styles.statLabel}>Filtered</Text>
        </View>
        <View style={styles.statItem}>
          <Text style={[styles.statNumber, { color: '#34C759' }]}>
            {alerts.filter(a => new Date(a.timestamp) > new Date(Date.now() - 24*60*60*1000)).length}
          </Text>
          <Text style={styles.statLabel}>Recent</Text>
        </View>
      </View>

      {/* Filters */}
      <View style={styles.filtersContainer}>
        <ScrollView horizontal showsHorizontalScrollIndicator={false}>
          <FilterButton
            title="All"
            active={filter === 'all'}
            onPress={() => handleFilterChange('all')}
          />
          <FilterButton
            title="Today"
            active={filter === 'today'}
            onPress={() => handleFilterChange('today')}
            color="#34C759"
          />
          <FilterButton
            title="Recent"
            active={filter === 'recent'}
            onPress={() => handleFilterChange('recent')}
            color="#FF9500"
          />
        </ScrollView>
      </View>

      {/* Sort and Settings */}
      <View style={styles.controlsRow}>
        <View style={styles.sortContainer}>
          <Text style={styles.sortLabel}>Sort by:</Text>
          <TouchableOpacity
            style={styles.sortButton}
            onPress={() => {
              const sorts = ['newest', 'oldest', 'duration'];
              const currentIndex = sorts.indexOf(sortBy);
              const nextSort = sorts[(currentIndex + 1) % sorts.length];
              handleSortChange(nextSort);
            }}
          >
            <Text style={styles.sortButtonText}>
              {sortBy === 'newest' ? 'Newest' : 
               sortBy === 'oldest' ? 'Oldest' : 'Duration'}
            </Text>
            <Ionicons name="chevron-down" size={16} color="#007AFF" />
          </TouchableOpacity>
        </View>

        <View style={styles.notificationToggle}>
          <Text style={styles.toggleLabel}>Notifications</Text>
          <Switch
            value={notificationsEnabled}
            onValueChange={setNotificationsEnabled}
            trackColor={{ false: '#f2f2f7', true: '#007AFF' }}
          />
        </View>
      </View>

      {/* Alerts List */}
      <FlatList
        data={filteredAlerts}
        keyExtractor={(item) => item.id}
        renderItem={({ item }) => (
          <AlertItem alert={item} onDismiss={dismissAlert} />
        )}
        refreshControl={
          <RefreshControl refreshing={refreshing} onRefresh={fetchAlerts} />
        }
        ListEmptyComponent={() => (
          <View style={styles.emptyState}>
            <Ionicons name="checkmark-circle" size={64} color="#34C759" />
            <Text style={styles.emptyTitle}>No Alerts</Text>
            <Text style={styles.emptySubtitle}>
              {filter === 'all' 
                ? 'No parking violations detected'
                : `No violations found for ${filter} filter`
              }
            </Text>
          </View>
        )}
        style={styles.alertsList}
      />

      {/* Action Buttons */}
      {alerts.length > 0 && (
        <View style={styles.actionButtons}>
          <TouchableOpacity
            style={[styles.actionButton, styles.refreshButton]}
            onPress={fetchAlerts}
          >
            <Ionicons name="refresh" size={20} color="#007AFF" />
            <Text style={styles.refreshButtonText}>Refresh</Text>
          </TouchableOpacity>

          <TouchableOpacity
            style={[styles.actionButton, styles.clearButton]}
            onPress={clearAllAlerts}
          >
            <Ionicons name="trash" size={20} color="#FF3B30" />
            <Text style={styles.clearButtonText}>Clear All</Text>
          </TouchableOpacity>
        </View>
      )}
    </View>
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
  headerStats: {
    flexDirection: 'row',
    backgroundColor: 'white',
    padding: 16,
    marginHorizontal: 16,
    marginTop: 16,
    borderRadius: 12,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  statItem: {
    flex: 1,
    alignItems: 'center',
  },
  statNumber: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#1c1c1e',
  },
  statLabel: {
    fontSize: 12,
    color: '#8e8e93',
    marginTop: 4,
  },
  filtersContainer: {
    padding: 16,
  },
  filterButton: {
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 20,
    marginRight: 8,
  },
  filterButtonText: {
    fontSize: 14,
    fontWeight: '600',
  },
  controlsRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: 16,
    paddingBottom: 16,
  },
  sortContainer: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  sortLabel: {
    fontSize: 14,
    color: '#48484a',
    marginRight: 8,
  },
  sortButton: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 12,
    paddingVertical: 6,
    backgroundColor: '#f2f2f7',
    borderRadius: 8,
  },
  sortButtonText: {
    fontSize: 14,
    color: '#007AFF',
    marginRight: 4,
  },
  notificationToggle: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  toggleLabel: {
    fontSize: 14,
    color: '#48484a',
    marginRight: 8,
  },
  alertsList: {
    flex: 1,
  },
  alertItem: {
    flexDirection: 'row',
    backgroundColor: 'white',
    padding: 16,
    marginHorizontal: 16,
    marginBottom: 8,
    borderRadius: 12,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  alertIcon: {
    marginRight: 12,
    alignItems: 'center',
    justifyContent: 'center',
  },
  alertContent: {
    flex: 1,
  },
  alertTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#1c1c1e',
    marginBottom: 4,
  },
  alertLocation: {
    fontSize: 14,
    color: '#48484a',
    marginBottom: 2,
  },
  alertDescription: {
    fontSize: 14,
    color: '#8e8e93',
    marginBottom: 4,
  },
  alertTime: {
    fontSize: 12,
    color: '#8e8e93',
  },
  alertActions: {
    alignItems: 'center',
    justifyContent: 'center',
  },
  alertDuration: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#FF3B30',
    marginBottom: 8,
  },
  dismissButton: {
    padding: 4,
  },
  emptyState: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 64,
  },
  emptyTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#1c1c1e',
    marginTop: 16,
  },
  emptySubtitle: {
    fontSize: 16,
    color: '#8e8e93',
    textAlign: 'center',
    marginTop: 8,
  },
  actionButtons: {
    flexDirection: 'row',
    padding: 16,
    gap: 12,
  },
  actionButton: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 16,
    borderRadius: 12,
  },
  refreshButton: {
    backgroundColor: '#f2f2f7',
  },
  refreshButtonText: {
    color: '#007AFF',
    fontWeight: '600',
    marginLeft: 8,
  },
  clearButton: {
    backgroundColor: '#ffebee',
  },
  clearButtonText: {
    color: '#FF3B30',
    fontWeight: '600',
    marginLeft: 8,
  },
});

export default AlertsScreen;