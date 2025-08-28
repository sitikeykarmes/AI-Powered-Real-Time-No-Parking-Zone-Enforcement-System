import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  ScrollView,
  StyleSheet,
  TouchableOpacity,
  Switch,
  Alert,
  Linking,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import * as Notifications from 'expo-notifications';
import AsyncStorage from '@react-native-async-storage/async-storage';

const SettingRow = ({ 
  title, 
  subtitle, 
  icon, 
  onPress, 
  rightComponent, 
  showChevron = true 
}) => (
  <TouchableOpacity
    style={styles.settingRow}
    onPress={onPress}
    disabled={!onPress}
  >
    <View style={styles.settingLeft}>
      <Ionicons name={icon} size={24} color="#007AFF" style={styles.settingIcon} />
      <View style={styles.settingText}>
        <Text style={styles.settingTitle}>{title}</Text>
        {subtitle && <Text style={styles.settingSubtitle}>{subtitle}</Text>}
      </View>
    </View>
    <View style={styles.settingRight}>
      {rightComponent}
      {showChevron && onPress && (
        <Ionicons name="chevron-forward" size={20} color="#c7c7cc" />
      )}
    </View>
  </TouchableOpacity>
);

const SettingsSection = ({ title, children }) => (
  <View style={styles.section}>
    <Text style={styles.sectionTitle}>{title}</Text>
    <View style={styles.sectionContent}>
      {children}
    </View>
  </View>
);

const SettingsScreen = () => {
  const [settings, setSettings] = useState({
    notifications: true,
    alertSound: true,
    vibration: true,
    autoRefresh: true,
    alertThreshold: 5,
    dataUsage: 'wifi_only',
    theme: 'system',
  });

  // Load settings from storage
  useEffect(() => {
    loadSettings();
  }, []);

  const loadSettings = async () => {
    try {
      const savedSettings = await AsyncStorage.getItem('app_settings');
      if (savedSettings) {
        setSettings(JSON.parse(savedSettings));
      }
    } catch (error) {
      console.error('Error loading settings:', error);
    }
  };

  const saveSettings = async (newSettings) => {
    try {
      await AsyncStorage.setItem('app_settings', JSON.stringify(newSettings));
      setSettings(newSettings);
    } catch (error) {
      console.error('Error saving settings:', error);
    }
  };

  const updateSetting = (key, value) => {
    const newSettings = { ...settings, [key]: value };
    saveSettings(newSettings);
  };

  const handleNotificationToggle = async (value) => {
    if (value) {
      const { status } = await Notifications.requestPermissionsAsync();
      if (status !== 'granted') {
        Alert.alert(
          'Permission Required',
          'Please enable notifications in your device settings to receive alerts.',
          [
            { text: 'Cancel' },
            { text: 'Open Settings', onPress: () => Linking.openSettings() }
          ]
        );
        return;
      }
    }
    updateSetting('notifications', value);
  };

  const showAlertThresholdPicker = () => {
    Alert.alert(
      'Alert Threshold',
      'Set the number of seconds a vehicle must be in a no-parking zone before triggering an alert.',
      [
        { text: '3 seconds', onPress: () => updateSetting('alertThreshold', 3) },
        { text: '5 seconds', onPress: () => updateSetting('alertThreshold', 5) },
        { text: '10 seconds', onPress: () => updateSetting('alertThreshold', 10) },
        { text: '15 seconds', onPress: () => updateSetting('alertThreshold', 15) },
        { text: 'Cancel', style: 'cancel' },
      ]
    );
  };

  const showDataUsagePicker = () => {
    Alert.alert(
      'Data Usage',
      'Choose when to load video feeds.',
      [
        { text: 'WiFi Only', onPress: () => updateSetting('dataUsage', 'wifi_only') },
        { text: 'WiFi + Cellular', onPress: () => updateSetting('dataUsage', 'all') },
        { text: 'Cancel', style: 'cancel' },
      ]
    );
  };

  const showThemePicker = () => {
    Alert.alert(
      'App Theme',
      'Choose your preferred app appearance.',
      [
        { text: 'Light', onPress: () => updateSetting('theme', 'light') },
        { text: 'Dark', onPress: () => updateSetting('theme', 'dark') },
        { text: 'System', onPress: () => updateSetting('theme', 'system') },
        { text: 'Cancel', style: 'cancel' },
      ]
    );
  };

  const clearAppData = () => {
    Alert.alert(
      'Clear App Data',
      'This will reset all app settings and clear cached data. Are you sure?',
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Clear Data',
          style: 'destructive',
          onPress: async () => {
            try {
              await AsyncStorage.clear();
              setSettings({
                notifications: true,
                alertSound: true,
                vibration: true,
                autoRefresh: true,
                alertThreshold: 5,
                dataUsage: 'wifi_only',
                theme: 'system',
              });
              Alert.alert('Success', 'App data has been cleared.');
            } catch (error) {
              Alert.alert('Error', 'Failed to clear app data.');
            }
          },
        },
      ]
    );
  };

  const showAbout = () => {
    Alert.alert(
      'About Smart Parking System',
      'Version 1.0.0\n\nA comprehensive parking detection and monitoring system using AI-powered computer vision.\n\nDeveloped with React Native and Expo.',
      [{ text: 'OK' }]
    );
  };

  return (
    <ScrollView style={styles.container}>
      {/* Notifications */}
      <SettingsSection title="Notifications">
        <SettingRow
          title="Push Notifications"
          subtitle="Receive alerts for parking violations"
          icon="notifications"
          rightComponent={
            <Switch
              value={settings.notifications}
              onValueChange={handleNotificationToggle}
              trackColor={{ false: '#f2f2f7', true: '#007AFF' }}
            />
          }
          showChevron={false}
        />
        <SettingRow
          title="Alert Sound"
          subtitle="Play sound for new violations"
          icon="volume-high"
          rightComponent={
            <Switch
              value={settings.alertSound}
              onValueChange={(value) => updateSetting('alertSound', value)}
              trackColor={{ false: '#f2f2f7', true: '#007AFF' }}
            />
          }
          showChevron={false}
        />
        <SettingRow
          title="Vibration"
          subtitle="Vibrate for new violations"
          icon="phone-portrait"
          rightComponent={
            <Switch
              value={settings.vibration}
              onValueChange={(value) => updateSetting('vibration', value)}
              trackColor={{ false: '#f2f2f7', true: '#007AFF' }}
            />
          }
          showChevron={false}
        />
      </SettingsSection>

      {/* Detection Settings */}
      <SettingsSection title="Detection Settings">
        <SettingRow
          title="Alert Threshold"
          subtitle={`${settings.alertThreshold} seconds`}
          icon="timer"
          onPress={showAlertThresholdPicker}
        />
        <SettingRow
          title="Auto Refresh"
          subtitle="Automatically refresh feeds"
          icon="refresh"
          rightComponent={
            <Switch
              value={settings.autoRefresh}
              onValueChange={(value) => updateSetting('autoRefresh', value)}
              trackColor={{ false: '#f2f2f7', true: '#007AFF' }}
            />
          }
          showChevron={false}
        />
      </SettingsSection>

      {/* Data & Performance */}
      <SettingsSection title="Data & Performance">
        <SettingRow
          title="Data Usage"
          subtitle={settings.dataUsage === 'wifi_only' ? 'WiFi Only' : 'WiFi + Cellular'}
          icon="cellular"
          onPress={showDataUsagePicker}
        />
      </SettingsSection>

      {/* Appearance */}
      <SettingsSection title="Appearance">
        <SettingRow
          title="Theme"
          subtitle={settings.theme === 'system' ? 'System' : 
                   settings.theme === 'light' ? 'Light' : 'Dark'}
          icon="color-palette"
          onPress={showThemePicker}
        />
      </SettingsSection>

      {/* System Information */}
      <SettingsSection title="System Information">
        <SettingRow
          title="Detection Engine"
          subtitle="YOLO + CNN + Random Forest"
          icon="hardware-chip"
          rightComponent={
            <View style={styles.statusBadge}>
              <Text style={styles.statusText}>Active</Text>
            </View>
          }
          showChevron={false}
        />
        <SettingRow
          title="Backend Connection"
          subtitle="surveillance-viewer.preview.emergentagent.com"
          icon="cloud"
          rightComponent={
            <View style={[styles.statusBadge, styles.onlineStatus]}>
              <Text style={[styles.statusText, { color: 'white' }]}>Online</Text>
            </View>
          }
          showChevron={false}
        />
      </SettingsSection>

      {/* Support & Info */}
      <SettingsSection title="Support & Information">
        <SettingRow
          title="Help & Support"
          subtitle="Get help with the app"
          icon="help-circle"
          onPress={() => Alert.alert('Help', 'Contact support at support@example.com')}
        />
        <SettingRow
          title="Privacy Policy"
          subtitle="View our privacy policy"
          icon="shield-checkmark"
          onPress={() => Alert.alert('Privacy Policy', 'Privacy policy information would be shown here.')}
        />
        <SettingRow
          title="About"
          subtitle="App version and information"
          icon="information-circle"
          onPress={showAbout}
        />
      </SettingsSection>

      {/* Advanced */}
      <SettingsSection title="Advanced">
        <SettingRow
          title="Clear App Data"
          subtitle="Reset settings and clear cache"
          icon="trash"
          onPress={clearAppData}
          rightComponent={null}
          showChevron={false}
        />
      </SettingsSection>

      {/* Footer */}
      <View style={styles.footer}>
        <Text style={styles.footerText}>
          Smart Parking Detection System
        </Text>
        <Text style={styles.footerSubtext}>
          Powered by AI Computer Vision
        </Text>
      </View>
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f8f9fa',
  },
  section: {
    marginTop: 32,
  },
  sectionTitle: {
    fontSize: 14,
    fontWeight: '600',
    color: '#8e8e93',
    textTransform: 'uppercase',
    letterSpacing: 0.5,
    marginHorizontal: 16,
    marginBottom: 8,
  },
  sectionContent: {
    backgroundColor: 'white',
    marginHorizontal: 16,
    borderRadius: 12,
    overflow: 'hidden',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  settingRow: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 16,
    borderBottomWidth: 1,
    borderBottomColor: '#f2f2f7',
  },
  settingLeft: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
  },
  settingIcon: {
    marginRight: 16,
  },
  settingText: {
    flex: 1,
  },
  settingTitle: {
    fontSize: 16,
    color: '#1c1c1e',
    fontWeight: '400',
  },
  settingSubtitle: {
    fontSize: 14,
    color: '#8e8e93',
    marginTop: 2,
  },
  settingRight: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  statusBadge: {
    paddingHorizontal: 8,
    paddingVertical: 4,
    backgroundColor: '#f2f2f7',
    borderRadius: 6,
    marginRight: 8,
  },
  onlineStatus: {
    backgroundColor: '#34C759',
  },
  statusText: {
    fontSize: 12,
    fontWeight: '600',
    color: '#48484a',
  },
  footer: {
    padding: 32,
    alignItems: 'center',
  },
  footerText: {
    fontSize: 16,
    fontWeight: '600',
    color: '#1c1c1e',
  },
  footerSubtext: {
    fontSize: 14,
    color: '#8e8e93',
    marginTop: 4,
  },
});

export default SettingsScreen;