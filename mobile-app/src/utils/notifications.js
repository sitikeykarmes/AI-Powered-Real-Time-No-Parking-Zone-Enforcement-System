import * as Notifications from 'expo-notifications';
import { Alert } from 'react-native';

class NotificationService {
  constructor() {
    this.isEnabled = true;
  }

  async initialize() {
    // Request permissions
    const { status } = await Notifications.requestPermissionsAsync();
    
    if (status !== 'granted') {
      Alert.alert(
        'Notification Permission',
        'Please enable notifications to receive parking violation alerts.',
        [
          { text: 'Cancel' },
          { text: 'Settings', onPress: () => Linking.openSettings() }
        ]
      );
      return false;
    }

    // Configure notification handler
    Notifications.setNotificationHandler({
      handleNotification: async () => ({
        shouldShowAlert: true,
        shouldPlaySound: true,
        shouldSetBadge: true,
      }),
    });

    return true;
  }

  async sendViolationAlert(violation) {
    if (!this.isEnabled) return;

    try {
      await Notifications.scheduleNotificationAsync({
        content: {
          title: '🚨 Parking Violation Detected!',
          body: `Vehicle #${violation.vehicleId} in no-parking zone at ${violation.location} for ${violation.duration.toFixed(1)}s`,
          data: {
            violationId: violation.id,
            vehicleId: violation.vehicleId,
            location: violation.location,
            type: 'violation'
          },
          badge: 1,
        },
        trigger: null, // Show immediately
      });
    } catch (error) {
      console.error('Error sending notification:', error);
    }
  }

  async sendSystemAlert(title, message, data = {}) {
    if (!this.isEnabled) return;

    try {
      await Notifications.scheduleNotificationAsync({
        content: {
          title,
          body: message,
          data: {
            ...data,
            type: 'system'
          },
        },
        trigger: null,
      });
    } catch (error) {
      console.error('Error sending system notification:', error);
    }
  }

  setEnabled(enabled) {
    this.isEnabled = enabled;
  }

  async clearBadge() {
    await Notifications.setBadgeCountAsync(0);
  }

  async clearAll() {
    await Notifications.dismissAllNotificationsAsync();
    await this.clearBadge();
  }
}

export default new NotificationService();