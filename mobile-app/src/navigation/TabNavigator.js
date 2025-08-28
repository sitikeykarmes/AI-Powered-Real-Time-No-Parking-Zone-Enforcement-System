import React from 'react';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { Ionicons } from '@expo/vector-icons';
import { StatusBar } from 'expo-status-bar';

import DashboardScreen from '../screens/DashboardScreen';
import VideoFeedScreen from '../screens/VideoFeedScreen';
import AlertsScreen from '../screens/AlertsScreen';
import SettingsScreen from '../screens/SettingsScreen';

const Tab = createBottomTabNavigator();

const TabNavigator = () => {
  return (
    <>
      <StatusBar style="dark" />
      <Tab.Navigator
        screenOptions={({ route }) => ({
          tabBarIcon: ({ focused, color, size }) => {
            let iconName;

            if (route.name === 'Dashboard') {
              iconName = focused ? 'home' : 'home-outline';
            } else if (route.name === 'Video Feed') {
              iconName = focused ? 'videocam' : 'videocam-outline';
            } else if (route.name === 'Alerts') {
              iconName = focused ? 'warning' : 'warning-outline';
            } else if (route.name === 'Settings') {
              iconName = focused ? 'settings' : 'settings-outline';
            }

            return <Ionicons name={iconName} size={size} color={color} />;
          },
          tabBarActiveTintColor: '#007AFF',
          tabBarInactiveTintColor: 'gray',
          headerStyle: {
            backgroundColor: '#f8f9fa',
          },
          headerTitleStyle: {
            fontWeight: 'bold',
          },
        })}
      >
        <Tab.Screen 
          name="Dashboard" 
          component={DashboardScreen}
          options={{
            title: 'Dashboard',
            headerTitle: 'Smart Parking System'
          }}
        />
        <Tab.Screen 
          name="Video Feed" 
          component={VideoFeedScreen}
          options={{
            title: 'Video Feed',
            headerTitle: 'CCTV Video Feed'
          }}
        />
        <Tab.Screen 
          name="Alerts" 
          component={AlertsScreen}
          options={{
            title: 'Alerts',
            headerTitle: 'Parking Alerts'
          }}
        />
        <Tab.Screen 
          name="Settings" 
          component={SettingsScreen}
          options={{
            title: 'Settings',
            headerTitle: 'App Settings'
          }}
        />
      </Tab.Navigator>
    </>
  );
};

export default TabNavigator;