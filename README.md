# Here are your Instructions

# Smart Parking Mobile App

A React Native mobile application for real-time CCTV parking detection and monitoring system.

## Features

### 📱 Bottom Tab Navigation

- **Dashboard**: System overview, statistics, and recent alerts
- **Video Feed**: Live CCTV monitoring with AI overlays
- **Alerts**: Real-time violation notifications and history
- **Settings**: App configuration and preferences

### 🎥 Video Monitoring

- Stream live video from 8 CCTV locations:
  - AB-1 Parking
  - AB-3 Parking
  - AB-3 Front
  - GymKhana
  - AB-1 Front
  - Aavin
  - Vmart
  - Sigma Block
- Real-time AI detection overlays
- Parking zone classification (Parking vs No-Parking Zone)
- Vehicle tracking with bounding boxes

### 🚨 Smart Alerts

- Real-time violation detection
- Push notifications for parking violations
- Configurable alert thresholds (3-15 seconds)
- Alert filtering and sorting
- Local notification system

### 🎛️ Settings & Configuration

- Notification preferences
- Alert threshold customization
- Data usage controls (WiFi only/WiFi + Cellular)
- Theme selection (Light/Dark/System)
- App data management

## Technology Stack

- **Frontend**: React Native with Expo
- **Navigation**: React Navigation 6
- **Video**: Expo AV
- **Notifications**: Expo Notifications
- **Storage**: AsyncStorage
- **Backend**: FastAPI (existing)
- **AI Detection**: YOLO + CNN + Random Forest

## Backend Integration

Connects to existing FastAPI backend:

- **Base URL**: ``
- **API Endpoints**:
  - `GET /api/videos` - Available video feeds
  - `GET /api/video/{name}` - Video stream
  - `GET /api/violations` - Violation history
  - `POST /api/violations` - Log new violation
  - `POST /api/reset-alerts` - Clear alerts

## Installation & Setup

### Prerequisites

- Vscode
- MongoDB - https://www.mongodb.com/try/download/community
  Download MongoDB and Install it.
- Expo Go App(in Mobile - Playstore) - for Physical Testing

### Setup

1. **Clone this Repo**:

- Open Vscode

```bash
git clone https://github.com/sitikeykarmes/AI-Powered-Real-Time-No-Parking-Zone-Enforcement-System.git
```

### Backend Setup

1. **Navigate to Backend Directory**:

   ```bash
   cd backend
   ```

2. **Install libraries**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Create a 'videos' folder in Backend Folder**:

   ```bash
   mkdir videos
   ```

   - Paste all your test videos in this 'videos' folder

### Development Setup

1. **Navigate to mobile-app directory**:

- Go back to Root Directory and then:
  ```bash
  cd mobile-app
  ```

2. **Install dependencies**:

   ```bash
   npm install
   ```

3. **Go to Windows Search, Open Command Prompt and Type**:

   ```bash
   ipconfig
   ```

   - You will see

   ```bash
   IPv4 Address. . . . . . . . . . . : 172.20.162.71
   ```

   Copy this : 172.20.162.71 (Will be different for your Network)

4. **Find app.json in mobile-app**:

   - Scroll Down, You will see:

   ```bash
   "extra": {
      "backendUrl": "http://192.168.33.6:8001"
    },
   ```

   Just replace the 192.168.33.6 to 172.20.162.71(which you copied from Command Prompt)

5. **Start development server**:

   ```bash
   npx expo start
   ```

6. **Run on device/simulator**:
   - **Physical Device**: Scan QR code with Expo Go app

## Project Structure

```
mobile-app/
├── src/
│   ├── components/
│   │   └── VideoPlayer.js          # Video streaming component
│   ├── navigation/
│   │   └── TabNavigator.js         # Bottom tab navigation
│   ├── screens/
│   │   ├── DashboardScreen.js      # System overview
│   │   ├── VideoFeedScreen.js      # CCTV monitoring
│   │   ├── AlertsScreen.js         # Violations & alerts
│   │   └── SettingsScreen.js       # App preferences
│   ├── services/
│   │   └── api.js                  # Backend API integration
│   └── utils/
│       └── notifications.js        # Notification
├── App.js                          # Main app component
├── app.json                        # Expo configuration
└── package.json                    # Dependencies
```

## API Integration Details

### Video Streaming

- Videos served from backend `/api/video/{videoName}`
- Supports standard MP4 format
- Automatic fallback for missing videos

### Real-time Detection

- Mock detection simulation (2-second intervals)
- Vehicle tracking with violation detection
- Configurable alert thresholds

### Notification System

- Push notifications for violations
- Local notification scheduling
- Badge count management
- Sound and vibration support

## Mobile-Specific Features

### Responsive Design

- Optimized for mobile screens
- Touch-friendly interface
- Native iOS/Android styling

### Performance

- Efficient video streaming
- Background processing
- Memory management
- Battery optimization

### Permissions

- **Camera**: For future AR features
- **Notifications**: Violation alerts
- **Network**: API communication
- **Vibration**: Alert feedback

## Testing

### Web Preview

```bash
npm run web
```

Access at: `http://localhost:8001`

### Device Testing

1. Install Expo Go from App Store/Play Store
2. Scan QR code from `npx expo start`
3. Test all features on physical device

### Simulator Testing

```bash
# iOS (Mac only)
npm run ios

# Android
npm run android
```

## Deployment

### Development Build

```bash
expo build:ios
expo build:android
```

### Production Build

```bash
# iOS
expo build:ios --release-channel production

# Android
expo build:android --release-channel production
```

## Configuration

### Environment Variables

Backend URL configured in `src/services/api.js`:

```javascript
const BACKEND_URL = "https://localhost:8001";
```

### Notification Setup

Configured in `app.json` and `src/utils/notifications.js`

### Video Settings

Default video sources defined in API service

## Troubleshooting

### Common Issues

1. **Video not loading**:

   - Check backend connectivity
   - Verify video file exists
   - Check network permissions

2. **Notifications not working**:

   - Enable app notifications in device settings
   - Check permission status
   - Verify Expo notifications setup

3. **Build errors**:
   - Clear node_modules: `rm -rf node_modules && npm install`
   - Clear Expo cache: `expo r -c`
   - Update Expo CLI: `npm install -g @expo/cli`

### Debug Mode

```bash
expo start --dev-client
```

## Future Enhancements

- [ ] Real-time WebSocket integration
- [ ] AR overlay features
- [ ] Offline mode support
- [ ] Advanced analytics
- [ ] Multi-language support
- [ ] Dark mode optimization

## Support

For technical issues or questions:

- Check backend API status
- Review Expo documentation
- Test on multiple devices
- Check network connectivity

## License

Integrated with existing Smart Parking Detection System.

For Contact: kartikey8025@gmail.com and shambhavi.k2020@gmail.com
