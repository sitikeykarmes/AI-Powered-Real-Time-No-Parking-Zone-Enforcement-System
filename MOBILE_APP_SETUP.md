# 📱 Smart Parking Mobile App - VSCode Setup & Testing Guide

## 🚀 Quick Start (Mobile Testing with Expo Go)

### **Prerequisites**
1. **Install Expo Go** on your phone:
   - **iOS**: [App Store](https://apps.apple.com/app/expo-go/id982107779)
   - **Android**: [Google Play](https://play.google.com/store/apps/details?id=host.exp.exponent)

2. **Ensure phone and computer are on same WiFi network**

### **Step 1: Open Project in VSCode**

```bash
# Open the mobile app project
code /app/mobile-app
```

### **Step 2: Install Dependencies (if needed)**

```bash
# Navigate to mobile app directory
cd /app/mobile-app

# Install dependencies
npm install

# Install Expo CLI globally (if not installed)
npm install -g @expo/cli
```

### **Step 3: Start Development Server**

```bash
# Clear cache and start Expo
npx expo start --clear

# Alternative: Start with tunnel (for network issues)
npx expo start --tunnel
```

### **Step 4: Connect Your Phone**

1. **Scan QR Code**:
   - Open **Expo Go** app on your phone
   - Tap **"Scan QR Code"**
   - Point camera at QR code in terminal/browser

2. **Alternative Methods**:
   - **iOS**: Scan with Camera app, then tap notification
   - **Android**: Use Expo Go QR scanner

## 🖥️ **Current Status**

✅ **Mobile App Server Running**: `http://localhost:8081`
✅ **Backend API Connected**: `https://surveillance-viewer.preview.emergentagent.com`
✅ **All Dependencies Installed**

## 📱 **Testing the App**

### **What You'll See on Your Phone:**

1. **Dashboard Tab** 📊
   - System status and statistics
   - Recent parking violations
   - CCTV camera overview

2. **Video Feed Tab** 📹
   - Select from 8 CCTV locations:
     - AB-1 Parking, AB-3 Parking, AB-3 Front
     - GymKhana, AB-1 Front, Aavin, Vmart, Sigma Block
   - Real-time AI detection overlays
   - Vehicle tracking with bounding boxes

3. **Alerts Tab** 🚨
   - Real-time violation notifications
   - Filter and sort alerts
   - Push notification setup

4. **Settings Tab** ⚙️
   - Notification preferences
   - Alert threshold configuration
   - App theme and data usage settings

## 🔧 **VSCode Extensions (Recommended)**

Install these VSCode extensions for better development:

```bash
# Essential Extensions
- ES7+ React/Redux/React-Native snippets
- React Native Tools
- Expo Tools
- JavaScript (ES6) code snippets
- Auto Rename Tag
- Prettier - Code formatter
- ESLint
```

## 📂 **Project Structure Overview**

```
mobile-app/
├── 📁 src/
│   ├── 📁 components/
│   │   └── 📄 VideoPlayer.js       # Video streaming component
│   ├── 📁 navigation/
│   │   └── 📄 TabNavigator.js      # Bottom tab navigation
│   ├── 📁 screens/
│   │   ├── 📄 DashboardScreen.js   # Main dashboard
│   │   ├── 📄 VideoFeedScreen.js   # CCTV monitoring
│   │   ├── 📄 AlertsScreen.js      # Alerts management
│   │   └── 📄 SettingsScreen.js    # App settings
│   ├── 📁 services/
│   │   └── 📄 api.js               # Backend API calls
│   └── 📁 utils/
│       └── 📄 notifications.js     # Push notifications
├── 📄 App.js                       # Main app component
├── 📄 app.json                     # Expo configuration
└── 📄 package.json                 # Dependencies
```

## 🧪 **Testing Features**

### **1. Dashboard Testing**
- ✅ Check system status display
- ✅ Verify statistics cards
- ✅ Test refresh functionality
- ✅ View recent alerts

### **2. Video Feed Testing**
- ✅ Select different CCTV locations
- ✅ Test video playback
- ✅ Check detection overlays
- ✅ Verify violation alerts

### **3. Alerts Testing**
- ✅ Filter alerts (All, Today, Recent)
- ✅ Sort by newest/oldest/duration
- ✅ Test notification settings
- ✅ Clear alerts functionality

### **4. Settings Testing**
- ✅ Toggle notification preferences
- ✅ Adjust alert thresholds
- ✅ Change data usage settings
- ✅ Test theme options

## 🔄 **Development Commands**

```bash
# Start development server
npx expo start

# Start with cleared cache
npx expo start --clear

# Start with tunnel (for network issues)
npx expo start --tunnel

# Run on specific platform
npx expo start --ios      # iOS simulator (Mac only)
npx expo start --android  # Android emulator
npx expo start --web      # Web browser

# Install dependencies
npm install

# Update Expo SDK
npx expo install --fix

# Check project health
npx expo doctor
```

## 🐛 **Troubleshooting**

### **QR Code Not Appearing?**
```bash
# Try tunnel mode
npx expo start --tunnel

# Or use local network
npx expo start --lan
```

### **Connection Issues?**
1. Ensure phone and computer on same WiFi
2. Check firewall settings
3. Try tunnel mode: `npx expo start --tunnel`

### **App Not Loading?**
1. Clear Expo cache: `npx expo start --clear`
2. Reload in Expo Go: Shake phone → "Reload"
3. Check terminal for errors

### **Build Errors?**
```bash
# Clear node modules
rm -rf node_modules
npm install

# Clear Expo cache
npx expo start --clear

# Reset Metro cache
npx expo start --reset-cache
```

## 📡 **API Integration Details**

The mobile app connects to your existing backend:

- **Backend URL**: `https://surveillance-viewer.preview.emergentagent.com`
- **API Endpoints**: All `/api/*` routes
- **Video Streaming**: Direct video URLs
- **Real-time Updates**: Polling every 30 seconds
- **Offline Handling**: Graceful degradation

## 🎯 **Testing Checklist**

### **Basic Functionality**
- [ ] App loads successfully in Expo Go
- [ ] All 4 tabs navigate correctly
- [ ] Backend API connection works
- [ ] Video list loads from server

### **Video Features**
- [ ] Can select different CCTV locations
- [ ] Videos play with controls
- [ ] Mock detection overlays appear
- [ ] Violation detection triggers alerts

### **Notifications**
- [ ] Permission request appears
- [ ] Test notifications trigger
- [ ] Notification settings work
- [ ] Badge counts update

### **Responsive Design**
- [ ] UI scales correctly on phone
- [ ] Touch targets are appropriate
- [ ] Text is readable
- [ ] Navigation is smooth

## 🚀 **Next Steps After Testing**

1. **Build for Production**:
   ```bash
   # EAS Build (recommended)
   npm install -g @expo/eas-cli
   eas build --platform ios
   eas build --platform android
   ```

2. **Deploy to Stores**:
   - Configure app signing
   - Upload to App Store/Google Play
   - Submit for review

3. **Add Advanced Features**:
   - Real-time WebSocket integration
   - Offline mode support
   - Advanced analytics
   - Push notification server

## 📞 **Support**

If you encounter any issues:
1. Check the terminal output for errors
2. Verify network connectivity
3. Try clearing cache and restarting
4. Check Expo Go app version

**Happy Testing! 📱✨**