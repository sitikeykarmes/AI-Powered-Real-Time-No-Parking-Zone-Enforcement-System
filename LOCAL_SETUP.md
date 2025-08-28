# Local Development Setup

This guide will help you run the mobile app and backend locally on your PC.

## Prerequisites

1. **Node.js** (v16 or higher)
2. **Python** (v3.8 or higher) 
3. **MongoDB** (running locally or use MongoDB Atlas)
4. **Expo CLI**: `npm install -g @expo/cli`
5. **Expo Go app** on your mobile device

## Step 1: Setup Backend Locally

### 1.1 Install Python Dependencies
```bash
cd backend
pip install -r requirements.txt
```

### 1.2 Configure Local Environment
The backend is already configured to use `.env.local` when available.

Edit `/backend/.env.local` and update:
- `MONGO_URL`: Your local MongoDB URL
- `DB_NAME`: Local database name
- Ensure MongoDB is running on your PC

### 1.3 Add Your Real Videos
Replace the dummy video files in `/backend/videos/` with your real video files:
```bash
# Copy your real videos to:
/backend/videos/ab1.mp4
/backend/videos/ab3.mp4
/backend/videos/ab3_front.mp4
/backend/videos/gymkhana_1.mp4
/backend/videos/ab1_front.mp4
/backend/videos/aavin.mp4
/backend/videos/Vmart_1.mp4
/backend/videos/sigmablock.mp4
```

### 1.4 Start Backend Server
```bash
cd backend
python server.py
```

The backend will start on `http://0.0.0.0:8001`

## Step 2: Setup Mobile App Locally

### 2.1 Install Dependencies
```bash
cd mobile-app
yarn install
```

### 2.2 Configure Local Backend URL

**Find Your Local IP Address:**
- **Windows**: `ipconfig | findstr IPv4`
- **macOS**: `ifconfig | grep inet | grep -v 127.0.0.1`
- **Linux**: `hostname -I`

Edit `/mobile-app/.env.local` and replace `192.168.1.100` with your actual local IP:
```env
EXPO_PUBLIC_BACKEND_URL=http://YOUR_LOCAL_IP:8001
```

**Important**: 
- Don't use `localhost` or `127.0.0.1` - use your actual local network IP
- Make sure your mobile device is on the same WiFi network as your PC

### 2.3 Start Mobile App
```bash
cd mobile-app
yarn start
```

This will start the Expo dev server and show a QR code.

### 2.4 Run on Your Device
1. Install **Expo Go** app on your mobile device
2. Scan the QR code displayed in your terminal
3. The app will load on your device

## Step 3: Testing

1. **Backend**: Visit `http://YOUR_LOCAL_IP:8001/api/` to test API
2. **Videos**: Check `http://YOUR_LOCAL_IP:8001/api/videos` to see available videos
3. **Mobile App**: The app should connect to your local backend and show real videos

## Troubleshooting

### Backend Issues
- **MongoDB Connection**: Ensure MongoDB is running
- **Port in Use**: Change port in `.env.local` if 8001 is occupied
- **Videos Not Loading**: Check video files exist and have proper permissions

### Mobile App Issues
- **Can't Connect**: Verify IP address and both devices on same network
- **Babel Warnings**: These are fixed in the updated configuration
- **expo-av Warnings**: These are fixed by migrating to expo-video

### Fixed Warnings
✅ **Reanimated Plugin**: Now uses `react-native-worklets/plugin`  
✅ **expo-notifications**: Configured for development builds  
✅ **expo-av**: Replaced with `expo-video`  

## Development Workflow

1. **Backend Changes**: Restart `python server.py`
2. **Mobile Changes**: Expo will hot-reload automatically
3. **Environment Changes**: Restart both backend and mobile app

## Production vs Local

- **Local**: Uses `.env.local` files with local IP addresses
- **Production**: Uses `.env.production` files with cloud URLs

The app automatically detects which environment to use based on available config files.