@echo off
echo 🚀 Starting Local Development Environment
echo ========================================

:: Get local IP address on Windows
for /f "tokens=2 delims=:" %%i in ('ipconfig ^| findstr /i "IPv4"') do (
    set LOCAL_IP=%%i
    goto :found
)

:found
:: Remove leading spaces
set LOCAL_IP=%LOCAL_IP: =%

if "%LOCAL_IP%"=="" (
    echo ❌ Could not detect local IP address
    echo Please manually find your local IP and update .env.local files
    pause
    exit /b 1
)

echo 🔍 Detected Local IP: %LOCAL_IP%

:: Update mobile app .env.local with detected IP
echo 📝 Updating mobile app configuration...
echo EXPO_PUBLIC_BACKEND_URL=http://%LOCAL_IP%:8001 > mobile-app\.env.local

echo ✅ Configuration updated
echo.
echo 📋 Next Steps:
echo 1. Start Backend: cd backend ^&^& python server.py
echo 2. Start Mobile App: cd mobile-app ^&^& yarn start
echo 3. Scan QR code with Expo Go app
echo.
echo 🔗 Backend API: http://%LOCAL_IP%:8001/api/
echo 🎥 Videos API: http://%LOCAL_IP%:8001/api/videos

pause