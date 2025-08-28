#!/bin/bash

echo "🚀 Starting Local Development Environment"
echo "========================================"

# Get local IP address
LOCAL_IP=$(hostname -I | awk '{print $1}' 2>/dev/null || ifconfig | grep -E "inet.*broadcast" | awk '{print $2}' 2>/dev/null || ipconfig 2>/dev/null | grep "IPv4" | head -1 | awk '{print $NF}')

if [ -z "$LOCAL_IP" ]; then
    echo "❌ Could not detect local IP address"
    echo "Please manually find your local IP and update .env.local files"
    exit 1
fi

echo "🔍 Detected Local IP: $LOCAL_IP"

# Update mobile app .env.local with detected IP
echo "📝 Updating mobile app configuration..."
echo "EXPO_PUBLIC_BACKEND_URL=http://$LOCAL_IP:8001" > mobile-app/.env.local

echo "✅ Configuration updated"
echo ""
echo "📋 Next Steps:"
echo "1. Start Backend: cd backend && python server.py"
echo "2. Start Mobile App: cd mobile-app && yarn start"
echo "3. Scan QR code with Expo Go app"
echo ""
echo "🔗 Backend API: http://$LOCAL_IP:8001/api/"
echo "🎥 Videos API: http://$LOCAL_IP:8001/api/videos"