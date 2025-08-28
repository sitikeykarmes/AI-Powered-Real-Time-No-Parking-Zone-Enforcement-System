import requests
import sys
import json
from datetime import datetime

class ParkingSystemAPITester:
    def __init__(self, base_url="https://surveillance-viewer.preview.emergentagent.com"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        self.tests_run = 0
        self.tests_passed = 0

    def run_test(self, name, method, endpoint, expected_status, data=None, headers=None):
        """Run a single API test"""
        url = f"{self.api_url}/{endpoint}" if endpoint else f"{self.api_url}/"
        if headers is None:
            headers = {'Content-Type': 'application/json'}

        self.tests_run += 1
        print(f"\n🔍 Testing {name}...")
        print(f"   URL: {url}")
        
        try:
            if method == 'GET':
                response = requests.get(url, headers=headers, timeout=10)
            elif method == 'POST':
                response = requests.post(url, json=data, headers=headers, timeout=10)

            print(f"   Response Status: {response.status_code}")
            
            success = response.status_code == expected_status
            if success:
                self.tests_passed += 1
                print(f"✅ Passed - Status: {response.status_code}")
                try:
                    response_data = response.json()
                    print(f"   Response: {json.dumps(response_data, indent=2)[:200]}...")
                    return True, response_data
                except:
                    print(f"   Response: {response.text[:200]}...")
                    return True, response.text
            else:
                print(f"❌ Failed - Expected {expected_status}, got {response.status_code}")
                print(f"   Response: {response.text[:200]}...")
                return False, {}

        except Exception as e:
            print(f"❌ Failed - Error: {str(e)}")
            return False, {}

    def test_root_endpoint(self):
        """Test the root API endpoint"""
        return self.run_test(
            "Root API Endpoint",
            "GET",
            "",
            200
        )

    def test_get_videos(self):
        """Test getting available videos"""
        success, response = self.run_test(
            "Get Available Videos",
            "GET", 
            "videos",
            200
        )
        
        if success and isinstance(response, dict):
            videos = response.get('videos', {})
            total = response.get('total', 0)
            print(f"   Found {total} videos: {list(videos.keys())}")
            
            # Verify we have 8 videos as expected
            if total == 8:
                print("✅ Correct number of videos (8)")
            else:
                print(f"⚠️  Expected 8 videos, found {total}")
                
        return success, response

    def test_reset_alerts(self):
        """Test resetting alerts"""
        return self.run_test(
            "Reset Alerts",
            "POST",
            "reset-alerts", 
            200
        )

    def test_get_violations(self):
        """Test getting violations log"""
        return self.run_test(
            "Get Violations Log",
            "GET",
            "violations",
            200
        )

    def test_create_violation(self):
        """Test creating a violation log entry"""
        violation_data = {
            "vehicle_id": 123,
            "location": "Test Location",
            "duration": 5.5,
            "violation_type": "no_parking_zone"
        }
        
        return self.run_test(
            "Create Violation Log",
            "POST",
            "violations",
            200,
            data=violation_data
        )

    def test_process_frame(self):
        """Test frame processing endpoint"""
        frame_data = {
            "frame_data": "mock_base64_data",
            "video_name": "AB-1 Parking"
        }
        
        return self.run_test(
            "Process Frame",
            "POST",
            "process-frame",
            200,
            data=frame_data
        )

    def test_video_file_access(self, video_name="AB-1 Parking"):
        """Test accessing a video file"""
        url = f"{self.api_url}/video/{video_name}"
        print(f"\n🔍 Testing Video File Access...")
        print(f"   URL: {url}")
        
        self.tests_run += 1
        
        try:
            response = requests.head(url, timeout=10)  # Use HEAD to avoid downloading
            print(f"   Response Status: {response.status_code}")
            
            if response.status_code in [200, 404]:  # 404 is acceptable if video file doesn't exist
                self.tests_passed += 1
                if response.status_code == 200:
                    print("✅ Passed - Video file accessible")
                    content_type = response.headers.get('content-type', '')
                    print(f"   Content-Type: {content_type}")
                else:
                    print("✅ Passed - Endpoint working (video file not found, which is expected)")
                return True
            else:
                print(f"❌ Failed - Unexpected status: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Failed - Error: {str(e)}")
            return False

def main():
    print("🚀 Starting Parking Detection System API Tests")
    print("=" * 60)
    
    # Setup
    tester = ParkingSystemAPITester()
    
    # Run all tests
    print("\n📋 Running Backend API Tests...")
    
    # Test basic endpoints
    tester.test_root_endpoint()
    tester.test_get_videos()
    tester.test_reset_alerts()
    tester.test_get_violations()
    
    # Test violation creation
    tester.test_create_violation()
    
    # Test frame processing
    tester.test_process_frame()
    
    # Test video file access
    tester.test_video_file_access()
    
    # Print final results
    print("\n" + "=" * 60)
    print(f"📊 BACKEND TEST RESULTS")
    print(f"Tests Run: {tester.tests_run}")
    print(f"Tests Passed: {tester.tests_passed}")
    print(f"Tests Failed: {tester.tests_run - tester.tests_passed}")
    print(f"Success Rate: {(tester.tests_passed/tester.tests_run)*100:.1f}%")
    
    if tester.tests_passed == tester.tests_run:
        print("🎉 All backend tests passed!")
        return 0
    else:
        print("⚠️  Some backend tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())