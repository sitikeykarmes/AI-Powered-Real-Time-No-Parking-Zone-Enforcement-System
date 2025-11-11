#parking_detection.py
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO
import time
import os
import pickle
from datetime import datetime
from typing import List, Dict, Any, Optional
from sklearn.ensemble import RandomForestClassifier


class CNNFeatureExtractor(nn.Module):
    """CNN Model for deep feature extraction"""
    def __init__(self):
        super(CNNFeatureExtractor, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 16 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, 6)  # 6 additional deep features
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc_layers(x)
        return x


def calculate_iou(box1, box2):
    """
    Calculate intersection over union between two bounding boxes
    box format: [x1, y1, x2, y2]
    """
    # Input validation
    if len(box1) != 4 or len(box2) != 4:
        return 0.0
    
    # Ensure boxes are valid (x2 > x1, y2 > y1)
    if box1[2] <= box1[0] or box1[3] <= box1[1] or box2[2] <= box2[0] or box2[3] <= box2[1]:
        return 0.0
    
    # Determine coordinates of intersection
    x1_intr = max(box1[0], box2[0])
    y1_intr = max(box1[1], box2[1])
    x2_intr = min(box1[2], box2[2])
    y2_intr = min(box1[3], box2[3])
    
    # Calculate area of intersection
    w_intr = max(0, x2_intr - x1_intr)
    h_intr = max(0, y2_intr - y1_intr)
    area_intr = w_intr * h_intr
    
    # Calculate area of both bounding boxes
    area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # Calculate union area
    area_union = area_box1 + area_box2 - area_intr
    
    # Avoid division by zero
    if area_union == 0:
        return 0.0
    
    # Calculate IoU
    iou = area_intr / float(area_union)
    return iou


class VehicleTracker:
    """Vehicle tracker class to maintain identities of vehicles across frames"""
    def __init__(self, iou_threshold=0.5):
        self.tracked_vehicles = []  # List of [box, vehicle_id, first_detection_time, continuous_detection_time]
        self.next_id = 0
        self.iou_threshold = iou_threshold
    
    def update(self, current_boxes, current_time, is_no_parking_zone):
        if not current_boxes:
            # If no vehicles detected, reset continuous detection time for all tracked vehicles
            for vehicle in self.tracked_vehicles:
                if is_no_parking_zone:
                    # Only reset if we're in a no parking zone, otherwise keep timing
                    vehicle[3] = 0
            return []
        
        updated_vehicles = []
        matched_indices = set()
        
        # Match current detections with tracked vehicles
        for i, vehicle in enumerate(self.tracked_vehicles):
            tracked_box, vehicle_id, first_detection_time, continuous_detection_time = vehicle
            
            best_match_idx = -1
            best_iou = self.iou_threshold
            
            for j, current_box in enumerate(current_boxes):
                if j in matched_indices:
                    continue
                
                iou = calculate_iou(tracked_box, current_box)
                if iou > best_iou:
                    best_iou = iou
                    best_match_idx = j
            
            if best_match_idx >= 0:
                # Update tracked vehicle with new position
                matched_indices.add(best_match_idx)
                updated_box = current_boxes[best_match_idx]
                
                # Update continuous detection time if in no-parking zone
                if is_no_parking_zone:
                    continuous_detection_time += current_time - first_detection_time
                    first_detection_time = current_time
                else:
                    # Reset continuous detection time if we're in a parking zone
                    continuous_detection_time = 0
                    first_detection_time = current_time
                
                updated_vehicles.append([updated_box, vehicle_id, first_detection_time, continuous_detection_time])
        
        # Add new detections
        for j, current_box in enumerate(current_boxes):
            if j not in matched_indices:
                vehicle_id = self.next_id
                self.next_id += 1
                continuous_detection_time = 0  # Start with 0 time for new vehicles
                updated_vehicles.append([current_box, vehicle_id, current_time, continuous_detection_time])
        
        self.tracked_vehicles = updated_vehicles
        return self.tracked_vehicles


class ParkingDetectionSystem:
    """Main parking detection system with full functionality"""
    
    def __init__(self, model_path: str, csv_path: Optional[str] = None):
        """
        Initialize the parking detection system
        
        Args:
            model_path: Path to the trained classifier model (.pkl file)
            csv_path: Path to training CSV (optional, for training new model)
        """
        # Load the YOLO model
        self.yolo_model = YOLO("yolov8n.pt")
        
        # Define object categories and weights (matching gymkhana_alert)
        self.object_weights = {
            'car': 5,
            'motorcycle': 6,
            'parking meter': 6,
            'bus': 5,
            'truck': 5,
            'bicycle': 3,
            'road': -2,
            'person': -1,
            'building': -0.5,
            'footpath': -0.5,
            'sign': -0.25,
            'traffic light': -0.5,
            'fire hydrant': -1,
            'stop sign': -1,
            'parking area': 5,
            'no-parking sign': -3
        }
        
        # Initialize device and CNN model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cnn_model = CNNFeatureExtractor().to(self.device)
        self.cnn_model.eval()
        
        # Image transform for CNN input
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])
        
        # Vehicle tracker
        self.tracker = VehicleTracker(iou_threshold=0.3)
        
        # Persistent alerts
        self.persistent_alerts = {}
        
        # Load or create classifier model
        self.model_path = model_path
        self.csv_path = csv_path
        self.classifier_model = self._load_model()
    
    def _load_model(self):
        """Load classifier model or train a new one"""
        if os.path.exists(self.model_path):
            print(f"Loading existing model from {self.model_path}")
            with open(self.model_path, 'rb') as file:
                return pickle.load(file)
        
        elif self.csv_path and os.path.exists(self.csv_path):
            print(f"Training new model from {self.csv_path}")
            df = pd.read_csv(self.csv_path)
            
            # Prepare data for training
            feature_cols = ['parking_score', 'area_ratio', 'object_density', 'mean_hue', 
                           'mean_saturation', 'mean_value'] + [f'cnn_feature_{i}' for i in range(6)]
            X = df[feature_cols]
            y = df['label']
            
            # Train a RandomForestClassifier
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X, y)
            
            # Save the model for future use
            with open(self.model_path, 'wb') as file:
                pickle.dump(model, file)
            
            print(f"Model trained and saved to {self.model_path}")
            return model
        
        else:
            raise FileNotFoundError(
                f"Neither model file {self.model_path} nor CSV file {self.csv_path} found."
            )
    
    def extract_features_from_frame(self, frame):
        """Extract all features from a video frame"""
        # YOLO detection
        results = self.yolo_model(frame)
        detected_objects = [self.yolo_model.names[int(box.cls)] for box in results[0].boxes]

        parking_score = 0
        for obj in detected_objects:
            parking_score += self.object_weights.get(obj, 0)

        # Feature Engineering
        img_area = frame.shape[0] * frame.shape[1]

        vehicle_area = 0
        for box in results[0].boxes:
            if self.yolo_model.names[int(box.cls)] in ['car', 'motorcycle', 'bus', 'truck', 'bicycle']:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                vehicle_area += (x2 - x1) * (y2 - y1)

        area_ratio = vehicle_area / img_area if img_area > 0 else 0
        object_density = len(detected_objects) / img_area if img_area > 0 else 0

        # Color Features
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mean_hue = np.mean(hsv[:, :, 0])
        mean_saturation = np.mean(hsv[:, :, 1])
        mean_value = np.mean(hsv[:, :, 2])

        # CNN Deep Feature Extraction
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(rgb_frame)
        image_tensor = self.transform(image_pil).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            cnn_features = self.cnn_model(image_tensor).cpu().numpy().flatten()

        return [parking_score, area_ratio, object_density, mean_hue, mean_saturation, mean_value] + list(cnn_features)
    
    def predict_from_frame(self, frame):
        """Predict parking zone from a frame"""
        features = self.extract_features_from_frame(frame)
        
        # Create column names for all features
        column_names = ['parking_score', 'area_ratio', 'object_density', 'mean_hue', 
                       'mean_saturation', 'mean_value'] + [f'cnn_feature_{i}' for i in range(6)]
        
        # Create DataFrame with features
        input_data = pd.DataFrame([features], columns=column_names)
        
        # Make prediction
        prediction = self.classifier_model.predict(input_data)[0]
        return "Parking Zone" if prediction == 0 else "No Parking Zone"
    
    def get_vehicle_detections(self, frame):
        """Extract vehicle detections using YOLO"""
        results = self.yolo_model(frame)
        vehicle_boxes = []
        vehicle_info = []
        
        for box in results[0].boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = self.yolo_model.names[cls]
            
            # Only track vehicles with high confidence
            if class_name in ['car', 'motorcycle', 'bus', 'truck', 'bicycle'] and conf > 0.5:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                # Validate bounding box coordinates
                if x2 > x1 and y2 > y1:
                    vehicle_boxes.append([x1, y1, x2, y2])
                    vehicle_info.append({
                        'class': class_name,
                        'confidence': conf
                    })
        
        return vehicle_boxes, vehicle_info
    
    def process_frame(self, frame, alert_threshold_seconds=5):
        """
        Process a single frame and return detection results
        
        Args:
            frame: Input video frame
            alert_threshold_seconds: Time threshold for generating alerts
            
        Returns:
            Dictionary containing prediction, vehicles, and alerts
        """
        current_time = time.time()
        
        # Predict zone
        prediction = self.predict_from_frame(frame)
        
        # Get vehicle detections
        vehicle_boxes, vehicle_info = self.get_vehicle_detections(frame)
        
        # Update vehicle tracking
        is_no_parking_zone = (prediction == "No Parking Zone")
        tracked_vehicles = self.tracker.update(vehicle_boxes, current_time, is_no_parking_zone)
        
        # Process alerts
        new_alerts = []
        for vehicle_box, vehicle_id, first_detection_time, continuous_detection_time in tracked_vehicles:
            if is_no_parking_zone and continuous_detection_time > alert_threshold_seconds:
                if vehicle_id not in self.persistent_alerts:
                    alert_text = f"VIOLATION: Vehicle #{vehicle_id} in no-parking zone for {continuous_detection_time:.1f}s"
                    self.persistent_alerts[vehicle_id] = {
                        'text': alert_text,
                        'bbox': vehicle_box,
                        'timestamp': current_time,
                        'duration': continuous_detection_time,
                        'vehicle_id': vehicle_id
                    }
                    new_alerts.append(self.persistent_alerts[vehicle_id])
                else:
                    # Update existing alert
                    self.persistent_alerts[vehicle_id]['duration'] = continuous_detection_time
                    self.persistent_alerts[vehicle_id]['text'] = f"VIOLATION: Vehicle #{vehicle_id} in no-parking zone for {continuous_detection_time:.1f}s"
        
        # Prepare response with all tracked vehicles
        vehicles_data = []
        for i, (vehicle_box, vehicle_id, first_detection_time, continuous_detection_time) in enumerate(tracked_vehicles):
            # Get vehicle info if available
            v_info = vehicle_info[i] if i < len(vehicle_info) else {'class': 'vehicle', 'confidence': 0.8}
            
            # Determine status
            status = 'normal'
            box_color = 'green'
            
            if is_no_parking_zone:
                if continuous_detection_time > alert_threshold_seconds:
                    status = 'violation'
                    box_color = 'red'
                elif continuous_detection_time > alert_threshold_seconds / 2:
                    status = 'warning'
                    box_color = 'yellow'
                else:
                    status = 'normal'
                    box_color = 'green'
            else:
                box_color = 'blue'
            
            vehicles_data.append({
                'id': vehicle_id,
                'bbox': vehicle_box,
                'class': v_info['class'],
                'confidence': float(v_info['confidence']),
                'duration': continuous_detection_time,
                'status': status,
                'color': box_color
            })
        
        # Prepare final response
        response = {
            'prediction': prediction,
            'is_no_parking_zone': is_no_parking_zone,
            'vehicles': vehicles_data,
            'alerts': list(self.persistent_alerts.values()),
            'new_alerts': new_alerts,
            'timestamp': current_time,
            'total_violations': len(self.persistent_alerts)
        }
        
        return response
    
    def reset_alerts(self):
        """Reset all persistent alerts"""
        self.persistent_alerts.clear()
        print("All alerts have been reset")
    
    def get_statistics(self):
        """Get current statistics"""
        return {
            'total_vehicles_tracked': self.tracker.next_id,
            'active_violations': len(self.persistent_alerts),
            'tracked_vehicles_count': len(self.tracker.tracked_vehicles)
        }


# Example usage
if __name__ == "__main__":
    # Initialize the system with your model path
    model_path = r"C:\VIT\3rd-Yr\6th-Sem\E2-Deep Learning\Project\ProjectDraft1\ProjectNoParking\parking_model_train22.pkl"
    csv_path = r"C:\VIT\3rd-Yr\6th-Sem\E2-Deep Learning\Project\ProjectDraft1\ProjectNoParking\yolo_features_train22.csv"
    
    try:
        system = ParkingDetectionSystem(model_path, csv_path)
        
        # Example: Process a single frame
        # cap = cv2.VideoCapture(0)  # or video file path
        # ret, frame = cap.read()
        # if ret:
        #     result = system.process_frame(frame, alert_threshold_seconds=5)
        #     print(result)
        
        print("Parking Detection System initialized successfully!")
        print(f"Using device: {system.device}")
        
    except Exception as e:
        print(f"Error initializing system: {e}")