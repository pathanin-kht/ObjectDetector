import torch
import cv2
import numpy as np
from time import time
from datetime import datetime
import os
from collections import deque

class ModernObjectDetector:
    def __init__(self):
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
        self.setup_model_parameters()
        self.setup_display_settings()
        self.setup_recording()
        
    def setup_model_parameters(self):
        self.model.conf = 0.45
        self.model.iou = 0.45
        self.model.classes = None
        self.model.agnostic = False
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
    def setup_display_settings(self):
        self.colors = {
            'text_color': (240, 240, 240),
            'bg_color': (41, 41, 41),
            'accent_color': (0, 255, 0),
            'alert_color': (0, 0, 255)
        }
        self.fps_history = deque(maxlen=30)
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        
    def setup_recording(self):
        self.is_recording = False
        self.out = None
        self.recording_start_time = None
        
    def start_camera(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
    def create_overlay(self, img, fps, detections):
        overlay = img.copy()
        
        cv2.rectangle(overlay, (0, 0), (img.shape[1], 60), self.colors['bg_color'], -1)
        cv2.rectangle(overlay, (0, img.shape[0]-120), (300, img.shape[0]), self.colors['bg_color'], -1)
        
        fps_text = f'FPS: {fps:.1f}'
        cv2.putText(overlay, fps_text, (20, 40), self.font, 1, self.colors['accent_color'], 2)
        
        time_str = datetime.now().strftime('%H:%M:%S')
        cv2.putText(overlay, f'{time_str}', (img.shape[1]-200, 40), self.font, 1, self.colors['text_color'], 2)
        
        if self.is_recording:
            cv2.circle(overlay, (img.shape[1]-220, 35), 8, (0, 0, 255), -1)
            
        y_pos = img.shape[0] - 100
        cv2.putText(overlay, 'Detected Objects:', (20, y_pos), self.font, 0.8, self.colors['text_color'], 2)
        
        if detections:
            for label, count in detections.items():
                y_pos += 30
                cv2.putText(overlay, f'{label}: {count}', (30, y_pos), self.font, 0.7, self.colors['accent_color'], 2)
        else:
            y_pos += 30
            cv2.putText(overlay, 'No objects detected', (30, y_pos), self.font, 0.7, self.colors['accent_color'], 2)
        
        cv2.addWeighted(overlay, 0.82, img, 0.18, 0, img)
        return img
        
    def toggle_recording(self, frame):
        if not self.is_recording:
            filename = f'detection_{datetime.now().strftime("%Y%m%d_%H%M%S")}.mp4'
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            frame_size = (frame.shape[1], frame.shape[0])
            self.out = cv2.VideoWriter(filename, fourcc, 20.0, frame_size)
            self.is_recording = True
            self.recording_start_time = time()
        else:
            self.out.release()
            self.is_recording = False
            self.recording_start_time = None
            
    def process_frame(self, frame):
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = self.model(img_rgb)
        
        detections = {}
        for *box, conf, cls in results.xyxy[0].cpu().numpy():
            label = results.names[int(cls)].split()[0] 
            
            if label in detections:
                detections[label] += 1
            else:
                detections[label] = 1
    
        return results, detections
        
    def run(self):
        self.start_camera()
        prev_time = time()
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            current_time = time()
            fps = 1 / (current_time - prev_time)
            self.fps_history.append(fps)
            avg_fps = sum(self.fps_history) / len(self.fps_history)
            prev_time = current_time
            
            results, detections = self.process_frame(frame)
            
            img_with_boxes = np.squeeze(results.render())
            img_bgr = cv2.cvtColor(img_with_boxes, cv2.COLOR_RGB2BGR)
            
            final_image = self.create_overlay(img_bgr, avg_fps, detections)
            
            if self.is_recording:
                self.out.write(final_image)
            
            cv2.imshow("Modern Object Detection", final_image)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.toggle_recording(final_image)
            elif key == ord('c'):
                filename = f'capture_{datetime.now().strftime("%Y%m%d_%H%M%S")}.jpg'
                cv2.imwrite(filename, final_image)
                
        if self.is_recording:
            self.out.release()
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = ModernObjectDetector()
    detector.run()
