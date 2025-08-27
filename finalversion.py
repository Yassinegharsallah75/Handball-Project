import cv2
import numpy as np
import torch
from ultralytics import YOLO
import math
from collections import deque
import time

class HandballVelocityAnalyzer:
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        
        self.model = YOLO('yolov8s.pt')  
        
        
        self.player_conf_threshold = 0.5
        self.ball_conf_threshold = 0.3
        
        
        self.ball_positions = deque(maxlen=30)
        self.shot_detected = False
        self.shot_frame = None
        self.shot_velocity = None
        self.goal_detected = False
        self.goal_frame = None
        self.shooter_position = None
        
        
        self.goal_area = None  
        self.goal_detection_frames = 5  
        
        
        self.pixels_per_meter = self.width / 20.0
        
        
        self.output_video = None
        self.setup_output()
        
        
        self.ball_detection_history = deque(maxlen=5)
        self.player_detection_history = deque(maxlen=5)
    
    def setup_output(self):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_path = self.video_path.replace('.mp4', '_analyzed.mp4')
        self.output_video = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))
    
    def detect_goal_area(self, frame):
        """Attempt to detect goal area based on court lines and player positions"""
        if self.goal_area is not None:
            return self.goal_area
        
        
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=100, maxLineGap=10)
        
        goal_candidates = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                if abs(x2 - x1) < 20 and abs(y2 - y1) > 100:
                    goal_candidates.append((x1, y1, x2, y2))
        
        
        if goal_candidates:
            
            goal_candidate = max(goal_candidates, key=lambda x: x[0])
            x1, y1, x2, y2 = goal_candidate
            
            
            goal_width = 200  
            goal_height = 300  
            self.goal_area = (max(0, x1 - goal_width//2), 
                             min(self.height, min(y1, y2)), 
                             min(self.width, x1 + goal_width//2), 
                             min(self.height, max(y1, y2) + goal_height))
            
            print(f"Goal area detected: {self.goal_area}")
        
        return self.goal_area
    
    def detect_players_and_ball(self, frame):
        results = self.model(frame, conf=0.3, verbose=False)
        
        players = []
        ball_detected = False
        ball_position = None
        ball_confidence = 0
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    class_id = int(box.cls[0])
                    class_name = self.model.names[class_id]
                    conf = float(box.conf[0])
                    
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
                    
                    if class_name == 'person' and conf > self.player_conf_threshold:
                        players.append((x1, y1, x2, y2, center_x, center_y, conf))
                    
                    elif class_name == 'sports ball' and conf > self.ball_conf_threshold:
                        if conf > ball_confidence:  
                            ball_detected = True
                            ball_position = (center_x, center_y)
                            ball_confidence = conf
        
        
        self.player_detection_history.append(players)
        self.ball_detection_history.append((ball_detected, ball_position, ball_confidence))
        
        
        ball_detections = [det[0] for det in self.ball_detection_history]
        if len(ball_detections) > 0 and sum(ball_detections) / len(ball_detections) > 0.6:
            ball_detected = True
            
            ball_position = self.ball_detection_history[-1][1] if self.ball_detection_history[-1][0] else None
        else:
            ball_detected = False
            ball_position = None
        
        return players, ball_detected, ball_position
    
    def identify_shooter(self, players, ball_position):
        
        if not players or ball_position is None:
            return None
        
        ball_x, ball_y = ball_position
        min_distance = float('inf')
        shooter = None
        
        for player in players:
            x1, y1, x2, y2, center_x, center_y, conf = player
            
            distance = math.sqrt((ball_x - center_x)**2 + (ball_y - center_y)**2)
            
            
            player_size = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            normalized_distance = distance / (player_size / 2)  
            
            if normalized_distance < min_distance:
                min_distance = normalized_distance
                shooter = player
        
        
        if min_distance < 2.0:  
            return shooter
        
        return None
    
    def check_goal(self, ball_position, frame_count):
        """Check if the ball is in the goal area"""
        if self.goal_area is None or ball_position is None:
            return False
        
        goal_x1, goal_y1, goal_x2, goal_y2 = self.goal_area
        ball_x, ball_y = ball_position
        
        
        if goal_x1 <= ball_x <= goal_x2 and goal_y1 <= ball_y <= goal_y2:
            if not self.goal_detected:
                
                if hasattr(self, 'goal_candidate_frames'):
                    self.goal_candidate_frames += 1
                else:
                    self.goal_candidate_frames = 1
                
                if self.goal_candidate_frames >= self.goal_detection_frames:
                    self.goal_detected = True
                    self.goal_frame = frame_count
                    print(f"GOAL DETECTED at frame {frame_count}!")
                    return True
            return True
        
        
        if hasattr(self, 'goal_candidate_frames'):
            self.goal_candidate_frames = 0
        
        return False
    
    def calculate_velocity(self, frame_count):
        if len(self.ball_positions) < 2:
            return None
        
        (x1, y1, f1), (x2, y2, f2) = self.ball_positions[-2], self.ball_positions[-1]
        pixel_distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
        frame_diff = f2 - f1
        time_sec = frame_diff / self.fps
        if time_sec == 0:
            return None
        
        real_distance = pixel_distance / self.pixels_per_meter
        velocity_mps = real_distance / time_sec
        velocity_kmph = velocity_mps * 3.6
        return velocity_mps, velocity_kmph
    
    def analyze_video(self):
        frame_count = 0
        velocity_readings = []
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            frame_count += 1
            frame_copy = frame.copy()
            
            
            if self.goal_area is None and frame_count % 30 == 0:  
                self.detect_goal_area(frame_copy)
            
            players, ball_detected, ball_position = self.detect_players_and_ball(frame_copy)
            
            
            for player in players:
                x1, y1, x2, y2, center_x, center_y, conf = player
                cv2.rectangle(frame_copy, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                cv2.putText(frame_copy, f"Player {conf:.2f}", (int(x1), int(y1)-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            
            if ball_detected and ball_position:
                ball_x, ball_y = ball_position
                cv2.circle(frame_copy, (int(ball_x), int(ball_y)), 10, (255, 255, 0), -1)
                cv2.putText(frame_copy, "Ball", (int(ball_x)+15, int(ball_y)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
               
                self.ball_positions.append((ball_x, ball_y, frame_count))
                
                
                shooter = self.identify_shooter(players, ball_position)
                if shooter:
                    x1, y1, x2, y2, center_x, center_y, conf = shooter
                    
                    cv2.rectangle(frame_copy, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
                    cv2.putText(frame_copy, "SHOOTER", (int(x1), int(y1)-30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    self.shooter_position = (center_x, center_y)
                
                
                velocity = self.calculate_velocity(frame_count)
                if velocity:
                    vel_mps, vel_kmph = velocity
                    velocity_readings.append(vel_kmph)
                    
                    
                    if len(velocity_readings) > 5:
                        recent_avg = np.mean(velocity_readings[-3:])
                        previous_avg = np.mean(velocity_readings[-6:-3])
                        
                        if recent_avg > previous_avg * 1.5 and recent_avg > 30:  
                            if not self.shot_detected:
                                self.shot_detected = True
                                self.shot_frame = frame_count
                                self.shot_velocity = recent_avg
                                print(f"SHOT DETECTED at frame {frame_count}: {recent_avg:.1f} km/h")
                    
                    
                    cv2.putText(frame_copy, f"Velocity: {vel_kmph:.1f} km/h", 
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            
            if ball_detected and ball_position:
                goal_detected = self.check_goal(ball_position, frame_count)
                if goal_detected:
                    cv2.putText(frame_copy, "GOAL!", (self.width//2 - 100, 100), 
                               cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
            
            
            if self.goal_area:
                x1, y1, x2, y2 = self.goal_area
                cv2.rectangle(frame_copy, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                cv2.putText(frame_copy, "Goal Area", (int(x1), int(y1)-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            
            cv2.putText(frame_copy, f"Frame: {frame_count}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            if self.shot_detected:
                cv2.putText(frame_copy, f"Shot: {self.shot_velocity:.1f} km/h", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            if self.goal_detected:
                cv2.putText(frame_copy, "GOAL DETECTED", (10, 120), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            self.output_video.write(frame_copy)
            cv2.imshow("Handball Analysis", frame_copy)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.cap.release()
        self.output_video.release()
        cv2.destroyAllWindows()
        
        
        print("\n=== ANALYSIS SUMMARY ===")
        if self.shot_detected:
            print(f"Shot detected at frame: {self.shot_frame}")
            print(f"Shot velocity: {self.shot_velocity:.1f} km/h")
        
        if self.goal_detected:
            print(f"Goal detected at frame: {self.goal_frame}")
            
            if self.shot_detected:
                time_to_goal = (self.goal_frame - self.shot_frame) / self.fps
                print(f"Time from shot to goal: {time_to_goal:.2f} seconds")
        
        if velocity_readings:
            avg_velocity = np.mean(velocity_readings)
            max_velocity = np.max(velocity_readings)
            print(f"Average ball velocity: {avg_velocity:.1f} km/h")
            print(f"Maximum ball velocity: {max_velocity:.1f} km/h")

if __name__ == "__main__":
    video_path = r"C:\Users\ghars\Downloads\Andersson.mp4"
    analyzer = HandballVelocityAnalyzer(video_path)
    analyzer.analyze_video()
