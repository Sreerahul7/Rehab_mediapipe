import cv2
import mediapipe as mp
import math

# -----------------------------------------------------------
# Pose Detector Utility Class
# -----------------------------------------------------------
class PoseDetector:
    def __init__(self):
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.results = None

    def findPose(self, img):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)

    def getLandmarks(self, img):
        lmDict = {}
        if self.results and self.results.pose_landmarks:
            h, w, c = img.shape
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                lmDict[id] = (int(lm.x * w), int(lm.y * h))
        return lmDict

    def findAngleWithArc(self, img, p1, p2, p3, draw_color=(0, 0, 255)):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3

        # Calculate angle
        a = math.dist((x2, y2), (x3, y3))
        b = math.dist((x1, y1), (x3, y3))
        c = math.dist((x1, y1), (x2, y2))

        try:
            angle = math.degrees(math.acos((a*a + c*c - b*b) / (2*a*c)))
        except ValueError:
            angle = 0
        
        angle = int(angle)

        # Draw lines and joints
        cv2.line(img, (x1, y1), (x2, y2), draw_color, 4)
        cv2.line(img, (x2, y2), (x3, y3), draw_color, 4)
        cv2.circle(img, (x1, y1), 6, (255, 255, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 6, (255, 255, 255), cv2.FILLED)
        cv2.circle(img, (x3, y3), 6, (255, 255, 255), cv2.FILLED)
        
        # Draw Angle Text
        cv2.putText(img, str(angle), (x2 - 50, y2 + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, draw_color, 2)

        return angle, img

detector = PoseDetector()

# -----------------------------------------------------------
# Generic Exercise Base Class
# -----------------------------------------------------------
class ExerciseBase:
    def __init__(self, name):
        self.name = name
        self.count = 0
        self.stage = "down" 

    # Helper to calculate angle without drawing (for logic checks)
    def calculate_angle(self, p1, p2, p3):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        a = math.dist((x2, y2), (x3, y3))
        b = math.dist((x1, y1), (x3, y3))
        c = math.dist((x1, y1), (x2, y2))
        try:
            return int(math.degrees(math.acos((a*a + c*c - b*b) / (2*a*c))))
        except:
            return 180

    def detect(self, img, lmDict):
        raise NotImplementedError

# -----------------------------------------------------------
# PRECISE KNEE RAISE EXERCISE
# -----------------------------------------------------------
class KneeRaiseExercise(ExerciseBase):
    def __init__(self):
        super().__init__("Knee Raise")

    def detect(self, img, lmDict):
        draw_color = (255, 255, 255) # Default White
        feedback = "Ready"
        
        # Check if we have necessary landmarks
        if not (all(k in lmDict for k in [11, 23, 25, 27]) and all(k in lmDict for k in [12, 24, 26, 28])):
            return img, "No Pose", self.count

        # 1. Calculate Hip Angles (Pure Math, No Draw)
        l_hip_angle = self.calculate_angle(lmDict[11], lmDict[23], lmDict[25])
        r_hip_angle = self.calculate_angle(lmDict[12], lmDict[24], lmDict[26])

        # 2. Determine Active Leg
        if l_hip_angle > 160 and r_hip_angle > 160:
            feedback = "Ready"
            draw_color = (255, 255, 255)
            if self.stage == "up":
                self.stage = "down"
                self.count += 1
        else:
            # Active Leg Logic
            if l_hip_angle < r_hip_angle:
                # Left Leg Active
                active_hip = l_hip_angle
                active_knee = self.calculate_angle(lmDict[23], lmDict[25], lmDict[27])
                pts_hip = [11, 23, 25]
                pts_knee = [23, 25, 27]
            else:
                # Right Leg Active
                active_hip = r_hip_angle
                active_knee = self.calculate_angle(lmDict[24], lmDict[26], lmDict[28])
                pts_hip = [12, 24, 26]
                pts_knee = [24, 26, 28]

            # 3. Precision Checks
            if active_knee > 120:
                feedback = "Bend Knee"
                draw_color = (0, 0, 255) # Red
            elif active_hip > 100: 
                feedback = "Higher"
                draw_color = (0, 165, 255) # Orange
            else:
                feedback = "Good"
                draw_color = (0, 255, 0) # Green
                self.stage = "up"

            # 4. Draw Active Leg
            _, img = detector.findAngleWithArc(img, lmDict[pts_hip[0]], lmDict[pts_hip[1]], lmDict[pts_hip[2]], draw_color)
            _, img = detector.findAngleWithArc(img, lmDict[pts_knee[0]], lmDict[pts_knee[1]], lmDict[pts_knee[2]], draw_color)

        return img, feedback, self.count

# -----------------------------------------------------------
# SHOULDER ABDUCTION Exercise
# -----------------------------------------------------------
class ShoulderAbduction(ExerciseBase):
    def __init__(self):
        super().__init__("Shoulder Abduction")

    def detect(self, img, lmDict):
        l_angle = 0
        r_angle = 0
        draw_color_L = (0, 0, 255)
        draw_color_R = (0, 0, 255)
        feedback = "Fix"

        if all(k in lmDict for k in [23, 11, 13]):
            l_angle = self.calculate_angle(lmDict[23], lmDict[11], lmDict[13])
            draw_color_L = (0, 255, 0) if l_angle >= 80 else (0, 0, 255)

        if all(k in lmDict for k in [24, 12, 14]):
            r_angle = self.calculate_angle(lmDict[24], lmDict[12], lmDict[14])
            draw_color_R = (0, 255, 0) if r_angle >= 80 else (0, 0, 255)

        active_angle = max(l_angle, r_angle)

        if active_angle >= 80:
            feedback = "Good"
            self.stage = "up"
        else:
            feedback = "Fix"
        
        if active_angle <= 30 and self.stage == "up":
            self.stage = "down"
            self.count += 1

        if all(k in lmDict for k in [23, 11, 13]):
            _, img = detector.findAngleWithArc(img, lmDict[23], lmDict[11], lmDict[13], draw_color_L)
        if all(k in lmDict for k in [24, 12, 14]):
            _, img = detector.findAngleWithArc(img, lmDict[24], lmDict[12], lmDict[14], draw_color_R)

        return img, feedback, self.count

# -----------------------------------------------------------
# LUNGE Exercise
# -----------------------------------------------------------
class LungeExercise(ExerciseBase):
    def __init__(self):
        super().__init__("Lunge")

    def detect(self, img, lmDict):
        check_angle = 180
        draw_color = (0, 0, 255) 
        feedback = "Fix"

        if all(k in lmDict for k in [23, 25, 27]):
            check_angle = self.calculate_angle(lmDict[23], lmDict[25], lmDict[27])
        elif all(k in lmDict for k in [24, 26, 28]):
            check_angle = self.calculate_angle(lmDict[24], lmDict[26], lmDict[28])
            
        if check_angle <= 100:
            draw_color = (0, 255, 0)
            feedback = "Good"
            self.stage = "down"
        else:
            draw_color = (0, 0, 255)
            feedback = "Fix"
        
        if check_angle >= 150:
            feedback = "Ready"
            if self.stage == "down":
                self.stage = "up"
                self.count += 1

        if all(k in lmDict for k in [23, 25, 27]):
            _, img = detector.findAngleWithArc(img, lmDict[23], lmDict[25], lmDict[27], draw_color)
        if all(k in lmDict for k in [24, 26, 28]):
            _, img = detector.findAngleWithArc(img, lmDict[24], lmDict[26], lmDict[28], draw_color)

        return img, feedback, self.count

# -----------------------------------------------------------
# Setup & Main Loop
# -----------------------------------------------------------
exercises = {
    "lunge": LungeExercise(),
    "knee_raise": KneeRaiseExercise(),
    "shoulder": ShoulderAbduction()
}

current_exercise = exercises["knee_raise"]

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened(): cap = cv2.VideoCapture(0)

# Window Setup: WINDOW_NORMAL allows resizing/maximizing
window_name = "Exercise Detection"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

# Optional: Set a default large size, or let it handle itself
# cv2.resizeWindow(window_name, 1280, 720)

while True:
    success, img = cap.read()
    if not success: continue

    detector.findPose(img)
    lmDict = detector.getLandmarks(img)
    
    img, feedback, reps = current_exercise.detect(img, lmDict)

    # -------------------------------------------------------
    # UI OVERLAY
    # -------------------------------------------------------
    overlay = img.copy()
    x, y, w, h = 10, 10, 280, 120
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 0), -1)
    alpha = 0.5
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    
    cv2.putText(img, f"{current_exercise.name}", (x + 10, y + 30),
                cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1)
    
    cv2.putText(img, f"Reps: {reps}", (x + 10, y + 70),
                cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 255), 2)
    
    # Text Color Logic
    text_color = (255, 255, 255)
    if feedback == "Good": text_color = (0, 255, 0)
    elif feedback == "Fix" or feedback == "Bend Knee": text_color = (0, 0, 255)
    elif feedback == "Higher": text_color = (0, 165, 255)
    
    cv2.putText(img, f"Feedback: {feedback}", (x + 10, y + 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
    
    # Simple help text for fullscreen
    cv2.putText(img, "Press 'F' for Fullscreen", (10, img.shape[0] - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    cv2.imshow(window_name, img)

    key = cv2.waitKey(1)
    
    # Key Controls
    if key == ord('1'): current_exercise = exercises["lunge"]
    if key == ord('2'): current_exercise = exercises["knee_raise"]
    if key == ord('3'): current_exercise = exercises["shoulder"]
    
    # Full Screen Toggle Logic
    if key == ord('f') or key == ord('F'):
        prop = cv2.getWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN)
        if prop == cv2.WINDOW_FULLSCREEN:
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
        else:
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    if key == 27: break # ESC to exit

cap.release()
cv2.destroyAllWindows()