import cv2
import mediapipe as mp
import pyautogui
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# Get screen size
screen_width, screen_height = pyautogui.size()

# Open Webcam
cap = cv2.VideoCapture(0)

print("Hand Gesture Mouse Control Started")
print("Move your index finger to move the mouse.")
print("Pinch index and thumb to click.")
print("Press 'q' to exit.")

prev_x, prev_y = 0, 0
curr_x, curr_y = 0, 0
smoothing_factor = 7  # Higher = smoother but more delay

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Flip the frame horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)
    
    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the image and find hands
    results = hands.process(rgb_frame)
    
    frame_height, frame_width, _ = frame.shape
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Key Landmarks
            # 8: Index Finger Tip
            # 4: Thumb Tip
            
            index_finger_tip = hand_landmarks.landmark[8]
            thumb_tip = hand_landmarks.landmark[4]
            
            # Convert normalized coordinates to pixel coordinates
            x = int(index_finger_tip.x * frame_width)
            y = int(index_finger_tip.y * frame_height)
            
            # Map hand coordinates to screen coordinates
            # We map a smaller rectangle of the frame to the full screen to make it easier to reach edges
            # Frame margin to ignore edges
            margin = 100 
            
            # Clamp coordinates to the working area
            x_clamped = np.clip(x, margin, frame_width - margin)
            y_clamped = np.clip(y, margin, frame_height - margin)
            
            # Normalize to 0-1 range within the working area
            x_norm = (x_clamped - margin) / (frame_width - 2 * margin)
            y_norm = (y_clamped - margin) / (frame_height - 2 * margin)
            
            # Map to screen size
            target_x = screen_width * x_norm
            target_y = screen_height * y_norm
            
            # Smoothing
            curr_x = prev_x + (target_x - prev_x) / smoothing_factor
            curr_y = prev_y + (target_y - prev_y) / smoothing_factor
            
            # Move Mouse
            pyautogui.moveTo(curr_x, curr_y)
            
            prev_x, prev_y = curr_x, curr_y
            
            # Click Detection
            # Calculate distance between index tip and thumb tip
            thumb_x = int(thumb_tip.x * frame_width)
            thumb_y = int(thumb_tip.y * frame_height)
            
            distance = np.hypot(x - thumb_x, y - thumb_y)
            
            # Draw distance for debugging
            cv2.line(frame, (x, y), (thumb_x, thumb_y), (255, 0, 0), 2)
            
            # Click threshold (adjust based on camera resolution/distance)
            if distance < 30:
                cv2.circle(frame, (x, y), 10, (0, 255, 0), cv2.FILLED)
                pyautogui.click()
                pyautogui.sleep(0.2) # Prevent double clicks
                
    cv2.imshow('Hand Gesture Mouse', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
