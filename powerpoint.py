import cv2
import mediapipe as mp
import numpy as np
import pyautogui

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Gesture recognition state
last_gesture = None
gesture_cooldown = 15  # Cooldown period to avoid repeated triggers
gesture_counter = 0

def detect_swipe_direction(hand_landmarks):
    """Detect if a swipe gesture has occurred by checking hand movement direction."""
    index_tip_x = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x
    wrist_x = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x
    return 'right' if index_tip_x > wrist_x else 'left'

while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    # Flip the image for a mirrored display
    image = cv2.flip(image, 1)
    
    # Convert BGR to RGB for MediaPipe processing
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process the image to find hands
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        # Use only the first detected hand for control
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # Calculate distances to recognize gestures
        thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
        index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
        
        # Calculate distances for gesture recognition
        thumb_index_dist = np.linalg.norm(np.array([thumb_tip.x, thumb_tip.y]) - np.array([index_tip.x, index_tip.y]))
        thumb_middle_dist = np.linalg.norm(np.array([thumb_tip.x, thumb_tip.y]) - np.array([middle_tip.x, middle_tip.y]))

        # Gesture recognition logic with cooldown
        if gesture_counter == 0:
            # 1. Swipe Gestures for Slide Control
            swipe_direction = detect_swipe_direction(hand_landmarks)
            if swipe_direction == 'right' and last_gesture != 'next_slide':
                pyautogui.press('right')
                last_gesture = 'next_slide'
                gesture_counter = gesture_cooldown
            elif swipe_direction == 'left' and last_gesture != 'prev_slide':
                pyautogui.press('left')
                last_gesture = 'prev_slide'
                gesture_counter = gesture_cooldown

            # 2. Fist Gesture for Play/Pause Toggle
            elif thumb_index_dist < 0.05 and thumb_middle_dist < 0.05 and last_gesture != 'pause_play':
                pyautogui.press('space')  # Toggles play/pause in most presentation software
                last_gesture = 'pause_play'
                gesture_counter = gesture_cooldown

            # 3. Thumbs Up Gesture for Volume Up
            elif thumb_tip.y < wrist.y and last_gesture != 'volume_up':
                pyautogui.press('volumeup')
                last_gesture = 'volume_up'
                gesture_counter = gesture_cooldown

            # 4. Thumbs Down Gesture for Volume Down
            elif thumb_tip.y > wrist.y and last_gesture != 'volume_down':
                pyautogui.press('volumedown')
                last_gesture = 'volume_down'
                gesture_counter = gesture_cooldown

        # Draw landmarks for debugging and user feedback
        mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        cv2.putText(image, f'Gesture: {last_gesture}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Cooldown timer to avoid repeated gesture triggers
    if gesture_counter > 0:
        gesture_counter -= 1
    else:
        last_gesture = None  # Reset gesture state after cooldown

    # Display the image
    cv2.imshow('Presentation Controller', image)
    
    # Break loop with 'q'
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
