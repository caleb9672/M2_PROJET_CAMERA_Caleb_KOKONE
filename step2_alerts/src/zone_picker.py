import cv2
import json
import os

# Path to the frame we just saved
frame_path = r'c:\Users\DELL\Documents\APEKE\step1_detection\src\hall_porte_droite_frame.jpg'
points = []

def click_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Point selected: ({x}, {y})")
        points.append([x, y])        
        # Draw the point and lines
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
        if len(points) > 1:
            cv2.line(img, tuple(points[-2]), tuple(points[-1]), (0, 255, 0), 2)
        cv2.imshow('Define Zone - Click to add points, Press "q" to finish', img)

if not os.path.exists(frame_path):
    print(f"Error: Frame not found at {frame_path}")
else:
    img = cv2.imread(frame_path)
    cv2.imshow('Define Zone - Click to add points, Press "q" to finish', img)
    cv2.setMouseCallback('Define Zone - Click to add points, Press "q" to finish', click_event)
    
    print("Instructions:")
    print("1. Click on the image to define the vertices of your alert zone (polygon).")
    print("2. Press 'q' when you are finished.")
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    if points:
        # Close the polygon
        if len(points) > 2:
            print("Closing polygon...")
        
        print("\nYour coordinates:")
        print(json.dumps(points))
        
        # Save to a temporary file for easy copy-paste
        output_path = r'c:\Users\DELL\Documents\APEKE\step2_alerts\zone_coords.json'
        with open(output_path, 'w') as f:
            json.dump(points, f)
        print(f"\nCoordinates saved to {output_path}")