import cv2
import numpy as np

# Open video capture (0 for the default webcam)
cap = cv2.VideoCapture(0)

while True:
   try:
       # Load the mask image
       mask_image = cv2.imread('screenshot.jpeg')
       # Convert to RGBA if it has only 3 channels
       if mask_image.shape[2] == 3:
           mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2BGRA)

       # Create an alpha channel if it doesn't exist
       # (Here we assume a fully opaque mask image if it had no alpha channel originally)
       if mask_image.shape[2] == 4:
           mask_alpha = mask_image[:, :, 3] / 255.0
       else:
           # Assuming the mask is fully opaque if no alpha channel is present
           mask_alpha = np.ones((mask_image.shape[0], mask_image.shape[1]))

       # Capture frame-by-frame
       ret, frame = cap.read()

       if not ret:
           break

       # Define the position where the mask will be placed
       x_offset = 50
       y_offset = 50

       # Resize the mask image if needed
       height, width = mask_image.shape[:2]
       mask_resized = cv2.resize(mask_image, (int(200 * float(width / height)), 200))  # Resize to fit ROI

       # Get dimensions of the mask
       mask_height, mask_width = mask_resized.shape[:2]

       # Define the region of interest (ROI) on the frame
       roi = frame[y_offset:y_offset + mask_height, x_offset:x_offset + mask_width]

       # Extract the alpha mask of the resized mask image
       if mask_resized.shape[2] == 4:
           mask_alpha = mask_resized[:, :, 3] / 255.0
       else:
           mask_alpha = np.ones((mask_resized.shape[0], mask_resized.shape[1]))

       mask_inv_alpha = 1.0 - mask_alpha

       # Blend the mask with the ROI
       for c in range(0, 3):
           roi[:, :, c] = (mask_alpha * mask_resized[:, :, c] + mask_inv_alpha * roi[:, :, c])

       # Place the modified ROI back into the frame
       frame[y_offset:y_offset + mask_height, x_offset:x_offset + mask_width] = roi

       # Display the resulting frame
       cv2.imshow('Video with Mask Overlay', frame)

       # Break the loop if 'q' is pressed
       if cv2.waitKey(1) & 0xFF == ord('q'):
           break
   except:
       continue

# Release the capture and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()
