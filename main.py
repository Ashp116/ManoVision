import math
import cv2
import numpy as np
import glfw
from OpenGL.GL import *
from OpenGL.GLU import *
import time

from Handtracking import HandDetector
from cube_renderer import setup_opengl, draw_cube, capture_opengl_frame

# Global variables
mouseDown = False
angle = np.zeros((1, 2))
delta_angle = np.zeros((1, 2))
new_mouse_origin = None
cube_size = 1.0  # Default size of the cube

debug_mode = False
draw_mouse = True

def get_mouse_pos(lmList):
    # Calculate the position based on the hand landmarks
    return (lmList["Right"][4][0:2] + (lmList["Right"][8][0:2] - lmList["Right"][4][0:2]))

def clamp(n, min, max):
    if n < min:
        return min
    elif n > max:
        return max
    else:
        return n


def main():
    if not glfw.init():
        return

    # Create a windowed mode window and its OpenGL context
    window = glfw.create_window(800, 600, "OpenGL Cube on Video Capture", None, None)
    if not window:
        glfw.terminate()
        return
    glfw.hide_window(window)

    # Make the window's context current
    glfw.make_context_current(window)
    setup_opengl()

    # Set up OpenCV video capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video capture.")
        glfw.terminate()
        return

    # Set up OpenGL viewport and projection matrix
    glViewport(0, 0, 800, 600)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, 800 / 600, 0.1, 50.0)
    glMatrixMode(GL_MODELVIEW)

    detector = HandDetector()
    global angle, delta_angle, new_mouse_origin, mouseDown

    pTime = 0  # Previous time
    while not glfw.window_should_close(window):
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame.")
            break

        # The cube is rendered at 800 x 600 while the original video different size.
        # This matrix gives the scalar multiplier for the width and height of the original video mouse position to the
        # new mouse position in 800 x 600 space.
        # Hadamard Quotient:
        scaled_frame_size_mul = np.divide(np.array([800, 600]), np.flip(np.array(list(frame.shape)))[1:3]) # Hadamard Quotient

        # scaled_frame_size_mul = np.diag(
        #     np.linalg.inv(
        #         np.array([[frame.shape[1], 0],
        #                   [0, frame.shape[0]]]))
        #     .dot(np.array([[800, 0],
        #                    [0, 600]]))
        # )

        frame = detector.findHands(cv2.flip(frame, 1), draw=debug_mode)
        lmList = detector.find_position(frame=frame)

        if lmList and lmList.get("Right"):
            if np.linalg.norm(lmList["Right"][8][2:4] - lmList["Right"][4][2:4]) < .036 and not mouseDown:
                mouseDown = True
                delta_angle = np.zeros((1, 2))
                new_mouse_origin = scaled_frame_size_mul * get_mouse_pos(lmList)
            elif np.linalg.norm(lmList["Right"][8][2:4] - lmList["Right"][4][2:4]) > .036 and mouseDown:
                mouseDown = False
                angle += delta_angle
                delta_angle = np.zeros((1, 2))
        else:
            mouseDown = False
            angle += delta_angle
            delta_angle = np.zeros((1, 2))

        # Clear the buffer and set up the camera view
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        glTranslatef(0.0, 0.0, -5)  # Move the cube back

        glRotatef(clamp(math.degrees(angle[0, 1] + delta_angle[0, 1]), -90, 90), 1, 0, 0)  # Rotate the cube on x-axis
        glRotatef(math.degrees(angle[0, 0] + delta_angle[0, 0]), 0, 1, 0)  # Rotate the cube y-axis

        # Draw the cube
        draw_cube(cube_size)

        # Capture OpenGL frame
        opengl_image = capture_opengl_frame(window)

        # Resize video frame to match the OpenGL window size
        frame = cv2.resize(frame, (800, 600))

        # Convert OpenGL image to have an alpha channel
        opengl_image_alpha = cv2.cvtColor(opengl_image, cv2.COLOR_RGBA2BGRA)

        # Combine video frame and OpenGL image (with cube fully opaque)
        combined = np.where(opengl_image_alpha[:, :, 3:] == 255, opengl_image_alpha[:, :, :3], frame)

        if mouseDown:
            scaled_position = scaled_frame_size_mul * (get_mouse_pos(lmList))
            pre_delta_angle = ((new_mouse_origin - scaled_position) / 100).reshape(1, 2)

            if (math.pi/2 < angle[0, 1] + pre_delta_angle[0, 1] or angle[0, 1] + pre_delta_angle[0, 1] < -math.pi/2):
                delta_angle = np.array([pre_delta_angle[0, 0], delta_angle[0, 1]]).reshape(1, 2)
            else:
                delta_angle = pre_delta_angle

            if draw_mouse:
                cv2.circle(combined, tuple(int(value.item()) for value in scaled_position), 5, (0, 0, 0), cv2.FILLED)

        # Calculate FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        # Display FPS on the frame
        cv2.putText(combined, f'FPS: {int(fps)}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        # Display the status of debug_mode and draw_mouse
        cv2.putText(combined, f'Debug: {debug_mode}', (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
                    cv2.LINE_AA)
        cv2.putText(combined, f'Draw Mouse: {draw_mouse}', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
                    cv2.LINE_AA)

        # Display the resulting frame
        cv2.imshow("OpenCV Window", combined)

        # Exit on ESC key
        if cv2.waitKey(1) & 0xFF == 27:  # ESC key to exit
            break

    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    glfw.terminate()

if __name__ == "__main__":
    main()
