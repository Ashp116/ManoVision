import glfw
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import cv2

# Global variables
angle = 0
cube_size = 1.0  # Default size of the cube

def setup_opengl():
    glEnable(GL_DEPTH_TEST)  # Enable depth testing
    glClearColor(0, 0, 0, 0.01)  # Clear color (black with full opacity)
    glDisable(GL_BLEND)  # Disable blending

def draw_cube(size):
    half_size = size / 2.0
    glBegin(GL_QUADS)

    # Front face
    glColor4f(1.0, 0.0, 0.0, 1.0)  # Red with full opacity
    glVertex3f(-half_size, -half_size, half_size)
    glVertex3f(half_size, -half_size, half_size)
    glVertex3f(half_size, half_size, half_size)
    glVertex3f(-half_size, half_size, half_size)

    # Back face
    glColor4f(0.0, 1.0, 0.0, 1.0)  # Green with full opacity
    glVertex3f(-half_size, -half_size, -half_size)
    glVertex3f(-half_size, half_size, -half_size)
    glVertex3f(half_size, half_size, -half_size)
    glVertex3f(half_size, -half_size, -half_size)

    # Top face
    glColor4f(0.0, 0.0, 1.0, 1.0)  # Blue with full opacity
    glVertex3f(-half_size, half_size, -half_size)
    glVertex3f(-half_size, half_size, half_size)
    glVertex3f(half_size, half_size, half_size)
    glVertex3f(half_size, half_size, -half_size)

    # Bottom face
    glColor4f(1.0, 1.0, 0.0, 1.0)  # Yellow with full opacity
    glVertex3f(-half_size, -half_size, -half_size)
    glVertex3f(half_size, -half_size, -half_size)
    glVertex3f(half_size, -half_size, half_size)
    glVertex3f(-half_size, -half_size, half_size)

    # Right face
    glColor4f(1.0, 0.0, 1.0, 1.0)  # Magenta with full opacity
    glVertex3f(half_size, -half_size, -half_size)
    glVertex3f(half_size, half_size, -half_size)
    glVertex3f(half_size, half_size, half_size)
    glVertex3f(half_size, -half_size, half_size)

    # Left face
    glColor4f(0.0, 1.0, 1.0, 1.0)  # Cyan with full opacity
    glVertex3f(-half_size, -half_size, -half_size)
    glVertex3f(-half_size, -half_size, half_size)
    glVertex3f(-half_size, half_size, half_size)
    glVertex3f(-half_size, half_size, -half_size)

    glEnd()

def capture_opengl_frame(window):
    width, height = glfw.get_framebuffer_size(window)
    glPixelStorei(GL_PACK_ALIGNMENT, 1)
    pixels = glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE)
    image = np.frombuffer(pixels, dtype=np.uint8).reshape(height, width, 4)
    image = np.flipud(image)  # Flip the image vertically to match OpenCV coordinate system
    return image

def save_frame_as_image(image):
    filename = 'openGL_overlay.png'
    cv2.imwrite(filename, image)
    print(f'Saved image as {filename}')

def main():
    if not glfw.init():
        return

    # Create a windowed mode window and its OpenGL context
    window = glfw.create_window(800, 600, "OpenGL Cube on Video Capture", None, None)
    if not window:
        glfw.terminate()
        return

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

    global angle
    while not glfw.window_should_close(window):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Clear the buffer and set up the camera view
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        glTranslatef(0.0, 0.0, -5)  # Move the cube back
        glRotatef(angle, 1, 1, 0)  # Rotate the cube

        # Draw the cube with the specified size
        draw_cube(cube_size)

        # Capture OpenGL frame
        opengl_image = capture_opengl_frame(window)

        # Resize video frame to match the OpenGL window size
        frame = cv2.resize(frame, (800, 600))

        # Convert OpenGL image to have an alpha channel
        opengl_image_alpha = cv2.cvtColor(opengl_image, cv2.COLOR_RGBA2BGRA)

        # Combine video frame and OpenGL image (with cube fully opaque)
        combined = np.where(opengl_image_alpha[:, :, 3:] == 255, opengl_image_alpha[:, :, :3], frame)

        # Display the resulting frame
        cv2.imshow("OpenCV Window", combined)

        # Save the frame as an image (only once)
        if cv2.waitKey(1) & 0xFF == ord('s'):  # Press 's' to save
            save_frame_as_image(combined)

        # Exit on ESC key
        if cv2.waitKey(1) & 0xFF == 27:  # ESC key to exit
            break

        # Update rotation angle
        angle += 2
        if angle > 360:
            angle -= 360

    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    glfw.terminate()

if __name__ == "__main__":
    main()
