import glfw
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np

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
