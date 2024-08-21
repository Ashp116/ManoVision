import cv2
import numpy as np
import math
import time
import pyopencl as cl

# Initialize video capture
cap = cv2.VideoCapture(0)  # Use 0 for the default camera

# Define 3D cube vertices and edges
points = np.array([
    [-1, -1, 1],
    [1, -1, 1],
    [1, 1, 1],
    [-1, 1, 1],
    [-1, -1, -1],
    [1, -1, -1],
    [1, 1, -1],
    [-1, 1, -1]
], dtype=np.float32)

cube_edges = [
    (0, 1), (1, 2), (2, 3), (3, 0),  # Front face
    (4, 5), (5, 6), (6, 7), (7, 4),  # Back face
    (0, 4), (1, 5), (2, 6), (3, 7)  # Connecting edges
]

edge_colors = [
    (255, 0, 0),  # Red
    (0, 255, 0),  # Green
    (0, 0, 255),  # Blue
    (255, 255, 0),  # Cyan
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Yellow
    (255, 255, 255),  # White
    (128, 0, 128)  # Purple
]


# Function to project 3D points to 2D
def project_points(points, angle_x, angle_y, angle_z, scale, translate):
    rx = np.array([[1, 0, 0], [0, math.cos(angle_x), -math.sin(angle_x)], [0, math.sin(angle_x), math.cos(angle_x)]],
                  dtype=np.float32)
    ry = np.array([[math.cos(angle_y), 0, math.sin(angle_y)], [0, 1, 0], [-math.sin(angle_y), 0, math.cos(angle_y)]],
                  dtype=np.float32)
    rz = np.array([[math.cos(angle_z), -math.sin(angle_z), 0], [math.sin(angle_z), math.cos(angle_z), 0], [0, 0, 1]],
                  dtype=np.float32)

    points = np.dot(points, rx)
    points = np.dot(points, ry)
    points = np.dot(points, rz)

    points = points * scale
    points[:, :2] += translate

    return points[:, :2]


# Set up OpenCL context and queue
platform = cl.get_platforms()[0]
device = platform.get_devices()[0]
context = cl.Context([device])
queue = cl.CommandQueue(context)

# OpenCL kernel (pass-through for this example)
program_source = """
__kernel void passthrough(__global uchar *image, int width, int height) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    int idx = y * width * 3 + x * 3;
    image[idx] = image[idx];       // Red channel
    image[idx + 1] = image[idx + 1]; // Green channel
    image[idx + 2] = image[idx + 2]; // Blue channel
}
"""
program = cl.Program(context, program_source).build()

# FPS calculation variables
cpu_fps_start_time = time.time()
cpu_frame_count = 0
cpu_fps = 0.0

gpu_fps_start_time = time.time()
gpu_frame_count = 0
gpu_fps = 0.0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Process the CPU-rendered frame
    cpu_frame = frame.copy()
    angle_x = cv2.getTickCount() / cv2.getTickFrequency() * 0.5
    angle_y = cv2.getTickCount() / cv2.getTickFrequency() * 0.3
    angle_z = cv2.getTickCount() / cv2.getTickFrequency() * 0.7

    projected_points_cpu = project_points(points, angle_x, angle_y, angle_z, scale=100,
                                          translate=np.array([cpu_frame.shape[1] // 2, cpu_frame.shape[0] // 2],
                                                             dtype=np.float32))

    for idx, edge in enumerate(cube_edges):
        pt1 = tuple(projected_points_cpu[edge[0]].astype(int))
        pt2 = tuple(projected_points_cpu[edge[1]].astype(int))
        cv2.line(cpu_frame, pt1, pt2, edge_colors[idx % len(edge_colors)], 2)

    # Calculate and display CPU FPS
    cpu_frame_count += 1
    elapsed_time = time.time() - cpu_fps_start_time
    if elapsed_time > 1.0:
        cpu_fps = cpu_frame_count / elapsed_time
        cpu_fps_start_time = time.time()
        cpu_frame_count = 0

    cv2.putText(cpu_frame, f'CPU FPS: {cpu_fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Process the GPU-rendered frame
    frame_np = frame.astype(np.uint8).flatten()
    mf = cl.mem_flags
    image_buf = cl.Buffer(context, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=frame_np)

    program.passthrough(queue, (frame.shape[1], frame.shape[0]), None, image_buf, np.int32(frame.shape[1]),
                        np.int32(frame.shape[0]))

    cl.enqueue_copy(queue, frame_np, image_buf).wait()
    gpu_frame = frame_np.reshape((frame.shape[0], frame.shape[1], 3))

    projected_points_gpu = project_points(points, angle_x, angle_y, angle_z, scale=100,
                                          translate=np.array([gpu_frame.shape[1] // 2, gpu_frame.shape[0] // 2],
                                                             dtype=np.float32))

    for idx, edge in enumerate(cube_edges):
        pt1 = tuple(projected_points_gpu[edge[0]].astype(int))
        pt2 = tuple(projected_points_gpu[edge[1]].astype(int))
        cv2.line(gpu_frame, pt1, pt2, edge_colors[idx % len(edge_colors)], 2)

    # Calculate and display GPU FPS
    gpu_frame_count += 1
    elapsed_time = time.time() - gpu_fps_start_time
    if elapsed_time > 1.0:
        gpu_fps = gpu_frame_count / elapsed_time
        gpu_fps_start_time = time.time()
        gpu_frame_count = 0

    cv2.putText(gpu_frame, f'GPU FPS: {gpu_fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Display the results
    cv2.imshow('CPU Video Capture with Color 3D Cube', cpu_frame)
    cv2.imshow('GPU Video Capture with Color 3D Cube', gpu_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
