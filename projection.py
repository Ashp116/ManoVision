import pygame
import numpy as np
import math

cos, sin = math.cos, math.sin

WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0, 138)
BLACK = (0, 0, 0)

face_colors = [
    (100, 0, 0, 138),  # CYAN
    (0, 100, 0, 138),  # MAGENTA
    (0, 0, 100, 138),  # YELLOW
    (50, 0, 0, 138),   # LIGHT_BLUE
    (0, 50, 100, 138), # ORANGE
    (50, 100, 0, 138)  # PURPLE
]

WIDTH, HEIGHT = 800 , 600
pygame.display.set_caption("3D")
screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.SRCALPHA)

scale = 170
scroll_scale = 20

mouseDown = False

circle_pos = (WIDTH / 2, HEIGHT / 2)
mouse_new_origin = [WIDTH / 2, HEIGHT / 2]

points = np.array([
    [-1, -1,  1],
    [ 1, -1,  1],
    [ 1,  1,  1],
    [-1,  1,  1],
    [-1, -1, -1],
    [ 1, -1, -1],
    [ 1,  1, -1],
    [-1,  1, -1]
])

projection_matrix = np.matrix([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 0],
])

mouse_rotation_matrix = np.matrix([
    [-1.11333333],
    [0.0],
    [-0.3925],
])

previous_mouse_rotation_matrix = mouse_rotation_matrix

projected_points = [
    [n, n] for n in range(len(points))
]

projected_points_labels = [
    [n, n] for n in range(len(points))
]

pygame.font.init() # you have to call this at the start,
                   # if you want to use this module.
my_font = pygame.font.SysFont('Comic Sans MS', 30)

def connect_points(i, j, points):
    pygame.draw.line(
        screen, BLACK, (points[i][0], points[i][1]), (points[j][0], points[j][1]))

def draw_polygon_alpha(surface, color, points):
    lx, ly = zip(*points)
    min_x, min_y, max_x, max_y = min(lx), min(ly), max(lx), max(ly)
    target_rect = pygame.Rect(min_x, min_y, max_x - min_x, max_y - min_y)
    shape_surf = pygame.Surface(target_rect.size, pygame.SRCALPHA)
    pygame.draw.polygon(shape_surf, color, [(x - min_x, y - min_y) for x, y in points])
    surface.blit(shape_surf, target_rect)
def draw_surface(a,b,c,d, points, COLOR=GREEN):
    draw_polygon_alpha(screen, COLOR, [
        (points[a][0], points[a][1]),
        (points[b][0], points[b][1]),
        (points[c][0], points[c][1]),
        (points[d][0], points[d][1]),
    ])

def clamp(n, min, max):
    if n < min:
        return min
    elif n > max:
        return max
    else:
        return n

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                pygame.quit()
                exit()
        if event.type == pygame.MOUSEWHEEL:
            scale = clamp(abs((event.y * scroll_scale) + scale), 30, 170)
        if event.type == pygame.MOUSEBUTTONDOWN:
            mouseDown = True
            mouse_new_origin = list(pygame.mouse.get_pos())
            previous_mouse_rotation_matrix = np.zeros(mouse_rotation_matrix.shape)
        if event.type == pygame.MOUSEBUTTONUP:
            mouseDown = False
            mouse_rotation_matrix = (previous_mouse_rotation_matrix + mouse_rotation_matrix)
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:
                try:
                    pygame.image.save(screen, "test/screenshot.jpeg")
                except:
                    ""
        if event.type == pygame.MOUSEMOTION:
            if mouseDown:
                mousePos = np.matrix(list(pygame.mouse.get_pos()))

                previous_mouse_rotation_matrix[2, 0] = (mouse_new_origin[0] - mousePos[0, 0]) / WIDTH * 2
                previous_mouse_rotation_matrix[0, 0] = (mousePos[0, 1] - mouse_new_origin[1]) / HEIGHT * 2

    screen.fill(WHITE)
    screen.set_alpha(0)

    rot_matrix = (previous_mouse_rotation_matrix + mouse_rotation_matrix) if mouseDown else mouse_rotation_matrix
    rot_matrix[0, 0] = clamp(rot_matrix[0, 0], -math.pi, 0)

    xRot = rot_matrix[0, 0]
    yRot = rot_matrix[1, 0]
    zRot = rot_matrix[2, 0]

    #print(xRot)
    rot_x = np.matrix([
        [1, 0, 0],
        [0, cos(xRot), -sin(xRot)],
        [0, sin(xRot), cos(xRot)]
    ])

    rot_y = np.matrix([
        [cos(yRot), 0, sin(yRot)],
        [0, 1, 0],
        [-sin(yRot), 0, cos(yRot)]
    ])

    rot_z = np.matrix([
        [cos(zRot), -sin(zRot), 0],
        [sin(zRot), cos(zRot), 0],
        [0, 0, 1]
    ])

    for i, point in enumerate(points):
        rotation2D = rot_x.dot(rot_z).dot(rot_y).dot(point.reshape((3, 1)))

        projected2d = np.dot(projection_matrix, rotation2D)

        x = projected2d[0, 0] * scale + circle_pos[0]
        y = projected2d[1, 0] * scale + circle_pos[1]

        projected_points[i] = [x, y]
        projected_points_labels[i] = my_font.render(str(i), False, RED)


    for p in range(4):
        #print(p, (p + 1) % 4, p + 4, ((p + 1) % 4) + 4)
        # connect_points(p, (p + 1) % 4, projected_points)
        # connect_points(p + 4, ((p + 1) % 4) + 4, projected_points)
        # connect_points(p, (p + 4), projected_points)

        a = p
        b = (a + 1) % 4
        c = ((b + 4) % 4) + 4
        d = a + 4

        draw_surface(a, b, c, d, projected_points, COLOR=face_colors[p])

    for p in range(2):

        a = 0 + (4 * p)
        b = 1 + (4 * p)
        c = 2 + (4 * p)
        d = 3 + (4 * p)
        draw_surface(a, b, c, d, projected_points, COLOR=face_colors[p + 4])

    for i, point in enumerate(projected_points):
        pygame.draw.circle(screen, BLACK, tuple(point), 5)
        screen.blit(projected_points_labels[i], tuple(point))

    # Draw Origin
    pygame.draw.circle(screen, RED, circle_pos, 5)

    pygame.display.update()