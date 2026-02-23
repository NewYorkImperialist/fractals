import pygame

pygame.init()
WIDTH, HEIGHT = 600, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Mandelbrot Iteration Loop')

# Load and scale your image once
try:
    zack_surf = pygame.image.load('obama.png').convert()
    zack_surf = pygame.transform.scale(zack_surf, (WIDTH, HEIGHT))
except:
    print("zack.png not found. Using white instead.")
    zack_surf = pygame.Surface((WIDTH, HEIGHT))
    zack_surf.fill((255, 255, 255))

BLACK = (0, 0, 0)
current_iter = 1  # Start at 1 iteration
running = True

def draw_mandelbrot(iterations):
    """Draws the fractal based on a specific iteration count."""
    pixels = pygame.PixelArray(screen)
    for x in range(WIDTH):
        for y in range(HEIGHT):
            # Map pixel to complex plane
            c_re = (x - WIDTH/2) / (WIDTH/4) - 0.5
            c_im = (y - HEIGHT/2) / (HEIGHT/4)
            
            z = 0 + 0j
            c = complex(c_re, c_im)
            is_in_set = True
            
            for i in range(iterations):
                z = z*z + c
                if abs(z) > 2:
                    is_in_set = False
                    break
            
            if is_in_set:
                pixels[x][y] = zack_surf.get_at((x, y))
            else:
                pixels[x][y] = BLACK
    pixels.close()

# Initial draw
draw_mandelbrot(current_iter)

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Loop iterations 1 to 30
    if current_iter < 30:
        current_iter += 1
        draw_mandelbrot(current_iter)
        pygame.display.set_caption(f'Mandelbrot - Iterations: {current_iter}')

    pygame.display.flip()

pygame.quit()
