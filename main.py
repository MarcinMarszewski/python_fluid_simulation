import pygame
from fluid_optimised import Simulation

PARTICLE_GRID_SIZE = 20
REST_DENSITY = 7.0
BOUNDS_SIZE = 10
DAMPING_FACTOR = 0.8
TIME_STEP = 0.001
SMOOTHING_RADIUS = 0.5
PRESSURE_MULTIPLIER = 8.8
NEAR_PRESSURE_MULTIPLIER = 4.0
GRAVITIONAL_CONSTANT = 4
VISCOSITY_CONSTANT = 1.8
ERR_RATE = 10e-6

WIDTH, HEIGHT = 600, 600
PARTICLE_RADIUS = 4
BG_COLOR = (30, 30, 40)

def map_to_screen(pos, bounds_size, width, height):
    x = int((pos[0] / bounds_size) * width)
    y = int((pos[1] / bounds_size) * height)
    return x, y

def density_to_color(density, min_density, max_density):
    t = (density - min_density) / (max_density - min_density + 1e-5)
    t = max(0.0, min(1.0, t))
    r = int(100 + 105 * t)
    g = int(200 - 100 * t)
    b = int(255 - 105 * t)
    return (r, g, b)

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Fluid Simulation")
    clock = pygame.time.Clock()

    sim = Simulation(ERR_RATE, PARTICLE_GRID_SIZE, REST_DENSITY, BOUNDS_SIZE, DAMPING_FACTOR, TIME_STEP, SMOOTHING_RADIUS, PRESSURE_MULTIPLIER, NEAR_PRESSURE_MULTIPLIER, GRAVITIONAL_CONSTANT, VISCOSITY_CONSTANT)

    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        sim.simulate_step()

        densities = [p.near_density for p in sim.particles]
        min_density = min(densities)
        max_density = max(densities)

        screen.fill(BG_COLOR)
        for particle in sim.particles:
            x, y = map_to_screen(particle.position, BOUNDS_SIZE, WIDTH, HEIGHT)
            color = density_to_color(particle.near_density, min_density, max_density)
            pygame.draw.circle(screen, color, (x, y), PARTICLE_RADIUS)
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()
