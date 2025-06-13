import pygame
from fluid_array import Simulation

PARTICLE_GRID_SIZE = 30
REST_DENSITY = 7.0
BOUNDS_SIZE = 20
DAMPING_FACTOR = 0.8
TIME_STEP = 0.0005
SMOOTHING_RADIUS = 0.8
PRESSURE_MULTIPLIER = 20.8
NEAR_PRESSURE_MULTIPLIER = 20.0
GRAVITIONAL_CONSTANT = 4
VISCOSITY_CONSTANT = 2.6
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

    sim = Simulation(
        ERR_RATE, PARTICLE_GRID_SIZE, REST_DENSITY, BOUNDS_SIZE, DAMPING_FACTOR,
        TIME_STEP, SMOOTHING_RADIUS, PRESSURE_MULTIPLIER, NEAR_PRESSURE_MULTIPLIER,
        GRAVITIONAL_CONSTANT, VISCOSITY_CONSTANT
    )

    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        sim.simulate_step()

        densities = sim.near_densities
        min_density = densities.min()
        max_density = densities.max()

        screen.fill(BG_COLOR)
        for i in range(sim.N):
            if sim.positions[i, 0] < 0 or sim.positions[i, 0] > BOUNDS_SIZE or \
               sim.positions[i, 1] < 0 or sim.positions[i, 1] > BOUNDS_SIZE:
                continue
            x, y = map_to_screen(sim.positions[i], BOUNDS_SIZE, WIDTH, HEIGHT)
            color = density_to_color(sim.near_densities[i], min_density, max_density)
            pygame.draw.circle(screen, color, (x, y), PARTICLE_RADIUS)
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()