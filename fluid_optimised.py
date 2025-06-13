from particle import Particle2D
from math import sqrt
import random
from kernels import cubic_kernel
from kernels import near_density_kernel
from kernels import quad_spiky_kernel
from kernels import quad_spiky_kernel_derivative

class Simulation:
    def __init__(self, err_rate, grid_size, rest_density, boudns_size, damping_factor, time_step, smoothing_radius, pressure_multiplier, near_pressure_multiplier, gravitional_constant, viscosity_constant):
        self.err_rate = err_rate
        self.grid_size = grid_size
        self.rest_density = rest_density
        self.boudns_size = boudns_size
        self.damping_factor = damping_factor
        self.time_step = time_step
        self.smoothing_radius = smoothing_radius
        self.pressure_multiplier = pressure_multiplier
        self.gravitional_constant = gravitional_constant
        self.viscosity_constant = viscosity_constant
        self.near_pressure_multiplier = near_pressure_multiplier

        self.particles = []

        for i in range(grid_size):
            for j in range(grid_size):
                self.particles.append(
                    Particle2D(
                        position=(1.73* i / (self.rest_density), 4+ 1.73* j / (self.rest_density)),
                        velocity=(0.0, 0.0),
                        mass=1.0,         
                        density=0.0,
                        near_density=0.0
                        
                    )
                )

    def simulate_step(self):
        print("Simulating step...")
        self.compute_boundaries()
        self.predict_positions()
        self.calculate_densities()
        self.calculate_pressure_forces()
        self.apply_velocity()

    def compute_boundaries(self):
        for particle in self.particles:
            if particle.position[0] < 0 or particle.position[0] > self.boudns_size:
                particle.velocity = (-particle.velocity[0] * self.damping_factor, particle.velocity[1] * self.damping_factor)
                particle.position = (
                    max(0, min(particle.position[0], self.boudns_size)),
                    particle.position[1]
                )
            if particle.position[1] < 0 or particle.position[1] > self.boudns_size:
                particle.velocity = (particle.velocity[0] * self.damping_factor, -particle.velocity[1] * self.damping_factor)
                particle.position = (
                    particle.position[0],
                    max(0, min(particle.position[1], self.boudns_size))
                )
    
    def apply_velocity(self):
        for particle in self.particles:
            particle.position = (
                particle.position[0] + particle.velocity[0] * self.time_step,
                particle.position[1] + particle.velocity[1] * self.time_step
            )

    def calculate_densities(self):
        self.build_spatial_grid()
        for i, particle in enumerate(self.particles):
            particle.density = 0.0
            particle.near_density = 0.0
            neighbor_indices = self.get_neighbors(particle)
            for j in neighbor_indices:
                if i == j:
                    continue
                other = self.particles[j]
                distance = sqrt(
                    (particle.predicted_position[0] - other.predicted_position[0]) ** 2 +
                    (particle.predicted_position[1] - other.predicted_position[1]) ** 2
                )
                if distance < self.smoothing_radius:
                    influence = quad_spiky_kernel(self.smoothing_radius, distance)
                    particle.density += other.mass * influence
                    near_influence = near_density_kernel(self.smoothing_radius, distance)
                    particle.near_density += other.mass * near_influence
                    

    def calculate_pressure_forces(self):
        for i, particle in enumerate(self.particles):
            pressure_gradient = [0.0, 0.0]
            viscosity_force = [0.0, 0.0]
            neighbor_indices = self.get_neighbors(particle)
            for j in neighbor_indices:
                if i == j:
                    continue
                other = self.particles[j]
                distance = sqrt(
                    (particle.position[0] - other.position[0]) ** 2 +
                    (particle.position[1] - other.position[1]) ** 2
                )
                direction = (0.0, 0.0)
                if distance == 0:
                    direction = (random.uniform(-1, 1), random.uniform(-1, 1))
                    distance = self.err_rate
                    direction = (
                        direction[0] / sqrt(direction[0] ** 2 + direction[1] ** 2),
                        direction[1] / sqrt(direction[0] ** 2 + direction[1] ** 2)
                    )
                else:
                    direction = (
                        (particle.position[0] - other.position[0]) / distance,
                        (particle.position[1] - other.position[1]) / distance
                    )
                slope = quad_spiky_kernel_derivative(self.smoothing_radius, distance)
                if slope == 0:
                    continue
                if other.density == 0:
                    continue
                pressure_gradient[0] += (-self.shared_near_pressure(particle.near_density, other.near_density) - self.shared_pressure(particle.density, other.density)) * slope * direction[0] * other.mass / other.density
                pressure_gradient[1] += (-self.shared_near_pressure(particle.near_density, other.near_density) - self.shared_pressure(particle.density, other.density)) * slope * direction[1] * other.mass / other.density
                influence = cubic_kernel(self.smoothing_radius, distance)
                viscosity_force[0] += (other.velocity[0] - particle.velocity[0]) * influence * other.mass / other.density
                viscosity_force[1] += (other.velocity[1] - particle.velocity[1]) * influence * other.mass / other.density
            if particle.density > 0:
                particle.velocity = (
                    particle.velocity[0] + (viscosity_force[0] * self.viscosity_constant + pressure_gradient[0]) * self.time_step / particle.density,
                    particle.velocity[1] + (viscosity_force[1] * self.viscosity_constant + pressure_gradient[1] + self.gravitional_constant) * self.time_step / particle.density
                )

    def predict_positions(self):
        for particle in self.particles:
            particle.predicted_position = (
                particle.position[0] + particle.velocity[0] * self.time_step,
                particle.position[1] + particle.velocity[1] * self.time_step
            )
    
    def shared_pressure(self, density_a, density_b):
        return self.pressure_multiplier * ((density_a + density_b)/2.0 - self.rest_density)
    
    def shared_near_pressure(self, near_density_a, near_density_b):
        return (near_density_a + near_density_b) * self.near_pressure_multiplier / 2.0


    def get_grid_index(self, position):
        cell_size = self.smoothing_radius
        return (
            int(position[0] // cell_size),
            int(position[1] // cell_size)
        )

    def build_spatial_grid(self):
        self.grid = {}
        for idx, particle in enumerate(self.particles):
            cell = self.get_grid_index(particle.position)
            if cell not in self.grid:
                self.grid[cell] = []
            self.grid[cell].append(idx)

    def get_neighbors(self, particle):
        cell = self.get_grid_index(particle.position)
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                neighbor_cell = (cell[0] + dx, cell[1] + dy)
                if neighbor_cell in self.grid:
                    neighbors.extend(self.grid[neighbor_cell])
        return neighbors
