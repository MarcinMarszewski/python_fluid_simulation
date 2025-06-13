import numpy as np
from math import sqrt
import random
import numba
from kernels import cubic_kernel, near_density_kernel, quad_spiky_kernel, quad_spiky_kernel_derivative

class Simulation:
    def __init__(self, err_rate, grid_size, rest_density, bounds_size, damping_factor, time_step, smoothing_radius, pressure_multiplier, near_pressure_multiplier, gravitional_constant, viscosity_constant):
        self.err_rate = err_rate
        self.grid_size = grid_size
        self.rest_density = rest_density
        self.bounds_size = bounds_size
        self.damping_factor = damping_factor
        self.time_step = time_step
        self.smoothing_radius = smoothing_radius
        self.pressure_multiplier = pressure_multiplier
        self.gravitional_constant = gravitional_constant
        self.viscosity_constant = viscosity_constant
        self.near_pressure_multiplier = near_pressure_multiplier

        N = grid_size * grid_size
        self.N = N

        self.positions = np.zeros((N, 2), dtype=np.float32)
        self.velocities = np.zeros((N, 2), dtype=np.float32)
        self.predicted_positions = np.zeros((N, 2), dtype=np.float32)
        self.densities = np.zeros(N, dtype=np.float32)
        self.near_densities = np.zeros(N, dtype=np.float32)
        self.masses = np.ones(N, dtype=np.float32)

        idx = 0
        for i in range(grid_size):
            for j in range(grid_size):
                self.positions[idx, 0] = 2 + 1.7 * i / rest_density
                self.positions[idx, 1] = 4 + 1.7 * j / rest_density
                idx += 1

    def simulate_step(self):
        self.compute_boundaries()
        self.predict_positions()
        self.calculate_densities()
        self.calculate_pressure_forces()
        self.apply_velocity()

    def compute_boundaries(self):
        for i in range(self.N):
            if self.positions[i, 0] < 0 or self.positions[i, 0] > self.bounds_size:
                self.velocities[i, 0] = -self.velocities[i, 0] * self.damping_factor
                self.positions[i, 0] = np.clip(self.positions[i, 0], 0, self.bounds_size)
            if self.positions[i, 1] < 0 or self.positions[i, 1] > self.bounds_size:
                self.velocities[i, 1] = -self.velocities[i, 1] * self.damping_factor
                self.positions[i, 1] = np.clip(self.positions[i, 1], 0, self.bounds_size)

    def apply_velocity(self):
        self.positions += self.velocities * self.time_step

    def calculate_densities(self):
        self.densities.fill(0.0)
        self.near_densities.fill(0.0)
        _calculate_densities_numba(
            self.N,
            self.predicted_positions,
            self.masses,
            self.smoothing_radius,
            self.densities,
            self.near_densities
        )

    def calculate_pressure_forces(self):
        _calculate_pressure_forces_numba(
            self.N,
            self.positions,
            self.velocities,
            self.densities,
            self.near_densities,
            self.masses,
            self.smoothing_radius,
            self.pressure_multiplier,
            self.rest_density,
            self.near_pressure_multiplier,
            self.viscosity_constant,
            self.gravitional_constant,
            self.time_step
        )

    def predict_positions(self):
        self.predicted_positions = self.positions + self.velocities * self.time_step

    def shared_pressure(self, density_a, density_b):
        return self.pressure_multiplier * ((density_a + density_b) / 2.0 - self.rest_density)

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
        for idx in range(self.N):
            cell = self.get_grid_index(self.positions[idx])
            if cell not in self.grid:
                self.grid[cell] = []
            self.grid[cell].append(idx)

    def get_neighbors(self, idx):
        cell = self.get_grid_index(self.positions[idx])
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                neighbor_cell = (cell[0] + dx, cell[1] + dy)
                if neighbor_cell in self.grid:
                    neighbors.extend(self.grid[neighbor_cell])
        return neighbors

# Numba-accelerated density calculation
@numba.njit(fastmath=True)
def _calculate_densities_numba(N, predicted_positions, masses, smoothing_radius, densities, near_densities):
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            dx = predicted_positions[i, 0] - predicted_positions[j, 0]
            dy = predicted_positions[i, 1] - predicted_positions[j, 1]
            distance = (dx * dx + dy * dy) ** 0.5
            if distance < smoothing_radius:
                influence = quad_spiky_kernel(smoothing_radius, distance)
                near_influence = near_density_kernel(smoothing_radius, distance)
                densities[i] += masses[j] * influence
                near_densities[i] += masses[j] * near_influence

@numba.njit(fastmath=True)
def _calculate_pressure_forces_numba(
    N, positions, velocities, densities, near_densities, masses, smoothing_radius,
    pressure_multiplier, rest_density, near_pressure_multiplier, viscosity_constant,
    gravitional_constant, time_step
):
    for i in range(N):
        pressure_gradient = np.zeros(2, dtype=np.float32)
        viscosity_force = np.zeros(2, dtype=np.float32)
        for j in range(N):
            if i == j:
                continue
            dx = positions[i, 0] - positions[j, 0]
            dy = positions[i, 1] - positions[j, 1]
            distance = (dx * dx + dy * dy) ** 0.5
            if distance == 0:
                direction = np.array([1.0, 0.0], dtype=np.float32)
                distance = 1e-6
            else:
                direction = np.array([dx / distance, dy / distance], dtype=np.float32)
            slope = quad_spiky_kernel_derivative(smoothing_radius, distance)
            if slope == 0:
                continue
            if densities[j] == 0:
                continue
            pressure = -((near_densities[i] + near_densities[j]) * near_pressure_multiplier / 2.0) - (pressure_multiplier * ((densities[i] + densities[j]) / 2.0 - rest_density))
            pressure_gradient += pressure * slope * direction * masses[j] / densities[j]
            influence = cubic_kernel(smoothing_radius, distance)
            viscosity_force += (velocities[j] - velocities[i]) * influence * masses[j] / densities[j]
        if densities[i] > 0:
            velocities[i] += (viscosity_force * viscosity_constant + pressure_gradient) * time_step / densities[i]
            velocities[i, 1] += gravitional_constant * time_step / densities[i]