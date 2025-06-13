from math import pi
import numba

@numba.njit(fastmath=True)
def near_density_kernel(radius, distance):
    if distance > radius:
        return 0
    volume = pi * (radius ** 5) / 4
    val = max(0, radius - distance)
    return (val ** 4) / volume

@numba.njit(fastmath=True)
def cubic_kernel(radius, distance):
    if distance > radius:
        return 0
    volume = pi * (radius ** 8) / 4
    val = max(0, radius - distance)
    return (val ** 3) / volume

@numba.njit(fastmath=True)
def quad_spiky_kernel(radius, distance):
    if distance > radius:
        return 0
    volume = pi * (radius ** 3) / 6
    return ((radius - distance) ** 2) / volume

@numba.njit(fastmath=True)
def quad_spiky_kernel_derivative(radius, distance):
    if distance > radius:
        return 0
    scale = 12 / ((radius ** 4) * pi)
    return (distance - radius) * scale