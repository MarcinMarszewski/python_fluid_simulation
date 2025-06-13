class Particle2D:
    def __init__(self, position, velocity, mass, density, near_density):
        self.position = position
        self.velocity = velocity
        self.mass = mass
        self.density = density
        self.near_density = near_density
        self.predicted_position = position
