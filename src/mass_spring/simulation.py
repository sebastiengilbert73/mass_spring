import cmath
import math

class MassSpring():
    def __init__(self, mass=0.1, gamma=0.3, k=5.0, x0=1.0, v0=0.0, zero_threshold=1e-6):
        self.mass = mass
        self.gamma = gamma
        self.k = k
        self.zero_threshold = zero_threshold
        self.x0 = x0
        self.v0 = v0

        # Compute alpha and omega
        self.alpha = self.gamma/(2 * self.mass)
        self.omega = cmath.sqrt(self.k/self.mass - self.gamma**2/(4 * self.mass**2))
        # Compute phi and A
        self.phi, self.A = self.compute_phi_A(x0, v0)

    def evaluate(self, t):
        i = complex(0, 1)
        return (self.A * cmath.exp(-self.alpha * t) *
                (cmath.exp(i * (self.omega * t + self.phi) ) + cmath.exp(-i * (self.omega * t + self.phi) ) ) ).real

    def evaluate_velocity(self, t):
        x = self.evaluate(t)
        i = complex(0, 1)
        return (-self.alpha * x
                + i * self.omega * self.A * cmath.exp(-self.alpha * t) * ( cmath.exp(i * (self.omega * t + self.phi)) - cmath.exp(-i * (self.omega * t + self.phi)))
                ).real

    def evaluate_acceleration(self, t):
        x = self.evaluate(t)
        v = self.evaluate_velocity(t)
        i = complex(0, 1)
        return (-self.alpha * v
                -i * self.omega * self.alpha * self.A * cmath.exp(-self.alpha * t) * ( cmath.exp(i * (self.omega * t + self.phi) ) - cmath.exp(-i * (self.omega * t + self.phi) ) )
                -self.omega**2 * x
        ).real

    def compute_phi_A(self, x0, v0):
        if abs(x0) <= self.zero_threshold:
            phi = math.pi/2
            A = -v0/(2 * self.omega)
        else:
            tan_phi = (-self.alpha - v0/x0)/self.omega
            phi = cmath.atan(tan_phi)
            A = x0/(2 * cmath.cos(phi))
        return phi, A