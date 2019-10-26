import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

AIR_DENSITY = 1.2 #kg/m^3
DRAG_COEFF = 0.3
REF_AREA = 0.0427
GRAVITY = 9.81
MASS = 0.04593

w = np.array([0,0])


def f(t, y):
    velocity = y[2:]
    force = (1/MASS)*get_force(velocity)
    return np.concatenate([velocity, force])


def get_force(v):
    gravity_force = MASS*GRAVITY*np.array([0,-1])
    direction = v / np.linalg.norm(v)
    drag_force = 0.5*AIR_DENSITY*DRAG_COEFF*REF_AREA*((w-v)**2).sum()*-direction
    return gravity_force + drag_force


def get_2d_angular_velocity(angle, magnitude):
    return magnitude * np.array([np.cos(angle), np.sin(angle)])


y0 = np.concatenate([[0,0],get_2d_angular_velocity(np.pi/4, 10)])

t1 = 30


class YZeroEvent:
    def __init__(self):
        self.terminal = True
        self.direction = -1

    def __call__(self, *args, **kwargs):
        return args[1][1]


result = solve_ivp(f, (0, t1), y0, events=YZeroEvent(), method='Radau', t_eval=np.linspace(0, t1, t1*20, endpoint=False))
print(result['message'], result['success'], result['t'][-1])
plt.plot(result['y'].T[:,0], result['y'].T[:,1])
plt.scatter(result['y'].T[:,0], result['y'].T[:,1])
plt.xlabel('distance [m]')
plt.ylabel('altitude [m]')
plt.grid()
plt.gca().xaxis.set_major_locator(plt.MultipleLocator(0.5))
plt.gca().yaxis.set_major_locator(plt.MultipleLocator(0.5))
plt.axis('equal')
plt.show()
