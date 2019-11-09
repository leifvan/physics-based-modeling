import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from functools import partial
from collections import namedtuple
from itertools import product

WorldPhysics = namedtuple("WorldPhysics", "air_density gravity wind_vector",
                          defaults=[1.2, 9.81, np.zeros(3)])
ObjectPhysics = namedtuple("ObjectPhysics", "drag_coeff ref_area mass")
golfball_physics = ObjectPhysics(drag_coeff=0.3, ref_area=np.pi * 0.0427 ** 2 / 4, mass=0.04593)


def f(t, y, world_physics, obj_physics):
    velocity = y[3:]
    force = (1 / obj_physics.mass) * get_force(velocity, world_physics, obj_physics)
    return np.concatenate([velocity, force])


def get_drag_force(v, world_physics, obj_physics):
    relative_velocity = world_physics.wind_vector - v
    direction = relative_velocity / np.linalg.norm(relative_velocity)
    drag_constants = 0.5 * world_physics.air_density * obj_physics.drag_coeff * obj_physics.ref_area
    return drag_constants * (relative_velocity ** 2).sum() * direction


def get_force(v, world_physics, obj_physics):
    gravity_force = obj_physics.mass * world_physics.gravity * np.array([0, 0, -1])
    drag_force = get_drag_force(v, world_physics, obj_physics)
    return gravity_force + drag_force


def get_2d_directional_velocity(angle, magnitude):
    return magnitude * np.array([np.cos(angle), np.sin(angle)])


class ZeroEvent:
    def __init__(self, i):
        self.terminal = True
        self.direction = -1
        self.i = i

    def __call__(self, *args, **kwargs):
        return args[1][self.i]


def calculate_trajectory(pos, velocity, world_physics, obj_physics):
    t1 = 30
    y0 = np.concatenate([pos, velocity])
    f_partial = partial(f, world_physics=world_physics, obj_physics=obj_physics)
    result = solve_ivp(f_partial, (0, t1), y0, events=ZeroEvent(2), method='Radau',
                       t_eval=np.linspace(0, t1, t1 * 30, endpoint=False))
    return result['y'].T


def get_3d_ax():
    ax = plt.figure().add_subplot(111, projection='3d')
    ax.set_xlabel('distance x[m]')
    ax.set_ylabel('distance y[m]')
    ax.set_zlabel('altitude [m]')
    return ax


def plot_trajectory(trajectory, ax=None, label=""):
    if ax is None:
        ax = get_3d_ax()

    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], label=label)
    print(f"{label} landed at ({trajectory[-1, 0]:.2f}, {trajectory[-1, 1]:.2f})")
    return ax


if __name__ == '__main__':

    # +++++++++++++++++++++++++++++++++++++++
    print()
    print("-"*10,"Exercise 2a)","-"*10)
    print()
    # +++++++++++++++++++++++++++++++++++++++


    world1_physics = WorldPhysics(air_density=0)
    traj1 = calculate_trajectory(pos=[0, 0, 0], velocity=[0, *get_2d_directional_velocity(np.pi / 4, 50)],
                                 world_physics=world1_physics, obj_physics=golfball_physics)

    plot_trajectory(traj1, label="50m/s, no drag")
    plt.legend()
    plt.suptitle("a) Simulation without drag")
    plt.show()

    # +++++++++++++++++++++++++++++++++++++++
    print()
    print("-" * 10, "Exercise 2c)", "-" * 10)
    print()
    # +++++++++++++++++++++++++++++++++++++++

    # create a function to be minimized
    def distance_wrt_angle(angle, velocity, world):
        traj = calculate_trajectory(pos=[0, 0, 0], velocity=[0, *get_2d_directional_velocity(angle, velocity)],
                                    world_physics=world, obj_physics=golfball_physics)
        return -traj[-1,1]


    # world with atmospheric drag
    world2_physics = WorldPhysics()
    # world with drag + 4m/s opposing wind
    world3_physics = WorldPhysics(wind_vector=np.array([0,-4,0]))

    # find optimal angle for both worlds and velocities 52 & 56 m/s
    for world, vel in product([world2_physics, world3_physics],[52,56]):
        min_angle = minimize_scalar(partial(distance_wrt_angle, velocity=vel, world=world),
                                    bounds=(np.pi/5, np.pi/3),
                                    method='Bounded')['x']

        print(f"angle = {min_angle/(np.pi*2)*360:.2f}°", "for initial velocity =",vel,"m/s and",
              "4 m/s opposing wind" if id(world) == id(world3_physics) else "no wind")

        traj_45 = calculate_trajectory(pos=[0, 0, 0], velocity=[0, *get_2d_directional_velocity(np.pi/4, vel)],
                                    world_physics=world, obj_physics=golfball_physics)
        ax = plot_trajectory(traj_45, label="45°")
        traj_opt = calculate_trajectory(pos=[0, 0, 0], velocity=[0, *get_2d_directional_velocity(min_angle, vel)],
                                    world_physics=world, obj_physics=golfball_physics)
        plot_trajectory(traj_opt, ax, label="optimal angle")
        plt.legend()
        plt.suptitle(f"c) {vel} m/s initial velocity, "+("4 m/s opposing wind" if id(world) == id(world3_physics) else "no wind"))
        plt.show()
        print()

    # +++++++++++++++++++++++++++++++++++++++
    print()
    print("-" * 10, "Exercise 2d)", "-" * 10)
    print()
    # +++++++++++++++++++++++++++++++++++++++

    world4_physics = WorldPhysics(wind_vector=np.array([7,0,0]))
    traj_xwind = calculate_trajectory(pos=[0,0,0], velocity=[0, *get_2d_directional_velocity(42/360*np.pi,45)],
                                      world_physics=world4_physics, obj_physics=golfball_physics)
    plot_trajectory(traj_xwind,label="crosswind")

    dist_travelled = np.linalg.norm(traj_xwind[-1])
    print(f"Ball travelled {dist_travelled:.2f} m.")
    plt.suptitle("d) 45 m/s initial velocity, 7 m/s crosswind")
    plt.legend()
    plt.show()