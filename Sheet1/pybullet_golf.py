import pybullet as p
import pybullet_data
from time import sleep
from plot_trajectory import get_2d_directional_velocity, get_drag_force, WorldPhysics, golfball_physics
from math import pi
import numpy as np

# create a simulation with a plane and a golfball

physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

planeId = p.loadURDF("plane.urdf", globalScaling=100)
p.changeDynamics(planeId, -1, restitution=0.9)

ballStartPos = [0, 0, 0.0213 + 0.002]
ballStartOrientation = p.getQuaternionFromEuler([0, 0, 0])
ballId = p.loadURDF("golfball.urdf", ballStartPos, ballStartOrientation)


# to see the partially inelastic collision, set this parameter to False:
stop_on_collision = False

# with (default) air density
world_physics = WorldPhysics(air_density=1.2)

# y-angle and velocity
angle = pi/4
velocity = 52


# modify simulation based on parameters
p.setGravity(0, 0, -world_physics.gravity)
dir_velocity = get_2d_directional_velocity(angle, velocity)
p.resetBaseVelocity(ballId, [0, *dir_velocity])
p.changeDynamics(ballId, -1, lateralFriction=0.05, linearDamping=0, angularDamping=0,
                 rollingFriction=0.1, mass=golfball_physics.mass, restitution=0.5)


cp = p.getContactPoints(planeId, ballId)

while 1:
    if len(cp) == 0 or not stop_on_collision:
        # get velocity and manually apply drag force
        vel, _ = np.array(p.getBaseVelocity(ballId))
        drag = get_drag_force(vel, world_physics, golfball_physics)
        # for some reason we have to divide by a very odd value here to get similar results to plot_trajectory.py
        vel += drag/11.04
        p.resetBaseVelocity(ballId, vel)

        # check for collision
        cp = p.getContactPoints(planeId, ballId)

        # simulate!
        p.stepSimulation()

        # move camera to ball position
        pos, _ = p.getBasePositionAndOrientation(ballId)
        *_, yaw, pitch, dist, _ = p.getDebugVisualizerCamera()
        p.resetDebugVisualizerCamera(cameraDistance=dist, cameraYaw=yaw, cameraPitch=pitch, cameraTargetPosition=pos)
    sleep(1/240.)