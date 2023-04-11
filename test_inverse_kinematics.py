import functools
import jax
import os

from datetime import datetime
from jax import numpy as jp
import matplotlib.pyplot as plt

from IPython.display import HTML, clear_output

import brax

import flax
from brax.envs import env
from brax import envs
from brax import base
from brax.io import model
from brax.io import json
from brax.io import html
from brax.io import mjcf

from a1 import A1
from inverse_kinematics.inverse_kinematics_controller import InverseKinematicsController

controller = InverseKinematicsController(Xdist=0.366, Ydist=0.28, height=0.25, coxa=0.038, femur=0.2, tibia=0.2, L=0.8, angle=0, T=1.0, dt=0.01)

a1_env = A1()
jit_env_reset = jax.jit(a1_env.reset, backend="cpu")
jit_env_step = jax.jit(a1_env.step, backend="cpu")

rollout = []
rng = jax.random.PRNGKey(seed=1)
state = jit_env_reset(rng=rng)

current_joint = jp.array([
    0., 1.4, -2.6, 
    0., 1.4, -2.6, 
    0., 1.4, -2.6, 
    0., 1.4, -2.6
])

# stand up
reference_joint = jp.array([
    0.0, 0.6, -1.5, 
    0.0, 0.6, -1.5, 
    0.0, 0.6, -1.5,
    0.0, 0.6, -1.5
])

trajectory = jp.linspace(current_joint, reference_joint, 100)

for action in trajectory:
    rollout.append(state.pipeline_state)
    state = jit_env_step(state, action)

# move from stable stance to first step of performance controller
reference_joint = jp.array(controller.get_action(
    joint_order = ["FR", "FL", "BR", "BL"], offset=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
))

trajectory = jp.linspace(state.pipeline_state.q[7:], reference_joint, 50)

for action in trajectory:
    rollout.append(state.pipeline_state)
    state = jit_env_step(state, action)

# start walking
for _ in range(500):
    rollout.append(state.pipeline_state)
    action = jp.array(controller.get_action(
        joint_order = ["FR", "FL", "BR", "BL"], offset=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ))
    state = jit_env_step(state, action)

with open("render.html", "w") as file:
    file.write(html.render(a1_env.sys, rollout, height=800))
file.close()