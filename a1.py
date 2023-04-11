import jax
from jax import numpy as jp
import brax
import flax
from brax.envs import env
from brax import envs
from brax import base
from brax.io import model
from brax.io import json
from brax.io import html
from brax.io import mjcf

class A1(env.PipelineEnv):
    def __init__(
        self, 
        path="a1/xml/a1.xml", 
        backend='generalized',
        reset_noise_scale=0.1,
        **kwargs
    ):
        sys = mjcf.load(path)
        n_frames = 1
        kwargs['n_frames'] = kwargs.get('n_frames', n_frames)
        
        super().__init__(sys=sys, backend=backend, **kwargs)
        
        self.qd_size = self.sys.qd_size()
        self.q_size = self.sys.q_size()
        self._reset_noise_scale = reset_noise_scale
        
    def reset(self, rng: jp.ndarray) -> env.State:
        # sit down
        q = jp.array([
            0., 0., 0.14, 
            1., 0., 0., 0., 
            0., 1.4, -2.6, 
            0., 1.4, -2.6, 
            0., 1.4, -2.6, 
            0., 1.4, -2.6
        ]) 
        
        qd = jp.zeros(self.qd_size) # velocity initialized to 0
        
        pipeline_state = self.pipeline_init(q, qd)
        obs = self._get_obs(pipeline_state)
        
        reward, done, zero = jp.zeros(3)
        info = {}
        
        return env.State(pipeline_state, obs, reward, done, info)

    def _get_obs(self, pipeline_state: base.State) -> jp.ndarray:
        """Observe ant body position and velocities."""	
        qpos = pipeline_state.q
        qvel = pipeline_state.qd

        return jp.concatenate([qpos] + [qvel])
    
    def step(self, state: env.State, action: jp.ndarray) -> env.State:
        """Run one timestep of the environment's dynamics."""
        # low level control turning reference position (input action) to torque:
        pipeline_state0 = state.pipeline_state
        
        joint_pos = pipeline_state0.q[7:]
        joint_vel = pipeline_state0.qd[6:]
        e_pos = action - joint_pos
        e_v = jp.zeros(12) - joint_vel

        feedback_abduction = e_pos*120.0 + e_v*0.5
        abduction_action = jp.array([feedback_abduction[j] for j in [0, 3, 6, 9]])

        feedback_hip = e_pos*80.0 + e_v*1.0
        hip_action = jp.array([feedback_hip[j] for j in [1, 4, 7, 10]])

        feedback_knee = e_pos*120.0 + e_v*2.0
        knee_action = jp.array([feedback_knee[j] for j in [2, 5, 8, 11]])

        action = jp.array([abduction_action, hip_action, knee_action]).T.reshape(-1)
        
        pipeline_state = self.pipeline_step(pipeline_state0, action)
        obs = self._get_obs(pipeline_state)
        
        reward, done, zero = jp.zeros(3)
        info = {}
        
        return state.replace(
            pipeline_state=pipeline_state, obs=obs, reward=reward, done=done
        )