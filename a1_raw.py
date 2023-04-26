"""
A1 env that includes the reset and step function that takes in all information of pipeline instead of using pipeline_state
"""

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

class A1Raw(env.PipelineEnv):
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
        
        # this might not be correct
        self.num_link = len(self.sys.link_names)
        
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
    
    def raw_pipeline_init(q, qd, x, xd):
        return brax.base.State(
            q=q,
            qd=qd,
            x=x,
            xd=xd,
            contact=None,
            com=jp.zeros(3),
            cinr=brax.base.Inertia(
                brax.base.Transform.zero((self.num_links,)),
                jp.zeros((self.num_links, 3, 3)),
                jp.zeros((self.num_links,)),
            ),
            cd=brax.base.Motion.zero((self.num_links,)),
            cdof=brax.base.Motion.zero((self.num_links,)),
            cdofd=brax.base.Motion.zero((self.num_links,)),
            mass_mx=jp.zeros((self.qd_size, self.qd_size)),
            mass_mx_inv=jp.zeros((self.qd_size, self.qd_size)),
            con_jac=jp.zeros(()),
            con_diag=jp.zeros(()),
            con_aref=jp.zeros(()),
            qf_smooth=jp.zeros_like(qd),
            qf_constraint=jp.zeros_like(qd),
            qdd=jp.zeros_like(qd),
        )
    
    def pipeline_parse(
        self, q, qd, x, xd, com, 
        cinr_transform_pos, cinr_transform_rot, cinr_i, cinr_mass, 
        cd_ang, cd_vel, cdof_ang, cdof_vel, 
        cdofd_ang, cdofd_vel, mass_mx_inv, 
        con_jac, con_aref, con_diag, 
        qf_smooth, qf_constraint, qdd
    ):
        return brax.generalized.base.State(
            q=q,
            qd=qd,
            x=x,
            xd=xd,
            contact=None,
            com=com,
            cinr=brax.base.Inertia(
                transform=brax.base.Transform(
                    cinr_transform_pos, cinr_transform_rot
                ),
                i=cinr_i,
                mass=cinr_mass
            ),
            cd=brax.base.Motion(
                ang=cd_ang,
                vel=cd_vel
            ),
            cdof=brax.base.Motion(
                ang=cdof_ang,
                vel=cdof_vel
            ),
            cdofd=brax.base.Motion(
                ang=cdofd_ang,
                vel=cdofd_vel
            ),
            mass_mx=jp.zeros_like(mass_mx_inv),
            mass_mx_inv=mass_mx_inv,
            con_jac=con_jac,
            con_aref=con_aref,
            con_diag=con_diag,
            qf_smooth=qf_smooth,
            qf_constraint=qf_constraint,
            qdd=qdd
        )
    
    def step(self, s: jp.ndarray, action: jp.ndarray, 
             x_pos, x_rot, xd_ang, xd_vel, com, 
             cinr_transform_pos, cinr_transform_rot, cinr_i, cinr_mass, 
             cd_ang, cd_vel, cdof_ang, cdof_vel, 
             cdofd_ang, cdofd_vel, mass_mx_inv, 
             con_jac, con_aref, con_diag, 
             qf_smooth, qf_constraint, qdd) -> env.State:
        
        # pipeline_state0 = (
        #     jp.zeros((16, 13, 3)),     # cinr.transform.pos
        #     jp.zeros((16, 13, 4)),     # cinr.transform.rot
        #     jp.zeros((16, 13, 3, 3)),  # cinr.i
        #     jp.zeros((16, 13)),        # cinr.mass
        #     jp.zeros((16, 13, 3)),     # cd.ang
        #     jp.zeros((16, 13, 3)),     # cd.vel
        #     jp.zeros((16, 18, 3)),     # cdof.ang
        #     jp.zeros((16, 18, 3)),     # cdof.vel
        #     jp.zeros((16, 18, 3)),     # cdofd.ang
        #     jp.zeros((16, 18, 3)),     # cdofd.vel
        #     jp.zeros((16, 18, 18)),    # mass_mx_inv
        #     jp.zeros((16, 1092, 18)),  # con_jac
        #     jp.zeros((16, 1092)),      # con_aref
        #     jp.zeros((16, 1092)),      # con_diag
        # )
        q = s[:self.q_size]
        qd = s[self.q_size:]
        x = brax.base.Transform(
            x_pos, x_rot
        )
        xd = brax.base.Motion(
            xd_ang, xd_vel
        )
        
        pipeline_state = self.pipeline_parse(q, qd, x, xd, com, 
            cinr_transform_pos, cinr_transform_rot, cinr_i, cinr_mass, 
            cd_ang, cd_vel, cdof_ang, cdof_vel, 
            cdofd_ang, cdofd_vel, mass_mx_inv, 
            con_jac, con_aref, con_diag, 
            qf_smooth, qf_constraint, qdd)
        
        return self._step(pipeline_state, action)
        
    
    def _step(self, state: brax.base.State, action: jp.ndarray) -> env.State:
        """Run one timestep of the environment's dynamics."""
        # low level control turning reference position (input action) to torque:
        joint_pos = state.q[7:]
        joint_vel = state.qd[6:]
        e_pos = action - joint_pos
        e_v = jp.zeros(12) - joint_vel

        feedback_abduction = e_pos*120.0 + e_v*0.5
        abduction_action = jp.array([feedback_abduction[j] for j in [0, 3, 6, 9]])

        feedback_hip = e_pos*80.0 + e_v*1.0
        hip_action = jp.array([feedback_hip[j] for j in [1, 4, 7, 10]])

        feedback_knee = e_pos*120.0 + e_v*2.0
        knee_action = jp.array([feedback_knee[j] for j in [2, 5, 8, 11]])

        action = jp.array([abduction_action, hip_action, knee_action]).T.reshape(-1)
        
        pipeline_state = self.pipeline_step(state, action)
        obs = self._get_obs(pipeline_state)
        
        reward, done, zero = jp.zeros(3)
        info = {}
        
        return env.State(pipeline_state, obs, reward, done, info)