"""Live-state initialization only; inherited production expert stages and physics."""
import bootstrap
import numpy as np
from treesim.fixed_base_picker import FixedBaseAutoPicker
from treesim.picker import AutoPicker
from treesim.arm_motion import ArmMotionProfile
from recovery_common import physics_digest, truth

class LiveRecoveryTeacher(FixedBaseAutoPicker):
 def __init__(self,env,category,profile,ik=None):
  before=physics_digest(env);self.takeover=truth(env)
  # Avoid FixedBaseAutoPicker's clean-reset pose check. AutoPicker constructor
  # only allocates bookkeeping; initialize measured command state immediately.
  AutoPicker.__init__(self,env.sim,env.tm,None,env.driver,env.driver.rp,ik=ik,arm_motion_profile=ArmMotionProfile(**profile))
  self.fail_reason=None;self.last_ik=None;self.planned_apple=int(env._reset_stance['apple_index']);self.stance=dict(env._reset_stance)
  self._q_cmd=env._obs['joint_pos'].astype(float).copy();self._q_goal=self._q_cmd.copy()
  if self.arm_motion is not None:self.arm_motion.reset(self._q_cmd[:7])
  self._target=env.sim.body_q_np()[int(env.tm.apple_bodies[self.planned_apple]),:3].copy()
  held=np.flatnonzero(self.apples._held_host[:self.apples.n])
  if len(held):
   assert held.tolist()==[self.planned_apple]
   self._target_apple=self.planned_apple
   self._retract_from=self._tcp_world().copy();self._retract_dist=0.
   initial='TRANSPORT' if self.apples.detached[self.planned_apple] else 'PULL'
   # Preserve width at initialization. First regular phase update maintains
   # closed command for a fruit already held, no release/re-hold spring call.
  else:
   assert category=='R1' and not self.apples.detached[self.planned_apple]
   initial='GRASP' if self.takeover['tcp_fruit_distance']<=.09 else 'REACH'
  self._goto(initial);self.initial_state=initial
  assert np.array_equal(self._q_cmd,env._obs['joint_pos'])
  after=physics_digest(env);assert before==after,'Teacher initialization mutated physics'
  self.initialization_audit=dict(physical_digest_before=before,physical_digest_after=after,physics_unchanged=True,command_joints_initialized_from_measured=True,motion_limiter_initialized_from_measured=True,initial_state=initial,no_home_reset=True,no_teleport=True)
 def _st_pull(self):
  self._fingers(self.FINGER_CLOSED)
  super()._st_pull()
 def _st_transport(self):
  self._fingers(self.FINGER_CLOSED)
  super()._st_transport()
