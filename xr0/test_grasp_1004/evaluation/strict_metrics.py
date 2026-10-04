"""Boundary-only same-fruit placement observer; never supplies policy inputs."""

class StrictPlacement:
    def __init__(self, held_steps=15, bucket_steps=60):
        self.held_steps=held_steps;self.bucket_steps=bucket_steps
        self.previous_held=None;self.current_held_steps=0;self.last_step=0
        self.fruits={};self.release_events=[];self.success_events=[]
        self.previous_detached=set();self.detach_steps={}
        self.first_legacy_success_step=None
    def update(self, step, held, detached, in_bucket, legacy_success):
        assert step==self.last_step+1, 'Strict observer requires every consecutive control boundary'
        self.last_step=step
        if legacy_success and self.first_legacy_success_step is None:self.first_legacy_success_step=step
        held=None if held is None else int(held)
        detached=set(map(int,detached));in_bucket=set(map(int,in_bucket))
        for fruit in detached-self.previous_detached:self.detach_steps.setdefault(fruit,step)
        self.previous_detached=detached
        if held is not None:
            item=self.fruits.setdefault(held,dict(first_grasp_step=step,held15_step=None,first_detach_after_grasp_step=None,release_step=None,stable_bucket_steps=0,max_stable_bucket_steps=0,strict_success_step=None))
            self.current_held_steps=self.current_held_steps+1 if held==self.previous_held else 1
            if self.current_held_steps>=self.held_steps and item['held15_step'] is None:item['held15_step']=step
            item['release_step']=None
        else:self.current_held_steps=0
        for fruit,item in self.fruits.items():
            detach_step=self.detach_steps.get(fruit)
            if detach_step is not None and detach_step>=item['first_grasp_step'] and item['first_detach_after_grasp_step'] is None:item['first_detach_after_grasp_step']=detach_step
        if self.previous_held is not None and self.previous_held!=held:
            fruit=self.previous_held;item=self.fruits[fruit]
            valid=item['held15_step'] is not None and fruit in detached and item['first_detach_after_grasp_step'] is not None
            self.release_events.append(dict(apple_id=fruit,step=step,had_held15=item['held15_step'] is not None,detached_at_release=fruit in detached,valid_chain=valid))
            item['release_step']=step if valid else None
        for fruit,item in self.fruits.items():
            stable=item['release_step'] is not None and held!=fruit and fruit in detached and fruit in in_bucket
            item['stable_bucket_steps']=item['stable_bucket_steps']+1 if stable else 0
            item['max_stable_bucket_steps']=max(item['max_stable_bucket_steps'],item['stable_bucket_steps'])
            if item['stable_bucket_steps']>=self.bucket_steps and item['strict_success_step'] is None:
                item['strict_success_step']=step
                self.success_events.append(dict(apple_id=fruit,step=step,release_step=item['release_step'],held15_step=item['held15_step'],first_detach_after_grasp_step=item['first_detach_after_grasp_step']))
        self.previous_held=held
        return self.summary()
    def summary(self):
        return dict(strict_success=bool(self.success_events),strict_success_events=list(self.success_events),legacy_success_ever=self.first_legacy_success_step is not None,first_legacy_success_step=self.first_legacy_success_step,fruit_chains={str(k):dict(v) for k,v in self.fruits.items()},release_events=list(self.release_events),held_required_boundaries=self.held_steps,bucket_required_boundaries=self.bucket_steps,boundary_only=True)

def fruit_snapshot(env):
    # Exact geometry from OrchardVLAEnv._info; observation only.
    import numpy as np
    from treesim import robot
    cp,cr=env._chassis_pose()
    positions=env.sim.body_q_np()[np.asarray(env.tm.apple_bodies),:3]
    local=cr.inv().apply(positions-cp)
    floor=robot._CHASSIS_Z+robot._CHASSIS[2]
    detached=np.asarray(env.sim.apples.detached,dtype=bool)
    bucket=((np.abs(local[:,0]-robot._BUCKET_CENTER_X)<robot._BUCKET_HALF)
            &(np.abs(local[:,1])<robot._BUCKET_HALF)
            &(local[:,2]>floor)&(local[:,2]<floor+robot._BUCKET_WALL_H)&detached)
    return dict(detached_apple_ids=np.flatnonzero(detached).tolist(),in_bucket_apple_ids=np.flatnonzero(bucket).tolist(),held_apple_id=None if env._held is None else int(env._held))

def make_continuing_env(base):
    class ContinueAfterBucketEnv(base):
        def step(self,*args,**kwargs):
            obs,reward,terminated,truncated,info=super().step(*args,**kwargs)
            info['legacy_terminated']=bool(terminated)
            info['legacy_success_continuation']=bool(terminated and not truncated)
            if terminated and not truncated:self._done=False
            # Physics, controller, reward and original success field are unchanged.
            return obs,reward,False,bool(truncated),info
    return ContinueAfterBucketEnv

def gt_segments(chunks,processed_targets):
    assert processed_targets in (5,30)
    for first,end,anchor,adapter in chunks:
        for start in range(first,end,processed_targets):
            yield start,min(start+processed_targets,end),anchor,adapter
