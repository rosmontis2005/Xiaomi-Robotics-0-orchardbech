"""Read-only rollout diagnostics; never solve IK, update intent, or advance physics.

All return values are JSON-compatible. Boundary snapshots can miss contacts within
one control step (the environment runs multiple physics frames per step); use the
real step info.grasp_assist_triggered as the authoritative event indicator.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


PALM_LOWER = np.array([-.025, -.04, .065], dtype=float)
PALM_UPPER = np.array([.025, .04, .125], dtype=float)
PALM_CENTER = (PALM_LOWER + PALM_UPPER) / 2.


def _palm_inside(local):
    # Keep strict inequalities identical to OrchardVLAEnv._update_assist.
    return bool(abs(local[0]) < .025 and abs(local[1]) < .04
                and .065 < local[2] < .125)


def snapshot_grasp(env):
    """Inspect existing contacts and geometric/intent gates without mutation.

    contact_apples contains apples touching either finger, not merely apples
    touching arbitrary tree/robot bodies. eligible_apple_ids applies the same
    two-finger plus palm-volume predicate as _update_assist; would_attach_now
    additionally applies mode, held-state, intent, and unique-candidate gates.
    Nearest-apple fields are diagnostic only and never choose a policy target.
    """
    obs = env._obs
    poses = env.sim.body_q_np()
    hand = poses[env.wrist]
    hand_inv = Rotation.from_quat(hand[3:]).inv()
    apple_bodies = [int(b) for b in env.tm.apple_bodies]
    apple_to_index = {body: i for i, body in enumerate(apple_bodies)}
    fingers_expected = set(int(b) for b in env.finger_bodies)
    contacts = env.sim.contacts
    contact_count = int(contacts.rigid_contact_count.numpy()[0])
    shape_body = env.tm.model.shape_body.numpy()
    shape0 = contacts.rigid_contact_shape0.numpy()[:contact_count]
    shape1 = contacts.rigid_contact_shape1.numpy()[:contact_count]
    touched = {}
    pair_counts = {}
    for a, b in zip(shape0, shape1):
        if a < 0 or b < 0:
            continue
        ba, bb = int(shape_body[a]), int(shape_body[b])
        for finger, fruit in ((ba, bb), (bb, ba)):
            if finger in fingers_expected and fruit in apple_to_index:
                touched.setdefault(fruit, set()).add(finger)
                key = (fruit, finger)
                pair_counts[key] = pair_counts.get(key, 0) + 1

    tcp_position = np.asarray(obs['tcp_pos_world'], dtype=float)
    apples = []
    contact_apples = []
    eligible = []
    for body in apple_bodies:
        apple_id = apple_to_index[body]
        world = np.asarray(poses[body, :3], dtype=float)
        local = hand_inv.apply(world - hand[:3])
        touching_fingers = touched.get(body, set())
        both = touching_fingers == fingers_expected and len(touching_fingers) == 2
        inside = _palm_inside(local)
        geometry_eligible = both and inside
        entry = dict(
            apple_id=apple_id,
            apple_body=body,
            world_position=world.tolist(),
            hand_local_position=local.tolist(),
            tcp_distance_m=float(np.linalg.norm(world - tcp_position)),
            palm_center_distance_m=float(np.linalg.norm(local - PALM_CENTER)),
            palm_volume_distance_m=float(np.linalg.norm(
                np.maximum(PALM_LOWER - local, 0.) + np.maximum(local - PALM_UPPER, 0.))),
            inside_palm_volume=inside,
            touching_finger_bodies=sorted(touching_fingers),
            both_fingers_contact=bool(both),
            geometry_eligible=bool(geometry_eligible),
        )
        apples.append(entry)
        if touching_fingers:
            contact_entry = dict(entry)
            contact_entry['contact_pair_counts'] = {
                str(finger): pair_counts[(body, finger)] for finger in sorted(touching_fingers)
            }
            contact_apples.append(contact_entry)
        if geometry_eligible:
            eligible.append(apple_id)

    intent = float(env._gripper)
    held = None if env._held is None else int(env._held)
    assist_enabled = env.config.grasp_mode != 'contact'
    intent_allows_attach = intent < 0.
    attach_allowed = assist_enabled and held is None and intent_allows_attach
    would_attach = attach_allowed and len(eligible) == 1
    return dict(
        grasp_mode=env.config.grasp_mode,
        gripper_intent=intent,
        commanded_width_m=float(env._gripper_width_command),
        measured_width_m=float(obs['gripper_width']),
        width_open_steps=int(env._width_open_steps),
        held_apple_id=held,
        assist_enabled=assist_enabled,
        intent_allows_attach=intent_allows_attach,
        would_release_now=bool(assist_enabled and intent > 0. and held is not None),
        expected_finger_bodies=sorted(fingers_expected),
        rigid_contact_count=contact_count,
        contact_apples=contact_apples,
        eligible_apple_ids=eligible,
        would_attach_now=bool(would_attach),
        would_attach_apple_id=eligible[0] if would_attach else None,
        ambiguous_eligible_apples=len(eligible) > 1,
        nearest_tcp_apple=min(apples, key=lambda a: a['tcp_distance_m']) if apples else None,
        nearest_palm_center_apple=min(apples, key=lambda a: a['palm_center_distance_m']) if apples else None,
        hand_position_world=np.asarray(hand[:3]).tolist(),
        hand_quaternion_world=np.asarray(hand[3:]).tolist(),
        tcp_position_world=tcp_position.tolist(),
    )


def snapshot_command(env, command):
    """Predict clipping components using the exact env.step pre-IK arithmetic.

    Call before env.step(**command). Does not solve IK or change cached targets.
    The post-step info.action_clipped should equal any_action_clipping for the
    unmodified environment. Internal joint-step limiting is a different metric.
    """
    action = np.asarray(command['action'], dtype=float)
    if action.shape != (7,) or not np.isfinite(action).all():
        raise ValueError('Expected finite native action with shape (7,)')
    config = env.config
    limits = np.array([config.max_translation] * 3 + [config.max_rotation] * 3 + [1.])
    clipped = np.clip(action, -limits, limits)
    obs = env._obs
    if np.any(clipped[:6]):
        target = np.asarray(obs['tcp_pos_world']) + clipped[:3]
        rotation = (Rotation.from_euler('xyz', clipped[3:6])
                    * Rotation.from_quat(obs['tcp_quat_world']))
        reused_cached_pose = False
    else:
        target = env._target_position.copy()
        rotation = env._target_rotation
        reused_cached_pose = True
    chassis_position, chassis_rotation = env._chassis_pose()
    target_base = chassis_rotation.inv().apply(target - chassis_position)
    bounded = np.clip(target_base, config.workspace_lower, config.workspace_upper)
    workspace_clipped = not np.allclose(target_base, bounded, rtol=0, atol=1e-9)
    workspace_target = chassis_position + chassis_rotation.apply(bounded)
    width_request = command.get('gripper_width')
    width_clipped = False
    width_bounded = None
    next_intent = float(env._gripper)
    next_open_steps = 0
    width_decrease = None
    if width_request is not None:
        width_request = float(width_request)
        if not np.isfinite(width_request):
            raise ValueError('Expected finite gripper width')
        width_bounded = float(np.clip(width_request, 0., .08))
        width_clipped = width_bounded != width_request
        next_open_steps = (int(env._width_open_steps) + 1
                           if width_bounded >= config.gripper_open_width else 0)
        width_decrease = float(env._gripper_width_command) - width_bounded
        if next_open_steps >= config.gripper_open_steps:
            next_intent = 1.
        elif width_bounded < env._gripper_width_command - config.gripper_close_deadband:
            next_intent = -1.
    elif clipped[6] != 0.:
        next_intent = float(np.sign(clipped[6]))
    components = dict(
        translation=bool(not np.array_equal(action[:3], clipped[:3])),
        rotation=bool(not np.array_equal(action[3:6], clipped[3:6])),
        discrete_gripper=bool(action[6] != clipped[6]),
        width=bool(width_clipped),
        workspace=bool(workspace_clipped),
    )
    return dict(
        requested_native_action=action.tolist(),
        bounded_native_action=clipped.tolist(),
        native_action_removed=(action - clipped).tolist(),
        translation_removed_norm_m=float(np.linalg.norm(action[:3] - clipped[:3])),
        rotation_euler_removed_norm_rad=float(np.linalg.norm(action[3:6] - clipped[3:6])),
        clipping_components=components,
        any_action_clipping=any(components.values()),
        requested_width_m=width_request,
        bounded_width_m=width_bounded,
        width_removed_m=None if width_request is None else width_request - width_bounded,
        previous_commanded_width_m=float(env._gripper_width_command),
        width_command_decrease_m=width_decrease,
        predicted_gripper_intent=next_intent,
        predicted_width_open_steps=next_open_steps,
        reused_cached_pose=reused_cached_pose,
        target_before_workspace_world=np.asarray(target).tolist(),
        target_before_workspace_chassis=np.asarray(target_base).tolist(),
        target_after_workspace_world=np.asarray(workspace_target).tolist(),
        target_after_workspace_chassis=np.asarray(bounded).tolist(),
        workspace_removed_norm_m=float(np.linalg.norm(target_base - bounded)),
        target_quaternion_world=rotation.as_quat().tolist(),
    )


def joint_command_delta(pre, post, *, max_joint_step, action_repeat):
    """Observe final joint-command movement; do not infer internal saturation.

    pre/post are copies of env._joint_command on either side of env.step. Equality
    with the accumulated per-frame limit is evidence of a ceiling, not a count
    of individual physics-frame clamps; IK failure and joint bounds also matter.
    """
    pre = np.asarray(pre, dtype=float)
    post = np.asarray(post, dtype=float)
    if pre.shape != post.shape or pre.ndim != 1:
        raise ValueError('Joint command snapshots must be equal-shape vectors')
    delta = post - pre
    ceiling = float(max_joint_step) * int(action_repeat)
    at_ceiling = np.abs(delta) >= ceiling - 1e-8
    return dict(
        pre_joint_command=pre.tolist(),
        post_joint_command=post.tolist(),
        delta=delta.tolist(),
        max_abs_delta=float(np.max(np.abs(delta))) if delta.size else 0.,
        accumulated_per_control_step_limit=ceiling,
        indices_at_accumulated_limit=np.flatnonzero(at_ceiling).tolist(),
        internal_joint_clipping_count=None,
    )
