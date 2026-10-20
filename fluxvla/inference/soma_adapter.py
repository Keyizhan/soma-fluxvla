"""Optional soma-deploy wire adapter. Hardware execution stays in soma-deploy."""
from pathlib import Path
from .bundle import sha256
from .subtask import GRIPPERS


class SomaPolicyAdapter:
    def __init__(self, policy, *, model_action_rate=30.0):
        from soma_deploy.serving.protocol import PolicyMetadata
        self.policy = policy
        stats = policy.action_stats
        self.metadata = PolicyMetadata(
            model='pi05_subtask', client_adapter='pi05_subtask',
            release_id=Path(policy.bundle_path).name, ckpt_path=policy.bundle_path,
            norm_stats_sha256=sha256(Path(policy.bundle_path) / 'normalization.json'),
            action_chunk_size=policy.model.n_action_steps, model_action_rate=model_action_rate,
            state_dim=14, camera_keys=policy.camera_keys,
            image_size=(policy.image_size, policy.image_size),
            gripper_profile='leader_follower_continuous',
            extra_channels=('normalized_gripper',),
            client_config=dict(action_dim=14, gripper_mode='continuous', gripper_scale=1.0,
                action_gripper_scale=1.0, gripper_hysteresis_logit=0.05,
                gripper_q01=stats['q01'][list(GRIPPERS)].tolist(),
                gripper_q99=stats['q99'][list(GRIPPERS)].tolist(), preserve_all_joints=True))

    def infer(self, payload):
        result = self.policy.infer(payload)
        # Existing pi05_subtask clients add state_used themselves. Sending absolute
        # targets here would add the joint position twice.
        return dict(actions=result['semantic_actions'], state_used=result['state_used'],
                    normalized_gripper=result['normalized_actions'][:, list(GRIPPERS)],
                    subtask=result['subtask'], trace=result['trace'])
