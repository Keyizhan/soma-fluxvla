"""Serve shared inference over the optional soma-deploy loopback protocol."""
import argparse
import json
from pathlib import Path
from fluxvla.inference.subtask import InferenceConfig, SubtaskPolicy


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', required=True)
    p.add_argument('--config', default='configs/deploy/subtask_text.json')
    p.add_argument('--device', default='cuda')
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, default=10093)
    p.add_argument('--allow-legacy-contract', action='store_true')
    a = p.parse_args()
    from soma_deploy.serving.protocol import require_loopback
    from soma_deploy.serving.ws_server import PolicyServer
    from fluxvla.inference.soma_adapter import SomaPolicyAdapter
    require_loopback(a.host)
    policy = SubtaskPolicy.from_bundle(a.bundle,
        config=InferenceConfig(**json.loads(Path(a.config).read_text())), device=a.device,
        allow_legacy_contract=a.allow_legacy_contract)
    adapter = SomaPolicyAdapter(policy)
    PolicyServer(adapter.metadata, adapter.infer, host=a.host, port=a.port,
                 idle_timeout_s=-1).serve_forever()


if __name__ == '__main__':
    main()
