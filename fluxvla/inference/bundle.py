"""Small, torch-free deployment integrity and conditioning checks."""
import hashlib
import json
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_bundle_files(path):
    root = Path(path).resolve()
    manifest = json.loads((root / "DEPLOYABLE.json").read_text())
    required = {"model.safetensors", "model_config.json", "normalization.json",
                "inference_contract.json", "tokenizer/tokenizer_config.json"}
    if not required.issubset(manifest["files"]):
        raise ValueError("bundle is missing required assets")
    for name, expected in manifest["files"].items():
        target = (root / name).resolve()
        if not target.is_relative_to(root):
            raise ValueError("bundle asset escapes its directory")
        if (not target.is_file() or target.stat().st_size != expected["size"]
                or sha256(target) != expected["sha256"]):
            raise ValueError(f"bundle integrity check failed: {name}")
    return manifest


def validate_conditioning(contract, config, *, allow_legacy_contract=False):
    if (contract.get("state_dim") != 14 or contract.get("action_dim") != 14
            or contract.get("padded_action_dim") != 32):
        raise ValueError("unsupported action/state dimensions")
    expected = contract.get("conditioning")
    if expected is None:
        if not allow_legacy_contract:
            raise ValueError("legacy bundle lacks an explicit training conditioning contract; "
                             "use allow_legacy_contract after checking its training source")
        if config.prefix_source != "text":
            raise ValueError("legacy override only supports the audited text training path")
        return
    if expected.get("training_action_prefix") != "text_bidirectional":
        raise ValueError("unsupported training action prefix")
    if contract.get("low_level_prefix_source") != "text":
        raise ValueError("export default conflicts with its training action prefix")
    if config.prefix_source == "generation":
        # Allowed only as an explicit runtime experiment; never the export default.
        if not expected.get("generation_is_experimental", False):
            raise ValueError("generation experiment is not declared by this export")
