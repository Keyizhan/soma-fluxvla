# Derived from FluxVLA (Limx Dynamics); see LICENSE and NOTICE.
from fluxvla.engines import TRANSFORMS
from .transform_inputs import ProcessParquetInputs
from .transform_actions import ProcessLiberoActions
from .normalize import NormalizeStatesAndActions
from .prompters import PreparePromptWithState
from .transform_prompts import ProcessPrompts
from .transform_images import ResizeImages, SimpleNormalizeImages
from .augment_images import AugmentImages
from .subtask_prompts import PrepareSubtaskPrompts
from .subtask_fast_prompts import PrepareSubtaskFastPrompts
from .subtask_bbox_prompts import PrepareSubtaskBboxPrompts

TRANSFORMS.register_module(module=AugmentImages, force=True)
