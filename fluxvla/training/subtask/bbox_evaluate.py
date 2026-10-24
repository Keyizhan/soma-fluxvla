"""Free-running grounding evaluation without teacher-forced targets or actions."""

from collections import Counter

import torch

from fluxvla.transforms.bbox_grounding import localization_metrics


@torch.inference_mode()
def evaluate_grounding(model, prompts, fixtures, indices, image_size, device, max_new_tokens=256):
    totals = Counter()
    was_training = model.training
    model.eval()
    try:
        for index in indices:
            record = fixtures.records[index]
            context = prompts.context(record["overall_task"], fixtures.states[index][:14])
            image = fixtures.image(index, image_size).to(device)[None]
            if image.dtype == torch.uint8:
                image = image.float() / 127.5 - 1
            args = {k: torch.as_tensor(v, device=device)[None] for k, v in context.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                generated = model.generate_subtask(
                    image,
                    args["input_ids"],
                    args["valid_mask"],
                    torch.as_tensor(fixtures.img_masks[index], device=device)[None],
                    eos_token_id=prompts.eos_id,
                    pad_token_id=prompts.pad_id,
                    ar_mask=args["ar_mask"],
                    max_new_tokens=max_new_tokens,
                    use_cache=True,
                    stop_token_ids=prompts.stop_token_ids(),
                )
            ids = generated["token_ids"][0, : int(generated["lengths"][0])].tolist()
            prediction = prompts.decode_subtask(ids)
            totals.update(localization_metrics(prediction, record["grounding_json"]))
    finally:
        model.train(was_training)

    def ratio(n, d):
        return totals[n] / totals[d] if totals[d] else None

    return dict(
        totals=totals,
        format_accuracy=ratio("valid_format", "examples"),
        type_accuracy=ratio("type_correct", "goals"),
        end_to_end_iou=ratio("iou_sum", "bbox_count"),
        valid_box_iou=ratio("iou_sum", "valid_bbox_count"),
        valid_box_center_error_q1000=ratio("center_error_sum", "valid_bbox_count"),
        missed_box_rate=ratio("missed_bbox", "bbox_count"),
    )
