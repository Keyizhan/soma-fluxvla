"""Generated label parsing; preserve full raw tokens for audit."""
def cut_at_action_sentinel(text: str) -> str:
    """Stop the subtask at the ``Action:`` boundary the training sequence puts after it.

    The model is trained on ``Subtask: <text>;\\nAction: <FAST>|`` and never emits EOS
    directly after the subtask, so a plain ``max_new_tokens`` stop leaves either a
    truncated label or a tail of FAST tokens.
    """
    lowered = text.lower()
    end = len(text)
    for marker in ("action:", "\n"):
        index = lowered.find(marker)
        if index >= 0:
            end = min(end, index)
    return text[:end].strip()


def subtask_for_action_prompt(text: str) -> str:
    """Remove the generated field delimiter before rebuilding a training prompt.

    High-level training emits ``clean_reference(label) + ';'``. Passing that
    delimiter back to ``clean_reference`` would turn the final ``.;`` into
    ``..``. Keep the decoded output intact in logs; strip only this delimiter
    from the text supplied to the action prompt builder.
    """
    label = text.strip().removesuffix(";").rstrip()
    if not label:
        raise ValueError("generated subtask has no label before its delimiter")
    return label

