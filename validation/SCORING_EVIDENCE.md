# Locked scoring evidence

Selected mode: `first_answer_token_logit_margin`, defined as `z(True_first_token) - z(Unknown_first_token)` at the final non-padding prompt token.

Strongest canonical evidence:

- `run_qwen_rq2bc_continuous.py:35` constructs A23's target as cached label logit column 0 minus column 2.
- The same source at line 91 computes live intervention margin as `lg[j,0]-lg[j,2]`.
- `run_qwen_full_depth_true_signal.py:61` constructs each label ID with `tokenizer(...).input_ids[0]`, proving those columns are first verbalizer tokens.

The older behavioral sequence scorer is not the A23/A35 target and is not called by this package. `FirstAnswerTokenMargin` is instantiated once and passed to every runtime stage. Multi-token verbalizers are audited in full, while their locked scoring token is explicitly recorded and required to differ between True and Unknown.
