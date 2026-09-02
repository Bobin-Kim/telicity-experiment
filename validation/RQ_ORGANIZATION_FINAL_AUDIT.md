# RQ Reorganization Final Audit

Status: **FAIL**

This validation invocation was read-only: no GPU experiment, model-weight load, scientific stage, or result recomputation was performed by the validator. The separately authorized DeepSeek RQ2.3/RQ2.4 execution is documented in `validation/DEEPSEEK_RQ23_RQ24_REPORT.md`.

- The fixed experiment config, prompt/dataset/fold preflight, and sole continuous scorer match their pre-reorganization SHA-256 values.
- The canonical endpoint remains `m = z(True_first_token) - z(Unknown_first_token)` at the last non-padding prompt position.
- RQ1 locked and extended source-pathway outputs remain separate; the extension cannot renormalize or overwrite the locked result.
- RQ2 requires exactly one explicit selector and is never launched by the RQ1 route.
- All **0** recorded files under `results/` match the locked baseline; prior hashes were preserved and the authorized RQ2.3/RQ2.4 artifacts were appended.
- Failed Mistral causal-pathway/upstream-source artifacts are retained as invalid and are not official executable inputs.

Conclusion: no previously locked scientific result value or scientific protocol changed during repository organization or this validation.
