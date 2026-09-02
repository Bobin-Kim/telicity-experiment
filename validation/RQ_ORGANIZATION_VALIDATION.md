# RQ1/RQ2 Organization Validation

Status: **PASS**

This validation invocation loaded no model weights and ran no GPU experiment or scientific stage. Result files were read only for hash verification.

## Checks

- PASS — `yaml_parsing`
- PASS — `shared_model_registry`
- PASS — `model_alias_resolution`
- PASS — `rq1_locked_route`
- PASS — `rq1_extension_is_optional`
- PASS — `rq1_never_routes_rq2`
- PASS — `rq2_has_six_subquestions`
- PASS — `rq2_excludes_rq1_and_legacy_upstream`
- PASS — `rq2_requires_explicit_subquestion`
- PASS — `locked_extended_pathways_separate`
- PASS — `no_alternate_executable_scoring_stage`
- PASS — `canonical_scoring_locked`
- PASS — `scientific_protocol_files_unchanged`
- PASS — `rq2_cli_no_selector_fails`
- PASS — `rq1_plan_dry_run`
- PASS — `rq1_extension_plan_dry_run`
- PASS — `rq2_plan_dry_run`
- PASS — `rq25_completed_artifact_reuse`
- PASS — `rq1_tokenizer_config_dry_run`
- PASS — `rq2_tokenizer_config_dry_run`
- PASS — `existing_result_hashes_preserved`
- PASS — `manifest_covers_every_local_result`
- PASS — `manifest_schema_complete`
- PASS — `manifest_paths_valid`
- PASS — `failed_mistral_artifacts_invalid`
- PASS — `rq1_manifest_has_locked_and_extension`

Result files hash-checked: **1347**.
Failed Mistral decision-alignment and upstream-origin artifacts remain classified invalid and are absent from official executable routes.
