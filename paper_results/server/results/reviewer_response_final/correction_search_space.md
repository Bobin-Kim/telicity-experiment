# 실제 correction search space와 selection provenance

새 search는 수행하지 않았다. 아래는 기존 코드/config/log에서 확인된 내용이다.

| Model | Detector candidates | Threshold candidates/rule | Site/component/direction candidates | Dose candidates | Preservation/selection | Candidate count / caveat |
|---|---|---|---|---|---|---|
| Qwen | 기존 TA gate | 0.3–0.9 (0.1 간격), 최종 0.7 | `fixed_l27`, `adaptive_l27`, `l20_only`, `l20_l27`, `l20_heads_l27`; 최종 `l20_heads_l27` | 각 명명 method에 고정된 −1/−2 SD 또는 adaptive | 완료된 held-out method 결과에서 strict D-preserving winner 선택 | 코드상 baseline 포함 13 실행 specs; Cartesian count는 해당 없음. **HELD_OUT_POST_SELECTION** |
| Mistral | answer-residual layer; nested-selected L13/L14/L15 | training/development score에서 strict threshold | final RMSNorm output의 exact True−Unknown readout | raw L2 strength 62 또는 64 (strict selected) | validation A/B/D changes=0인 strict row | 전체 원 후보 grid combination 수는 저장 config에서 **UNKNOWN**; 별도 immutable preregistration 미확인 |
| DeepSeek | L17:H8 head gate | fold별 1.1161–1.2854 | 후보 family 중 최종 residual L29, outer-train margin-covariance direction | 원 후보 grid 4/5/6/8 SD; 최종 nested family는 2 SD | 다른 4 outer folds에서 ≤5pp A/B/D, C/total 최대, 작은 dose, family order | 전체 Cartesian count **UNKNOWN**; **post-hoc safety-buffer/exploratory provenance** 유지 |
| Llama | 25/50/75% depth + frozen actuator layer (fold별 4 layers) | inner-validation non-C maximum 바로 위; zero validation FP에서 C recall/AUROC/layer tie-break | frozen fold-specific RQ2-core residual actuator | 1/2/3 SD; 최소 valid dose | validation C gain>0, A/B/D loss≤1pp | fold당 detector 4 × dose 3 = 12 evaluated pairings; preregistration chronology는 artifact timestamp로 확정 불가 |
| Gemma | 25/50/75% depth + frozen actuator layer (fold별 4 layers) | Llama와 동일 | frozen fold-specific RQ2-core residual actuator | 1/2/3 SD; 최소 valid dose | Llama와 동일 | fold당 12 pairings; preregistration chronology는 확정 불가 |

Held-out fold는 Qwen의 위 caveat를 제외하면 최종 평가에만 사용되었다고 저장 selection/config가 기록한다. DeepSeek family selection은 다른 네 outer-fold 결과를 사용한 leave-one-outer-fold-out 절차이며, 원 논문의 safety-buffer 성격은 confirmatory로 재명명하지 않는다.

Sources: `telicity_intervention/run_qwen_experiment6_adaptive.py`; `results/Mistral_RQ2_experiments_3_4_5/selective_correction/correction_selected_configs.csv`; `.../detector_nested_selection.csv`; `/home/bobin/MULTIMODEL_TRUE_UNKNOWN_PREFERENCE_RUNNER/results/deepseek-7b-rq2/rq2_4_extended/{selected_configs,internal_selected_configs}.csv`; `results/rq3_llama_gemma_confirmatory/{llama,gemma}/model_adaptive/fold*/selection.json`.
