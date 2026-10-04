# RQ2 Table 5 `R_gain` strict audit

## [A] 재계산된 정확한 1-SD 값

아래 값은 새 multi-dose actual-forward 원시 행에서 Group C를 fold별로 계산한 뒤 5개 fold를 동일 가중 평균한 결과다. Telicity 값은 새 decision multi-dose run이 직접 생성한 것이 아니라 `reviewer_response_robustness/rq2_telicity_multidose/per_example.csv`를 재사용한다.

| Model | G_tel | G_dec (new multi-dose) | 기계적 R_gain |
|---|---:|---:|---:|
| Qwen | -0.0143750000 | 2.7207080201 | 189.2666 |
| Mistral | -0.0578125000 | 2.1059375000 | 36.4270 |
| DeepSeek | -0.0333593750 | 2.0542968750 | 61.5808 |
| Llama | -0.0070312500 | 1.4725000000 | 209.4222 |
| Gemma | -0.0115234375 | 0.4164843750 | 36.1424 |

원시 margin 평균은 다음과 같다.

| Model | Tel -1 SD mean m | Tel +1 SD mean m | Decision -1 SD mean m | Decision +1 SD mean m |
|---|---:|---:|---:|---:|
| Qwen | 0.738750 | 0.710000 | -1.917500 | 3.523916 |
| Mistral | 17.822188 | 17.706563 | 15.108125 | 19.320000 |
| DeepSeek | 4.953438 | 4.886719 | 2.890156 | 6.998750 |
| Llama | 1.355156 | 1.341094 | -0.125625 | 2.819375 |
| Gemma | 0.687734 | 0.664688 | 0.260469 | 1.093437 |

중요: Qwen과 DeepSeek의 위 비율은 산술적으로는 계산되지만, 새 decision numerator와 Telicity denominator가 동일 intervention site/direction family를 사용하지 않으므로 matched-site Table 5 값으로 사용할 수 없다.

## [B] 기존 Table 5와의 차이

기존 Table 5 생성 경로를 원시 행에서 재현한 결과는 다음과 같다.

| Model | 기존 raw G_dec | raw G_tel | raw R_gain | 논문 반올림 | 새 G_dec | 새 기계적 R_gain |
|---|---:|---:|---:|---:|---:|---:|
| Qwen | 4.6037500000 | -0.0143750000 | 320.2609 | 320.3 | 2.7207080201 | 189.2666 |
| Mistral | 2.1025000000 | -0.0578125000 | 36.3676 | 36.4 | 2.1059375000 | 36.4270 |
| DeepSeek | 0.7485937500 | -0.0333593750 | 22.4403 | 22.4 | 2.0542968750 | 61.5808 |
| Llama | 1.4727343750 | -0.0070312500 | 209.4556 | 209.5 | 1.4725000000 | 209.4222 |
| Gemma | 0.4165234375 | -0.0115234375 | 36.1458 | 36.1 | 0.4164843750 | 36.1424 |

기존 Table 5의 다섯 값은 모두 저장된 원시 actual-forward 결과에서 오차 없이 재현됐다. Qwen과 DeepSeek의 차이는 denominator가 아니라 decision numerator에서 발생한다.

## [C] discrepancy의 정확한 원인

### 1. 기존 Table 5 생성 방식

`results/reviewer_response_robustness/code/aggregate.py`의 `comparisons()`는 다음을 결합한다.

- Qwen/DeepSeek/Llama/Gemma decision numerator: `results/rq2_core_replication/<model>/causal_gain_fold_summary.csv`
- Mistral decision numerator: `results/mistral_l29_same_site_telicity/decision_telicity_comparison.csv`
- Telicity denominator: `results/reviewer_response_robustness/rq2_telicity_multidose/model_summary.csv`
- 공식: `abs(mean_f G_dec,f) / abs(mean_f G_tel,f)`

원시 per-example 행으로 같은 계산을 다시 수행했으며 저장된 `rq2_decision_comparison.csv`와 정확히 일치했다.

### 2. Qwen

- 기존 matched-site pair: zero-based block 27 **block output**에서 C/D-only outer-training 방향을 사용한다.
- 새 multi-dose decision: zero-based block 27 **block input**에서 A/B/C/D training population의 `b1_preference_axes.npy` 방향을 사용한다.
- old/new decision-axis cosine은 fold별 0.420–0.446뿐이다.
- 새 projection SD는 기존의 약 0.746–0.803배다.
- 새 Qwen fold assignment도 기존 canonical fold와 다르다. Group-C fold 크기는 기존 20/20/20/20/20에서 21/21/20/19/19로 변했고, fold별 example set이 모두 불일치한다.
- 따라서 `4.60375 → 2.720708`은 aggregation 오류가 아니라 site, axis, scale, training population, fold assignment가 바뀐 별도 estimand다.

### 3. DeepSeek

- 기존 matched-site pair: folds 0/1/2/4는 zero-based block 25 output, fold 3은 block 26 output이다.
- 새 multi-dose decision: 모든 fold에서 zero-based block 28 output이다.
- old/new decision-axis cosine은 0.570–0.614다.
- 새 projection SD는 기존의 약 2.98–3.47배다.
- held-out example/fold set은 동일하지만 layer, direction, scale, training population이 다르다.
- 따라서 `0.748594 → 2.054297`은 새 downstream axis에 대한 다른 causal gain이다.

### 4. 나머지 모델

- Mistral, Llama, Gemma는 old/new decision axis cosine이 모든 fold에서 사실상 1.0이고 site, scale, example folds가 일치한다.
- 따라서 새 값과 기존 값의 작은 차이는 actual-forward 재실행의 수치/배치 차이 수준이며 substantive discrepancy가 아니다.

### 5. aggregation A–E

Methods와 기존 생성 코드에 대응하는 값은 **E = `abs(mean_f G_dec,f) / abs(mean_f G_tel,f)`**, 즉 B와 같은 ratio-of-fold-means다.

| Model | Regime | A pooled | B ratio of fold means | C mean fold ratios | D mean(abs G_dec)/mean(abs G_tel) | E Methods |
|---|---|---:|---:|---:|---:|---:|
| Qwen | existing Table 5 | 320.2609 | 320.2609 | 262.8712 | 156.7234 | 320.2609 |
| Qwen | new mechanical | 189.9130 | 189.2666 | 156.4957 | 92.6198 | 189.2666 |
| Mistral | existing Table 5 | 36.3676 | 36.3676 | 44.2377 | 36.3676 | 36.3676 |
| DeepSeek | existing Table 5 | 22.4403 | 22.4403 | 32.0403 | 22.4403 | 22.4403 |
| DeepSeek | new mechanical | 61.5808 | 61.5808 | 87.3249 | 61.5808 | 61.5808 |
| Llama | existing Table 5 | 209.4556 | 209.4556 | 237.0506 | 138.6103 | 209.4556 |
| Gemma | existing Table 5 | 36.1458 | 36.1458 | 41.4802 | 36.1458 | 36.1458 |

A와 B는 fold당 N이 같으면 동일하다. 새 Qwen만 fold 크기가 달라 A와 B가 달라진다. C와 D는 논문 Methods의 정의가 아니며, 작은/부호가 다른 fold-level Telicity denominator 때문에 크게 달라진다.

## [D] Methods 정의에 따라 Table 5에 사용해야 할 최종 값

Table 5가 의도한 estimand가 **동일 site에서 각 방향의 ±1 own-training-SD actual-forward causal gain을 비교하는 것**이라면 최종 값은 기존 raw-reproduced 값이다.

| Model | 최종 R_gain |
|---|---:|
| Qwen | **320.2609** (표기 320.3) |
| Mistral | **36.3676** (표기 36.4) |
| DeepSeek | **22.4403** (표기 22.4) |
| Llama | **209.4556** (표기 209.5) |
| Gemma | **36.1458** (표기 36.1) |

새 multi-dose의 189.27과 61.58을 Qwen/DeepSeek Table 5에 넣으면 안 된다. 그 numerator는 기존 Telicity denominator와 matched-site가 아니다. 새 axis 기준 ratio가 필요하다면 같은 새 site, 같은 folds에서 Telicity ±1 SD actual-forward를 별도로 수행해야 하지만, 이번 audit에서는 새 inference를 실행하지 않았다.

## [E] 수정해야 하는 코드/CSV/논문 위치

1. `telicity_intervention/run_rq2_multidose_and_independent_control.py`의 aggregation은 새 decision gain을 이전 Telicity summary와 site/fold compatibility 검사 없이 결합한다. Qwen과 DeepSeek ratio는 `NA_INCOMPATIBLE_SITE`로 처리하거나 명확히 “cross-experiment mechanical ratio”로 이름을 바꿔야 한다.
2. `results/rq2_reviewer_multidose_independent/telicity_vs_decision_multidose.csv`의 Qwen/DeepSeek `absolute_decision_to_telicity_gain_ratio`는 Table 5 대체값으로 사용하지 않아야 한다.
3. `results/reviewer_response_robustness/rq2_decision_comparison.csv`는 기존 matched-site Table 5 값을 정확히 보존하고 있으므로 수정할 필요가 없다.
4. 논문 Table 5 caption/Methods에는 `R_gain = |mean_f G_dec,f| / |mean_f G_tel,f|`, equal-weight five held-out folds, 각 방향 자체 outer-training projection SD, 동일 layer/hook site라고 명시해야 한다.
5. 논문에서 새 multi-dose decision gain을 별도 Table 4에 제시한다면, Qwen과 DeepSeek에 대해서는 Table 5 numerator와 다른 axis/site의 결과임을 주석으로 분리해야 한다.

## [F] AUDIT PASS

**AUDIT = PASS.** 기존 Table 5 값은 raw saved actual-forward 결과와 생성 코드에서 완전히 재현됐다. 불일치는 계산 오류나 반올림 오류가 아니라 Qwen과 DeepSeek의 새 decision intervention definition이 기존 matched-site Table 5 definition과 달라졌기 때문이다. 모델 weight loading, CUDA, 새 forward pass는 사용하지 않았다.

주요 수치 파일: `recomputed_1sd_values.csv`, `existing_gain_reproduction.csv`, `aggregation_A_to_E.csv`, `qwen_deepseek_fold_discrepancy.csv`, `intervention_definition_comparison.csv`, `decision_axis_scale_comparison.csv`, `audit.json`.
