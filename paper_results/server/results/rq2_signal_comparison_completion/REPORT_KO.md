# RQ2 Telicity–Decision Signal Comparison Completion

## 최종 상태

**PASS_CPU_COMPUTATION_REQUIRED**

언어모델·가중치·CUDA·forward pass를 전혀 사용하지 않고, 기존 PASS 완료 산출물의 baseline score와 저장 방향 벡터만 읽었다. GPU 재실행은 필요하지 않다.

## magnitude 정의

공통 1D 재집계는 각 outer fold의 학습 자료에서 중심과 단위 방향을 정하고, 해당 방향의 학습 projection SD로 나눈 점수의 절댓값을 held-out C/D에서 계산한 뒤, fold 내부 평균을 먼저 구하고 5개 fold를 동일 가중 평균한다. 즉 `mean |(x-center)·u| / SD_train((x-center)·u)`이다. Telicity와 decision은 같은 example, answer position, fold-selected residual site를 사용하지만 각자 자기 방향의 train SD로 표준화된다. 이는 개별 모델 내부의 상대적 표준화 magnitude이며 모델 간 raw activation 크기 비교가 아니다.

## 기존 수치와의 관계

- Qwen A19의 저장값은 Telicity `0.903933`, decision `0.898643`이다. N=28 Group-C 분석 집합의 표준화 score이며 Mistral의 low-rank norm과 동일한 통계가 아니다.
- Mistral A33의 저장된 low-rank decision/Telicity 비율은 C `0.985625`, D `1.015540`이다. 동일 Mistral protocol 내부에서는 유효하지만 Qwen A19 1D 통계와 합치면 안 된다.
- DeepSeek의 이전 34--37배 magnitude는 protocol re-audit에서 final RMSNorm과 raw block output 혼합으로 판정되어 철회되었다. 본 출력은 그 값을 사용하지 않는다.
- matched ±1 SD intervention dose는 주입량의 표준화 방식일 뿐, 두 raw representation magnitude가 같다는 증거가 아니다.

## cosine

다섯 모델의 `selected_layers_and_directions.json`에 fold별 same-site cosine이 이미 저장되어 있었다. NPZ의 단위 방향 내적으로 각각 재검증한다. DeepSeek는 새 forward pass 없이 해결된다. Llama/Gemma의 알려진 약 `-0.022` / `-0.078`도 같은 core 정의에서 확인된다. Qwen의 약 `.294`라는 과거 목표치는 선택된 same-site 1D 방향의 값으로는 확인되지 않았으므로 출처가 규명될 때까지 사용하지 않는다.

## 안전한 결론

공통 1D 정의로 저장 score를 재집계한 표와 과거 Qwen/Mistral/DeepSeek protocol별 표를 분리했다. 방향의 낮은 same-site cosine은 기하학적 비동일성을 지지하지만, magnitude 유사성이나 causal gain 유사성을 뜻하지 않는다. Mistral 내부 predictive direction은 exact output readout과 별도로 유지했다.

상세 수치, fold 값, CI, source SHA-256은 나머지 CSV/JSON을 참조한다.
