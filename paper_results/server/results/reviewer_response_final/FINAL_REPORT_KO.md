# Reviewer response final

**AUDIT = PASS** — 5 models × 5 folds, completed 25/25, missing=0, failed=0, non-finite=0.

## 기존 telicity direction multi-dose

AUDIT PASS인 기존 class-balanced dual-ridge(α=.01) 결과를 재사용했다. 요청된 0.5/1/2/3 training-fold SD 범위에서 dose를 2–3 SD로 높여도 원래 telicity axis의 margin 효과와 prediction flip은 작았다. 따라서 관측된 작은 효과를 1 SD dose 부족만으로 설명할 수 없다.

## adaptive rule에 완전히 matched된 dose-only sweep

|Model|Adaptive ΔC pp|0.5 SD|1 SD|2 SD|3 SD|3 SD ΔD pp|
|---|---:|---:|---:|---:|---:|---:|
|qwen|+10.0|+7.0|+10.0|+37.0|+47.0|-7.0|
|mistral|+56.0|+0.0|+2.0|+5.0|+8.0|-1.0|
|deepseek|+41.0|+0.0|+15.0|+41.0|+41.0|-1.0|
|llama|+46.0|+17.0|+46.0|+69.0|+74.0|-5.0|
|gemma|+8.0|+4.0|+8.0|+27.0|+53.0|-2.0|

여기서는 detector·threshold·gate·site·component·direction·sign·ordering·eligible examples를 고정하고 dose만 바꿨다. 따라서 이전 standardized common-dose 결과보다 adaptive-vs-fixed 비교에 직접 대응한다.

## Group C 95% bootstrap CI

|Model|Condition|ΔC pp [95% CI]|
|---|---|---:|
|qwen|adaptive_original|+10.0 [+5.0, +16.0]|
|qwen|fixed_0.5sd|+7.0 [+2.0, +12.0]|
|qwen|fixed_1sd|+10.0 [+5.0, +16.0]|
|qwen|fixed_2sd|+37.0 [+29.0, +46.0]|
|qwen|fixed_3sd|+47.0 [+38.0, +56.0]|
|mistral|adaptive_original|+56.0 [+47.0, +65.0]|
|mistral|fixed_0.5sd|+0.0 [+0.0, +0.0]|
|mistral|fixed_1sd|+2.0 [+0.0, +5.0]|
|mistral|fixed_2sd|+5.0 [+1.0, +10.0]|
|mistral|fixed_3sd|+8.0 [+3.0, +13.0]|
|deepseek|adaptive_original|+41.0 [+32.0, +50.0]|
|deepseek|fixed_0.5sd|+0.0 [+0.0, +0.0]|
|deepseek|fixed_1sd|+15.0 [+8.0, +22.0]|
|deepseek|fixed_2sd|+41.0 [+32.0, +50.0]|
|deepseek|fixed_3sd|+41.0 [+32.0, +50.0]|
|llama|adaptive_original|+46.0 [+37.0, +55.0]|
|llama|fixed_0.5sd|+17.0 [+10.0, +24.0]|
|llama|fixed_1sd|+46.0 [+37.0, +55.0]|
|llama|fixed_2sd|+69.0 [+61.0, +77.0]|
|llama|fixed_3sd|+74.0 [+66.0, +82.0]|
|gemma|adaptive_original|+8.0 [+3.0, +14.0]|
|gemma|fixed_0.5sd|+4.0 [+1.0, +8.0]|
|gemma|fixed_1sd|+8.0 [+3.0, +14.0]|
|gemma|fixed_2sd|+27.0 [+19.0, +36.0]|
|gemma|fixed_3sd|+53.0 [+43.0, +62.0]|

CI는 10,000회, seed 42의 paired fold-stratified per-example bootstrap이며 refitting/rule-selection uncertainty는 포함하지 않는다.

## Label transition

주요 교정은 Group C의 True→Unknown 전환이다. 조건·그룹별 True/False/Unknown 분포와 모든 전환은 `label_distributions.csv`와 `label_transitions.csv`에 저장했다. False로 이동한 전환은 별도 열로 감사했다.

## 실제 search space

`correction_search_space.md` 참조. Qwen의 HELD_OUT_POST_SELECTION과 DeepSeek의 post-hoc safety-buffer/exploratory provenance를 유지하며 숨기지 않았다.

## selected-site와 matched-site gain

`selected_vs_matched_gain_provenance.csv`에서 Table 5 selected-site gain과 Table 6 telicity-matched-site gain을 분리했다. Caption 권고: *Table 5 reports each model’s selected-site decision gain, whereas Table 6 recomputes decision gain at the telicity-matched site under each direction’s own ±1 training-SD dose; the two estimands need not coincide.*

## 논문 수정

- ‘1 SD가 너무 작아서 telicity 효과가 약했다’는 설명을 삭제하고, 0.5–3 SD에서도 원래 telicity axis의 causal leverage가 작았다고 기술한다.
- adaptive 우월성을 주장할 때는 이번 동일-rule dose-only 표를 사용하고, 모델별 이질성과 preservation trade-off를 함께 보고한다.
- Table 5/6 gain 명칭과 site 차이를 caption/footnote에 명시한다.
- Qwen selection 및 DeepSeek safety-buffer 한계를 Methods/Limitations에 명시한다.

## Unresolved

Mistral adaptive 원 dose는 raw-L2 62/64이고 fixed 조건은 training-projection SD이므로 adaptive와 fixed 간 물리 dose의 직접 등치는 불가하다. Qwen의 held-out post-selection과 DeepSeek의 exploratory safety-buffer는 이번 계산으로 해소되지 않는다. CI는 selection/refit uncertainty를 포함하지 않는다.
