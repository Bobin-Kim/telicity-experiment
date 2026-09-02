# DeepSeek Telicity Erasure, Re-emergence, and Decision Analysis

## 결론

분석 대상은 로컬 `deepseek-llm-7b-chat`(30개 dense decoder block)이며, 과학적 단위는 5개의 verb-disjoint held-out fold이다. Telicity는 ambiguous 조건인 Group C(telic)와 D(atelic)의 비교로 측정해 truth-condition 차이를 통제했다.

1. **Telicity information은 존재한다.** Residual에서 L5부터 사전 기준을 연속 충족했고, L19에서 가장 강했다(평균 balanced accuracy 0.970, fold 범위 0.950–1.000). Attention output은 L6부터 안정적이고 L17에서 0.955, MLP output은 L7부터 안정적이고 L17에서 0.965였다.
2. **Telicity erase 후 re-emergence가 발생한다.** 안정 인코딩 구간 L5–L27의 115/115 유효 fold에서 다음 layer의 BA가 모두 증가했다. 평균 BA는 erase 직후 0.669에서 L+1의 0.875로 회복했다(평균 +0.206). 사전 판정 기준(평균 ΔBA≥0.10, 양의 fold≥4)을 23개 layer 중 22개가 L+1에서 충족했고, L22만 L+2에서 충족했다.
3. **즉각적인 re-emergence는 Attention이 주도한다.** L5–L27에서 attention의 평균 기여는 +0.205 BA, MLP는 +0.0015였다. Attention은 115/115 fold에서 양수였고 112/115 fold에서 MLP보다 컸다.
4. **Layer별 mechanism은 거의 전 구간에서 동일한 attention-mediated reconstruction이다.** 예외적으로 L4와 L22는 L+1의 순 회복량이 0.10 문턱에 조금 못 미쳐 L+2에서 판정됐지만, 두 경우 모두 첫 다음 block의 attention이 이미 정보를 증가시켰다. 특히 L22에서는 attention +0.130을 MLP −0.035가 일부 상쇄해 순 L+1 회복이 +0.095가 됐다.
5. **Decision-related information은 L13–L17에서 명시적으로 형성되고 후반부에서 매우 강해진다.** `m = z_True − z_Unknown`을 예측하는 verb-disjoint OOF ridge에서 residual은 L14부터 안정적이었다(R²: L14 0.165, L15 0.472, L16 0.798, L17 0.936). 최댓값은 L30의 R² 0.994, Pearson r 0.997이었다. Attention은 L13부터 안정적이고 L25에서 R² 0.979, MLP는 L14부터 안정적이고 L26에서 R² 0.990으로 최고였다.

## Erasure 및 회복 결과

각 outer fold의 training split 안에서 다시 4-fold verb-disjoint calibration을 수행해 목표 post-erasure BA=0.70에 가장 가까운 INLP 차원을 선택했다. 선택된 subspace만 answer-token residual에 실제 forward hook으로 제거하고 L, L+1, L+2, L+3을 새 probe로 평가했다.

- L1–L2: inner calibration이 모든 fold에서 `k=0`을 선택했으므로 실제 erase가 없었고 re-emergence를 판정할 수 없다.
- L3: 4/5 fold에서 유효 erase; 유효 fold 평균 L+1 회복 +0.238, attention +0.250, MLP −0.013.
- L4: 5/5 유효; L+1 +0.090, L+2 +0.170. 첫 다음 block은 attention +0.080, MLP +0.010이었다.
- L5–L21: 모든 layer와 fold에서 유효했고 전부 L+1에서 re-emergence 기준을 충족했다. Layer별 attention 기여는 +0.150~+0.255, MLP 기여는 −0.020~+0.030이었다.
- L22: L+1 +0.095, L+2 +0.145. Attention +0.130 뒤 MLP −0.035의 부분 상쇄가 지연 판정의 원인이다.
- L23–L27: 모두 L+1에서 충족. Attention +0.175~+0.300, MLP −0.010~+0.025였다.

유효 erase가 있었던 L3–L27 전체 124 fold-pair에서 baseline BA 0.920은 erase 후 0.666으로 감소했고(평균 −0.254), L+1 0.868, L+2 0.891, L+3 0.903으로 회복했다. L+1 증가는 124/124 pair에서 양수였다.

이 패턴의 가장 직접적인 해석은 다음 block의 self-attention이 erase되지 않은 다른 token 위치의 문맥·사건구조 정보를 다시 answer-token stream으로 가져온다는 것이다. 따라서 “attention이 telicity를 새로 만든다”기보다 **분산되어 남아 있던 telicity evidence를 attention이 재수집·재구성한다**고 해석하는 편이 정확하다. 이번 intervention은 answer-token의 선형 INLP subspace만 제거했으므로, 모든 위치와 모든 비선형 telicity 정보를 지웠다는 뜻은 아니다.

## Decision information의 위치

초반 L1–L12에서는 output decision margin의 held-out 예측력이 낮거나 OOF R²가 음수였다. 전환은 L13–L17에서 나타났다.

| 구간 | 관찰 |
|---|---|
| L1–L12 | decision margin이 안정적으로 선형 복원되지 않음 |
| L13 | attention branch에서 안정적 onset 시작(R² 0.148, r 0.571) |
| L14–L17 | residual/MLP까지 빠르게 명시화; residual R² 0.165→0.936 |
| L18–L24 | residual R² 0.947→0.988, 두 branch 모두 높은 예측력 |
| L25–L30 | 거의 완성된 decision state; attention peak L25, MLP peak L26, residual peak L30 |

따라서 decision은 단일 module 한 곳에만 저장된다기보다, 중후반부터 attention과 MLP 양쪽에 읽을 수 있는 형태로 존재하며 residual stream에 누적되어 최종 layer에서 가장 직접적으로 나타난다. 이 부분은 **decodability/localization 결과**이며, decision axis를 직접 조작한 causal steering 결과는 아니다.

## Audit 및 해석 범위

최종 audit은 PASS였다: removal 135행, recovery 540행, component 135행, 27개 complete marker, 135개 intervention cache가 모두 존재했다. 모든 cache shape은 recovery `(200, 4, 4096)`, component `(200, 3, 4096)`였고 non-finite 값은 없었다. 최대 projection residual absolute coefficient는 0.0415였다.

이 결과는 현재 adapter가 DeepSeek의 Llama-like dense architecture에서도 residual/attention/MLP hook과 실제 intervention을 올바르게 수행함을 보여준다. 다만 DeepSeek-V2/V3 같은 MoE·특수 attention 구조나 parallel block 모델까지 자동으로 보장하지는 않는다. 그런 모델에는 module discovery, expert routing, branch 합성 방식에 맞춘 adapter와 별도 pilot audit이 필요하다.

## 산출물

- `results/deepseek-7b-telicity-rq/telicity_probe_summary.csv`
- `results/deepseek-7b-telicity-rq/decision_layerwise_oof.csv`
- `results/deepseek-7b-telicity-rq/removal_sweep/layerwise_mechanism.csv`
- `results/deepseek-7b-telicity-rq/removal_sweep/final_audit.json`
- `results/deepseek-7b-telicity-rq/removal_sweep/summary.json`
