# Telicity Cross-Context Validation

**AUDIT: PASS**

All directions, signs, thresholds, and SDs use outer-training verbs only. Decoding is evaluated on held-out verbs. The causal A/B direction is applied to held-out Group C at the pre-existing fold-specific RQ2 site; the audited C/D causal reference is reused.

## Selected-site decoding

| Model | Type | Within C/D BA | A/B→C/D BA | Within A/B BA | C/D→A/B BA | signed cosine |
|---|---|---:|---:|---:|---:|---:|
| qwen | probe | 0.970 | 0.705 | 0.920 | 0.715 | 0.184 |
| qwen | mean_difference | 0.730 | 0.720 | 0.730 | 0.770 | 0.713 |
| mistral | probe | 0.955 | 0.540 | 0.955 | 0.830 | 0.326 |
| mistral | mean_difference | 0.765 | 0.525 | 0.735 | 0.500 | 0.456 |
| deepseek | probe | 0.935 | 0.505 | 0.925 | 0.650 | 0.212 |
| deepseek | mean_difference | 0.875 | 0.500 | 0.710 | 0.575 | 0.156 |
| llama | probe | 0.940 | 0.570 | 0.975 | 0.870 | 0.381 |
| llama | mean_difference | 0.790 | 0.555 | 0.680 | 0.500 | 0.227 |
| gemma | probe | 0.930 | 0.520 | 0.920 | 0.750 | 0.372 |
| gemma | mean_difference | 0.780 | 0.570 | 0.815 | 0.670 | 0.265 |

## Group C causal transfer

| Model | Source | Dose | Causal gain | −dose Δmargin | +dose Δmargin | −dose flip | +dose flip |
|---|---|---:|---:|---:|---:|---:|---:|
| deepseek | AB | 0.5 | -0.0387 | +0.0392 | -0.0381 | 0.000 | 0.000 |
| deepseek | AB | 1 | -0.0773 | +0.0780 | -0.0766 | 0.000 | 0.000 |
| deepseek | AB | 2 | -0.1544 | +0.1558 | -0.1530 | 0.000 | 0.000 |
| deepseek | AB | 3 | -0.2311 | +0.2336 | -0.2286 | 0.000 | 0.000 |
| deepseek | CD | 0.5 | -0.0168 | +0.0164 | -0.0172 | 0.000 | 0.000 |
| deepseek | CD | 1 | -0.0334 | +0.0334 | -0.0333 | 0.000 | 0.000 |
| deepseek | CD | 2 | -0.0652 | +0.0644 | -0.0659 | 0.000 | 0.000 |
| deepseek | CD | 3 | -0.0979 | +0.0994 | -0.0964 | 0.000 | 0.000 |
| gemma | AB | 0.5 | -0.0071 | +0.0064 | -0.0079 | 0.000 | 0.000 |
| gemma | AB | 1 | -0.0151 | +0.0149 | -0.0153 | 0.000 | 0.000 |
| gemma | AB | 2 | -0.0308 | +0.0311 | -0.0305 | 0.000 | 0.000 |
| gemma | AB | 3 | -0.0460 | +0.0459 | -0.0461 | 0.000 | 0.010 |
| gemma | CD | 0.5 | -0.0056 | +0.0047 | -0.0066 | 0.000 | 0.000 |
| gemma | CD | 1 | -0.0115 | +0.0109 | -0.0121 | 0.000 | 0.000 |
| gemma | CD | 2 | -0.0229 | +0.0225 | -0.0232 | 0.000 | 0.010 |
| gemma | CD | 3 | -0.0345 | +0.0345 | -0.0346 | 0.000 | 0.010 |
| llama | AB | 0.5 | -0.0086 | +0.0087 | -0.0084 | 0.000 | 0.000 |
| llama | AB | 1 | -0.0179 | +0.0180 | -0.0178 | 0.000 | 0.000 |
| llama | AB | 2 | -0.0364 | +0.0352 | -0.0377 | 0.000 | 0.000 |
| llama | AB | 3 | -0.0535 | +0.0516 | -0.0555 | 0.000 | 0.000 |
| llama | CD | 0.5 | -0.0041 | +0.0037 | -0.0045 | 0.000 | 0.000 |
| llama | CD | 1 | -0.0070 | +0.0070 | -0.0070 | 0.000 | 0.000 |
| llama | CD | 2 | -0.0131 | +0.0123 | -0.0139 | 0.000 | 0.000 |
| llama | CD | 3 | -0.0199 | +0.0175 | -0.0223 | 0.000 | 0.010 |
| mistral | AB | 0.5 | +0.0130 | -0.0184 | +0.0075 | 0.000 | 0.000 |
| mistral | AB | 1 | +0.0362 | -0.0425 | +0.0300 | 0.000 | 0.000 |
| mistral | AB | 2 | +0.0716 | -0.0884 | +0.0547 | 0.000 | 0.000 |
| mistral | AB | 3 | +0.1102 | -0.1425 | +0.0778 | 0.000 | 0.000 |
| mistral | CD | 0.5 | -0.0281 | +0.0234 | -0.0328 | 0.000 | 0.000 |
| mistral | CD | 1 | -0.0578 | +0.0541 | -0.0616 | 0.000 | 0.000 |
| mistral | CD | 2 | -0.1214 | +0.1078 | -0.1350 | 0.000 | 0.000 |
| mistral | CD | 3 | -0.1816 | +0.1506 | -0.2125 | 0.000 | 0.000 |
| qwen | AB | 0.5 | -0.0431 | +0.0350 | -0.0512 | 0.000 | 0.000 |
| qwen | AB | 1 | -0.0775 | +0.0613 | -0.0938 | 0.000 | 0.000 |
| qwen | AB | 2 | -0.1619 | +0.1487 | -0.1750 | 0.000 | 0.000 |
| qwen | AB | 3 | -0.2606 | +0.2550 | -0.2662 | 0.000 | 0.010 |
| qwen | CD | 0.5 | -0.0156 | +0.0075 | -0.0238 | 0.000 | 0.000 |
| qwen | CD | 1 | -0.0144 | +0.0013 | -0.0275 | 0.000 | 0.000 |
| qwen | CD | 2 | -0.0300 | +0.0200 | -0.0400 | 0.000 | 0.000 |
| qwen | CD | 3 | -0.0606 | +0.0437 | -0.0775 | 0.000 | 0.000 |

## 보수적 해석

1. **A/B→C/D:** probe 방향은 source-trained threshold를 그대로 적용한 BA에서는 Qwen만 0.705로 뚜렷했고, 나머지는 0.505–0.570이었다. 그러나 threshold-free AUROC는 전 모델 0.887–0.946이었다. 따라서 telicity의 순위 정보는 문맥을 넘어 상당히 이전되지만, projection의 절대 기준점은 문맥에 따라 이동한다. 이를 “동일한 보편적 classifier가 그대로 이전된다”고 표현해서는 안 된다.
2. **C/D→A/B:** probe BA는 0.650–0.870, AUROC는 0.723–0.946으로 역방향 이전이 더 강했다. 이는 기존 C/D 방향이 C-vs-D NLI label만을 분리한다는 설명과는 맞지 않으며, telicity 관련 정보를 포함한다는 construct-validity 증거다.
3. **기하학적 정렬:** probe signed cosine은 0.184–0.381에 불과했다. mean-difference cosine도 Qwen(0.713)을 제외하면 0.156–0.456이었다. 따라서 단일한 context-invariant telicity axis보다는 context-dependent telicity directions가 더 안전한 결론이다.
4. **방향 정의 간 일치:** mean-difference 결과는 Qwen에서는 양방향 BA 0.720/0.770으로 이전됐지만, 다른 모델의 A/B→C/D BA는 0.500–0.570이었다. 두 방향 정의가 전 모델에서 같은 질적 패턴을 보였다고 할 수 없다.
5. **Group C causal transfer:** 1 SD에서 A/B 방향의 causal gain은 Qwen −0.0775 (95% CI [−0.1262, −0.0288]), Mistral +0.0362 ([+0.0085, +0.0640]), DeepSeek −0.0773 ([−0.1122, −0.0423]), Llama −0.0179 ([−0.0421, +0.0063]), Gemma −0.0151 ([−0.0186, −0.0116])이었다. 모든 모델에서 목표 projection 변화는 약 1 SD로 실현됐지만 margin 효과는 작고 prediction flip은 1 SD에서 0이었다. Mistral은 원래 C/D 방향의 효과(−0.0578)와 반대 부호이므로 동일한 causal telicity mechanism의 재현으로 해석할 수 없다.
6. **최종 결론:** 결과는 기존 C/D 방향이 순수한 NLI-label separator만은 아니라는 주장을 강화하지만, telicity가 모든 문맥에서 하나의 공통 축으로 완전히 분리됐다는 주장은 지지하지 않는다. 가장 안전한 표현은 “telicity information transfers across contexts primarily as rank/separation information, while its calibration, geometry, and decision-level use remain context- and model-dependent”이다.

Verb pairing은 A↔C 100개와 B↔D 100개로 완전한 일대일이었다. Selected-site probe의 전체 held-out verb projection correlation은 A/B→C/D에서 0.655–0.777(모델별 평균), C/D→A/B에서 0.424–0.798이었다. 다만 telic/atelic class 내부 correlation은 더 약하고 불균일하므로 이 수치는 보조 진단으로만 사용한다.
