> 이전 실험의 기록입니다. 현재 실행 파일은 MAGE.py / LoRA.py이며, 명령은 ../시작안내.md를 확인하세요.

# MAGE baseline와 LoRA의 동일 항체쌍 NLL 비교

## 이번 실험에서 실제로 측정한 것

검증셋에서 한 항체를 골라 평가한 것이 아니다. **동일한 검증 VH/VL 21쌍 전체**를 원본 MAGE와 학습한 LoRA 어댑터 양쪽에서 평가했다. 학습 중 매 epoch에서도 동일한 21쌍을 평가해 최적 epoch 3을 선택했다. 앞서 한 배치 smoke test에서만 가장 긴 학습 예제 한 개를 사용했다.

각 예제는 정답 항체 서열을 입력으로 주는 teacher-forcing 방식이다. 모델이 생성한 서로 다른 후보 두 개를 비교한 것이 아니다. 항원과 이전 정답 토큰을 조건으로 다음 정답 토큰의 log probability를 계산한다.

`NLL_i = -sum_t log p(y_it | antigen_i, y_i,<t) / T_i`

점수 구간은 `VH + [LC] + VL + EOS`다. 항원·[SEP]·padding은 점수에서 제외한다. 따라서 단위는 **nats per supervised token**이며, 서열 전체의 log-likelihood 합이나 결합 확률을 뜻하지 않는다.

서열별 평균 NLL을 얻은 뒤 전체 지표는 `sum_i(T_i * NLL_i) / sum_i(T_i)`로 계산했다. 여기서 총 토큰 수는 4,803개다. 전체 perplexity는 `exp(전체 token-weighted NLL)`이며, 서열별 perplexity의 단순 평균이 아니다.

## 관련 연구를 확인한 결과

1. **AbTune (2026), Fig. 1a,c**: 항체 LoRA 연구에서 방법별 분포를 box plot으로 보여주고, fine-tuning step에 따른 sequence perplexity 등의 변화를 곡선으로 표시했다. 평균과 표준편차 음영을 사용한 부분도 있다. 이번 코드는 분포와 시간축 표현을 참고한다. 단, AbTune은 masked LM의 test-time tuning이고, 이번 MAGE는 causal LM의 지도학습이므로 실험과 수치를 동일하게 비교하지 않는다.
   - https://pmc.ncbi.nlm.nih.gov/articles/PMC13356902/#s3
   - DOI: 10.1093/bib/bbag374
2. **ProGen2 (2023), Fig. 1c,d 및 Eq. 1**: 보류한 서열의 perplexity로 단백질 서열 분포 예측 능력을 비교했다. likelihood/perplexity 개선과 분자 기능 향상을 구분했다. 우리도 likelihood 지표와 생물학적 결합력 주장을 구분한다.
   - https://doi.org/10.1016/j.cels.2023.10.002
   - 논문: https://www.cell.com/article/S2405471223002727/pdf
3. **MAGE (2025), Methods: Antibody generation and filtering**: 생성 서열의 perplexity를 평가했고, 후보 선별에는 서열 특성과 실제 결합 검증도 함께 사용했다. 현재 그림은 이 논문의 특정 그림을 복제한 것이 아니라, 우리 데이터의 동일 서열 전후 likelihood 비교용으로 설계했다.
   - https://pmc.ncbi.nlm.nih.gov/articles/PMC12684077/
   - DOI: 10.1016/j.cell.2025.10.006

## 추가한 그림의 구성

- **A: Training dynamics** — epoch별 온라인 학습 NLL과 검증 NLL, 원본 MAGE 기준선, 선택한 epoch. epoch 0은 학습 전 검증 결과다. 학습 NLL은 dropout을 켜고 변하는 가중치로 계산한 값이므로, 같은 epoch 종료 모델을 eval mode로 측정한 학습셋 NLL과는 다르다.
- **B: Matched antibody pairs** — 두 모델의 분포를 box plot으로 표시하고, 같은 항체쌍의 점을 선으로 연결한다. 검은 마름모는 token-weighted 평균이다. 상자는 표준 IQR, 가운데 선은 median이며 confidence interval을 뜻하지 않는다. 평균만 비교해 일부 서열 악화를 숨기지 않도록 모든 점을 표시한다.
- **C: Every-pair change** — `LoRA NLL - baseline NLL`을 21쌍 모두 표시한다. 음수면 개선이다. V01 등의 alias와 원본 ID의 대응은 `paired_nll.csv`에 있다.

한 seed의 실험이며 검증 항체들 중 같은 유사도 그룹에 속한 것들이 있다. 독립 반복으로 간주한 오차막대·p-value·유의성 별표는 추가하지 않았다. 최적 epoch를 고른 동일한 검증셋의 결과이므로 독립 test 성능으로 쓰면 안 된다. NLL 감소만으로 PD-L1 결합력 개선을 판단하지 않는다.

## 직접 실행

프로젝트 루트에서 PowerShell로 실행한다. 기본 경로는 스크립트 위치를 기준으로 정하므로 학습 결과를 같은 구조에 두면 된다. **모델을 로드하거나 학습·추론을 다시 실행하지 않고 저장된 JSON/CSV만 읽는다.** CPU에서 실행 가능하다.

```powershell
$magePython = Join-Path (Get-Location) 'MAGE_ab_generation/.venv-mage-gpu/Scripts/python.exe'
& $magePython -X utf8 항체실험/visualize_lora.py
```

창으로도 보고 싶으면:

```powershell
& $magePython -X utf8 항체실험/visualize_lora.py --show
```

다른 실험 폴더를 선택하려면:

```powershell
& $magePython -X utf8 항체실험/visualize_lora.py --run-dir 항체실험/학습결과/본학습_20261108_시드42 --output-dir 항체실험/생성결과/presentation_figures
```

의존성: `matplotlib==3.9.2`. 기존 Python 3.11 가상환경에 설치했으며 torch/numpy 버전은 변경하지 않았다. 다른 환경은 `python -m pip install matplotlib==3.9.2`로 준비한다.

기본 결과 폴더는 `항체실험/학습결과/본학습_20261108_시드42/시각화/`다. 같은 출력 경로로 재실행하면 그림과 비교 CSV를 갱신한다. 원본 학습 결과는 수정하지 않는다.

- `lora_validation_comparison.png`: 300 dpi 그림, 발표 자료용.
- `lora_validation_comparison.svg`: 확대해도 선명한 벡터 그림, 편집용.
- `paired_nll.csv`: 원본 ID / 그룹 / 토큰 수 / 두 NLL / 변화량 / 두 perplexity.
- `plot_summary.json`: token-weighted 지표, 개선·악화 개수와 시각화 버전.

그림의 영문 라벨은 다른 컴퓨터의 한글 폰트 설치 여부와 무관하게 표시하기 위한 선택이다.
