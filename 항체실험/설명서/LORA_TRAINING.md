> 이전 실험의 기록입니다. 현재 실행 파일은 MAGE.py / LoRA.py이며, 명령은 ../시작안내.md를 확인하세요.

# PD-L1 MAGE 본학습 실험

실험 ID: `본학습_20261108_시드42`. 기존 106쌍 중 학습 85쌍 / 검증 21쌍, 90% either-chain 유사도 그룹 분할을 유지한다. 분할 시드와 학습 시드는 각각 42다.

원본 MAGE에서 **새 LoRA를 초기화**해 학습한다. 이전 한 배치 검사 어댑터를 이어 학습하지 않는다. 검증셋은 checkpoint 선택에 쓰므로 독립 최종 test set이 아니다.

## 완료 결과

7 epoch / optimizer 77 step에서 조기 종료했다. 선택한 checkpoint는 **3 epoch**다.
학습 소요 시간은 약 4분 50초였고, PyTorch 최대 GPU 할당 메모리는 3.4282 GiB였다.

- 원본 validation NLL: **0.8034229584** → best: **0.6587138322**.
- 원본 validation perplexity: **2.2331719154** → best: **1.9323054664**.
- best adapter를 실제로 다시 불러와 validation NLL이 저장 당시와 일치함을 확인했다.
- 원본 MAGE 파라미터 고정 조건은 유지됐다.
- 4–7 epoch에서는 validation NLL이 다시 증가했다. 최종 epoch의 어댑터 대신 `최적어댑터/`를 사용한다.
- 13개 로컬 단위 검사를 통과했고, 실행 코드 독립 검토에서 학습 수식·분할·저장 관련 결함은 발견되지 않았다.
- 한 번의 학습 실험만 수행했다. 이 결과는 한 seed와 한 validation split에 대한 탐색 결과다.

## 고정한 설정

- 대상: 27 block의 qkv_proj/out_proj, 총 54개 층.
- rank 8 / alpha 16 / dropout 0.05, 학습 파라미터 1,990,656개.
- 고정 base BF16, adapter FP32, BF16 autocast, 양자화 없음.
- microbatch 1, gradient accumulation 8. 85개 기준 epoch마다 optimizer update 11회이며 마지막 update는 5개 예제를 사용한다.
- AdamW lr 1e-4, weight decay 0.01, gradient clipping 1.0; constant learning rate, warmup 없음.
- 각 누적 window의 실제 항체 토큰 수로 loss를 정규화한다. 마지막 5개도 8로 나누지 않는다.
- 항원/[SEP]/padding은 제외하고 VH+[LC]+VL+EOS의 causal CE를 계산한다. 기존 원본 학습 스크립트의 loss 구현과 동일하다고 가정하지 않는다.
- 최대 20 epoch, epoch마다 검증. 유의한 검증 NLL 개선 기준 0.001 nats/token, patience 4. 최저 검증 NLL 어댑터를 저장하며 작은 개선도 best 갱신에 반영한다.
- 학습 전 원본 모델의 validation NLL/perplexity를 baseline으로 평가한다. 학습 이후 best는 **학습한 epoch들 중** 선택하며 baseline보다 나빴는지도 별도로 보고한다.
- Python/NumPy/Torch/CUDA 시드, Python hash seed, deterministic algorithms, cuBLAS workspace, TF32 비활성화를 기록한다. epoch별 학습 ID 순서도 저장한다. 재현 확인을 위한 동일 실험 반복은 하지 않는다.

## 실행 명령

프로젝트 루트에서 기존 가상환경을 사용한다. 아래 `--output`은 새 폴더여야 한다. 팀원이 다시 실행할 때는 출력 폴더 이름을 바꾼다.

```powershell
$magePython = Join-Path (Get-Location) 'MAGE_ab_generation/.venv-mage-gpu/Scripts/python.exe'
& $magePython -X utf8 -u 항체실험/lora_train.py --split 항체실험/학습결과/데이터분할_시드42 --output 항체실험/학습결과/본학습_20261108_시드42 --train-seed 42 --epochs 20
```

다른 컴퓨터에서 원본 모델·ProGen 소스·CUDA 환경을 준비하는 방법은 `LORA_REPRODUCE.md`를 참고한다. 실제 본학습 환경은 이번 run의 `requirements-lock.txt`, `environment.json`이 기준이다.

## 결과 파일

`항체실험/학습결과/본학습_20261108_시드42/`:

- `최적어댑터/`: 학습 epoch 중 validation NLL이 가장 낮은 어댑터. 생성이나 후속 평가에 사용할 파일이다.
- `마지막어댑터/`: 종료된 마지막 epoch의 어댑터. best와 같지 않을 수 있다.
- `토크나이저/`: 원본 모델 토크나이저 사본.
- `result.json`: 학습 완료 여부, baseline/best 지표, best epoch, 메모리, 실행 시간.
- `baseline_validation.json`, `best_validation_per_sequence.json`: 토큰 가중 validation NLL/perplexity와 각 항체쌍의 NLL.
- `history.csv`, `history.json`: epoch별 학습 NLL, 검증 NLL/perplexity, update 수, early stopping 상태.
- `settings.json`, `epoch_orders.json`: 인자·시드·학습 순서와 정확한 실행 명령.
- `split_manifest.json`, `input_hashes.json`: 원본 데이터·분할·가중치·토크나이저·소스코드 해시.
- `environment.json`, `requirements-lock.txt`: 하드웨어와 라이브러리 버전.
- `last_training_state.pt`: 마지막 epoch의 optimizer/RNG/early stopping 상태. 자동 resume CLI는 제공하지 않는다. 향후 재개 기능을 구현할 경우 해당 epoch의 `마지막어댑터/`와 함께 사용해야 하며 best adapter와 섞으면 안 된다. 이 파일은 직접 만든 신뢰 가능한 파일에 한해 읽는다.
- `progress.json`: 실행 중 진행 상태.

best adapter는 전체 MAGE가 아니다. 원본 base 모델을 동일한 config/가중치로 로드해 연결한다. PEFT config에 저장된 과거 절대 경로를 다른 컴퓨터에서 그대로 사용하지 않는다.

```python
# 프로젝트 루트에서 실행하는 예시. 학습은 하지 않는다.
import sys
from pathlib import Path
import torch
from peft import PeftModel

root = Path.cwd()
sys.path.insert(0, str(root / '항체실험' / '외부코드' / 'progen2'))
from progen.modeling_progen import ProGenForCausalLM

base = ProGenForCausalLM.from_pretrained(
    str(root / 'MAGE_ab_generation' / 'model_epoch4'),
    torch_dtype=torch.bfloat16, local_files_only=True)
model = PeftModel.from_pretrained(
    base, str(root / '항체실험' / '학습결과' /
              '본학습_20261108_시드42' / '최적어댑터'))
model.eval().to('cuda')
```

validation NLL의 감소는 보류한 서열에 대한 조건부 예측이 좋아졌다는 뜻이다. PD-L1 결합력, 항체의 새로움, 발현성·개발 가능성을 입증하지 않는다. source/patent 단위 독립성 및 MAGE 기존 학습 데이터와의 중복은 이번 분할에서 보장하지 않는다.
