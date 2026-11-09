> 이전 실험의 기록입니다. 현재 실행 파일은 MAGE.py / LoRA.py이며, 명령은 ../시작안내.md를 확인하세요.

# PD-L1 MAGE LoRA 준비 및 재현 안내

2026-11-07, RTX 5060 Laptop 8GB에서 실행했다. 이 작업은 **데이터 분할 + LoRA 연결 + 한 배치 역전파/optimizer 업데이트 검사**까지다. 전체 epoch 학습과 검증셋 성능 측정은 아직 하지 않았다.

## 기록된 설정

- 분할 시드 `42`, 학습 시드 `42`. 두 시드는 별도 인자다.
- 원본 106쌍 → 27개 연결 그룹 → 학습 85쌍 / 검증 21쌍.
- 중복을 제외한 원본 사슬 수: VH 88개 / VL 79개. VH/VL 쌍은 106개 모두 다르다.
- 유사도는 `1 - Levenshtein_distance / max(len(chain_a), len(chain_b))`다. 정렬 기반 percent identity와는 다르다.
- VH 또는 VL 중 하나라도 유사도 **90% 이상**이면 연결한다. A–B, B–C가 연결되면 A–C가 직접 유사하지 않아도 세 쌍 전체를 같은 그룹에 둔다.
- ID 정렬 후 그룹 단위로 나눈다. 검증 목표 크기는 21쌍이며, 그룹 크기 부분합으로 가장 가까운 크기를 선택한다. 동일 크기 후보 간 선택 순서는 시드로 고정한다. CSV 행 순서를 바꿔도 동일 분할이다.
- 학습/검증 간 1,785개 쌍을 전수 검사했다. 기준 위반 0건, 최대 VH 유사도 85.8333%, 최대 VL 유사도 89.8148%다.
- 원본 CSV SHA-256: `2ccd119cac70372e59ecf9710e9be0ed4b91f187cde2322b83a570079b60abe4`.

## LoRA 한 배치 검사

- 원본 MAGE 764,803,616개 파라미터는 고정했다.
- 27개 Transformer block의 `attn.qkv_proj`, `attn.out_proj`, 총 54개 선형층에 연결했다. qkv_proj는 Q/K/V가 합쳐진 층이다.
- `r=8`, `alpha=16`, `dropout=0.05`; 학습 파라미터 1,990,656개.
- BF16 base / FP32 adapter / BF16 autocast. 양자화는 사용하지 않았다.
- batch size 1, AdamW `lr=1e-4`, `weight_decay=0.01`, gradient clipping norm 1.0, **한 번의 optimizer step**. 이 값은 실행 검사 설정이며 최적 hyperparameter로 선정한 값이 아니다.
- 학습 데이터에서 토큰 길이가 가장 긴 예제 454토큰을 선택했다. 길이 동률은 ID 사전순으로 결정한다. 입력을 자르거나 1024토큰으로 불필요하게 padding하지 않았다.
- 항원과 `[SEP]`, padding은 loss에서 제외하고, 첫 VH 아미노산부터 `[LC]`, VL, 명시적으로 붙인 EOS까지 233토큰을 학습한다. 원본 MAGE 전체 학습 스크립트의 단순 실행 복제는 아니다.
- KV cache는 끄고 gradient checkpointing은 사용하지 않았다. 이 ProGen 구현에는 표준 gradient checkpointing 지원이 없다.
- 첫 실행: loss `1.1141371726989746`, gradient norm `0.9017638564109802`.
- trainable tensor 108개 모두 gradient가 존재하고 유한했다. 이 중 54개 tensor의 gradient가 0이 아니었다. LoRA 초기 B=0이므로 첫 역전파에서 A의 gradient가 0인 것은 정상이다. AdamW의 weight decay도 있으므로 실제 변경된 tensor 수와 nonzero-gradient tensor 수는 다를 수 있다.
- frozen base에는 gradient가 없고 parameter version이 유지됐다. 어댑터는 실제로 변경됐다.
- 어댑터 저장 후 재로딩한 전체 logits의 최대 절대차 `0.0`.
- 첫 실행 PyTorch 최대 할당 **3.4142 GiB**, 최대 예약 **3.4883 GiB**. 이는 한 배치의 텐서 메모리이며 `nvidia-smi`의 프로세스 전체 메모리와는 다르다.
- 저장된 `한단계어댑터_미완성`는 실행 검사용이다. 완성된 PD-L1 생성 모델로 취급하지 않는다. 본학습은 원본 MAGE에서 새 어댑터를 초기화해 시작하는 것이 기본이다.

## 기존 노트북 환경에서 다시 실행

프로젝트 루트에서 PowerShell로 실행한다. 결과 폴더가 이미 있으면 덮어쓰지 않고 종료하므로 새 이름을 사용한다.

```powershell
$magePython = Join-Path (Get-Location) 'MAGE_ab_generation/.venv-mage-gpu/Scripts/python.exe'
& $magePython -X utf8 항체실험/lora_data.py --output 항체실험/학습결과/team_split_seed42 --split-seed 42 --threshold 0.90 --val-fraction 0.20
& $magePython -X utf8 -u 항체실험/lora_smoke.py --split 항체실험/학습결과/team_split_seed42 --output 항체실험/학습결과/team_smoke_seed42 --train-seed 42
& $magePython -X utf8 -m unittest discover -s 항체실험 -p 'test_*.py' -v
& $magePython -m pip check
```

분할만 재현하려면 Python 3.11과 `rapidfuzz==3.10.1`만 있으면 된다. `train.csv`, `validation.csv`, `assignments.csv`, `split_manifest.json`이 나온다. 분할 manifest에는 모든 연결 edge와 행별 ID/그룹/분할 결과를 넣었다.

## 조원 컴퓨터에서 실행할 때

공유 ZIP에는 데이터, 실행 코드, 테스트, 분할 결과, 환경 기록과 한 단계 어댑터를 포함한다. 3GB 원본 모델 가중치와 Python 가상환경은 포함하지 않는다. 기존 프로젝트에 ZIP을 풀거나 아래 구조로 원본 모델과 ProGen 소스를 준비한다.

```text
프로젝트/
  datasets/PDL1_최종_튜닝용_실측양성_106쌍.csv
  MAGE_ab_generation/model_epoch4/    # 원본 MAGE 가중치, config, 토크나이저 파일
  항체실험/
    외부코드/progen2/               # 원본 ProGen 저장소
    lora_data.py
    lora_smoke.py
    check_mage_runtime.py
    학습결과/...
```

- 원본 모델: https://huggingface.co/perrywasdin/MAGE_V1 . 가중치는 실행 기록의 `input_hashes.json`과 SHA-256을 비교한다. 같은 이름의 모델이라도 hash가 다르면 동일 실험이라고 간주하지 않는다.
- ProGen 소스는 `git clone https://github.com/enijkamp/progen2 항체실험/외부코드/progen2` 후 `git -C 항체실험/외부코드/progen2 checkout 9b4d4fb5ec19c9e55c4bb06305c0e613e46c1cf5`로 고정한다.
- 실행 환경은 `학습결과/한배치검사_시드42/requirements-lock.txt`에 기록했다. 이 노트북에서 추가한 패키지는 `peft==0.5.0`, `rapidfuzz==3.10.1` 두 개이며 기존 torch/transformers 버전은 변경하지 않았다.
- 새 환경은 Python 3.11에서 `python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128`로 CUDA wheel을 준비하고, `python -m pip install -r 항체실험/학습결과/한배치검사_시드42/requirements-lock.txt`를 실행한다. GPU/OS에 맞는 wheel 지원 여부는 별도로 확인한다. 이 실행기는 CUDA BF16 지원을 요구한다.
- PEFT generic wrapper로 연결했으므로, 다른 컴퓨터에서 어댑터를 쓸 때 config의 저장 당시 절대 경로를 자동으로 사용하지 말고 로컬 `ProGenForCausalLM` base를 직접 로드한 뒤 `PeftModel.from_pretrained(base, adapter_path)`로 연결한다. base dtype도 BF16으로 맞춘다.

## 재현성 기록과 해석

- split manifest와 assignments가 실제 분할의 기준 자료다. 본학습 때 매번 새로 랜덤 분할하지 말고 고정된 파일을 사용한다.
- 실행기가 원본 CSV 해시, train.csv 내용, 분할 재계산 결과를 확인한 후 GPU 학습을 시작한다.
- `input_hashes.json`: 원본 CSV, split manifest, train.csv, MAGE 가중치/config/토크나이저, ProGen 모델 코드, 실행 스크립트 해시.
- `environment.json`: Python·패키지·CUDA·cuDNN·GPU·driver 정보와 ProGen commit.
- `settings.json`: 분할/학습 시드, LoRA/optimizer 설정, 예제 ID, loss 범위.
- `result.json`: gradient, 업데이트, 재로딩, loss, 메모리 결과.
- 라이브러리 RNG와 Python hash seed를 고정하고, deterministic algorithms와 cuBLAS 고정 workspace를 사용하며 TF32를 껐다. 같은 노트북에서의 반복 일치와 다른 GPU/버전에서의 bit 단위 일치는 구분해야 한다. 다른 GPU에서 수치가 완전히 같다고 보장하지 않는다.
- 90% 전장 사슬 유사도 기반 분할은 CDR3/계통 기반 분할과 다르다. 같은 특허/출처 그룹을 반드시 분리한 것도 아니다. MAGE의 기존 학습 데이터와 중복되는지는 이번에 검사하지 않았다. 검증셋은 hyperparameter 선택에 쓰는 용도이며, 독립 최종 test set은 아직 없다. 결합력 개선은 이 실행 검사로 판단할 수 없다.

구현 참고: [PEFT custom models](https://huggingface.co/docs/peft/en/developer_guides/custom_models), [PyTorch deterministic algorithms](https://docs.pytorch.org/docs/main/generated/torch.use_deterministic_algorithms.html).
