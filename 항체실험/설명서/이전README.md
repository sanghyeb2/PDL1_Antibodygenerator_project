> 이전 실험의 기록입니다. 현재 실행 파일은 MAGE.py / LoRA.py이며, 명령은 ../시작안내.md를 확인하세요.

# MAGE 실행 환경

공개 MAGE checkpoint를 현재 RTX 5060 Laptop에서 실행하는 별도 작업 폴더다.
공식 MAGE 저장소와 내려받은 가중치, ProGen2 코드의 내용은 수정하지 않는다.

## 환경

- Python: 공식 MAGE 폴더의 `.venv-mage-gpu/Scripts/python.exe` (3.11.6).
- PyTorch 2.7.1+cu128, CUDA runtime 12.8, NumPy 1.26.4.
- Transformers 4.32.1, tokenizers 0.13.3, accelerate 0.23.0.
- huggingface-hub 0.23.5, safetensors 0.4.5.
- 전체 실제 설치 버전: `requirements-lock.txt`.

논문의 PyTorch 2.1.0 환경을 재현한 것은 아니다. Blackwell GPU 호환을 위해 PyTorch를 변경했다.
의존성 검사는 통과했지만 이것만으로 모든 학습 기능의 호환성을 보장하지 않는다.
LoRA 라이브러리 설치·모듈 주입·학습은 아직 하지 않았다.

## 코드와 가중치

- `외부코드/progen2`: 공식 ProGen2 저장소, commit `9b4d4fb5ec19c9e55c4bb06305c0e613e46c1cf5`.
- `../MAGE_ab_generation/model_epoch4`: 다운로드한 MAGE 모델/토크나이저.
- `check_mage_runtime.py`: 필요한 원본 모듈을 경로로 연결하는 로컬 검사 스크립트.

원본 generate_antibodies.py의 디렉터리 가정과 불필요한 import를 수정하는 대신
별도 검사 스크립트에서 `progen.modeling_progen`을 직접 불러온다.
실제 모델 클래스, config, checkpoint는 공개 자료를 사용한다.

## 검사 실행: PowerShell

```powershell
$magePython = "C:\5학년 2학기\생성형AI\프로젝트\MAGE_ab_generation\.venv-mage-gpu\Scripts\python.exe"
& $magePython -m pip check
& $magePython -X utf8 "C:\5학년 2학기\생성형AI\프로젝트\항체실험\check_mage_runtime.py"
```

이 명령은 다운로드 없이 로컬 checkpoint만 읽는다. HF 캐시는 이 폴더 아래로 지정한다.
CUDA에서 FP16으로 로드하고, 공식 RBD 예제를 입력해 logits가 유한한지 검사한 뒤
최대 16토큰을 생성한다. checkpoint의 missing/unexpected/mismatched key가 있으면 실패한다.
결과·GPU 메모리·소요 시간은 `runtime_check.json`에 저장한다.
짧은 생성으로 기능 연결만 확인하며, 완전한 VH/VL 생성이나 실제 결합 검증은 아니다.
프로그램 종료 후 모델 프로세스의 GPU 메모리는 해제된다.

첫 검사에서는 764,803,616개 파라미터를 로드했고, 최대 할당 약 1.68 GiB였다.
이는 해당 짧은 추론 조건의 값이며, 전체 생성/긴 항원/학습에 필요한 메모리와 다르다.
Transformers의 PyTorch 내부 함수 관련 deprecation 경고가 있지만 검사는 통과했다.

## 다음 단계

2026-11-07 업데이트: PD-L1 106쌍을 그룹 단위로 85/21 분할하고 LoRA 한 배치
역전파 검사를 실행했다. 설정·결과·팀 공유용 재현 명령은 `LORA_REPRODUCE.md`를 참고한다.
위의 "LoRA 미실시" 문구와 아래 목록은 최초 실행 환경 구축 당시의 기록이다.

2026-11-08 본학습 완료: 7 epoch에서 조기 종료, best는 3 epoch이다.
검증 NLL은 0.8034에서 0.6587로 감소했다. 결과와 사용법은 `LORA_TRAINING.md`,
어댑터는 `학습결과/본학습_20261108_시드42/최적어댑터/`에 있다.

2026-11-09 시각화 추가: `visualize_lora.py`는 저장된 지표만 읽어 학습 곡선과
검증 21쌍의 baseline/LoRA 전후 비교 그림을 만든다. 실행법과 논문 참고는 `VISUALIZATION.md`.

1. 공식 예제에서 VH/VL 후보 생성과 분리 확인: 2026-10-29 완료. 아래 참고.
2. PD-L1 항원으로 소량 생성하고 기준 결과 보존.
3. LoRA 대상 선형층과 데이터 분할 검토 후 작은 학습 배치로 메모리 측정.

`requirements-lock.txt`는 현재 Windows 환경 기록이다. CUDA torch wheel은
`https://download.pytorch.org/whl/cu128`에서 별도로 선택해야 할 수 있다.

## 한 쌍 생성 검사

`generate_one_antibody.py`는 공식 RBD prompt를 읽어 최대 256개 새 token을 생성한다.
seed=42, top_p=0.9, temperature=1.0, FP16이며 학습하지 않는다.
이 256-token 예산은 자체 실행 검사용 설정으로, 논문의 길이 설정을 정확히 재현한다는 뜻은 아니다.
실행마다 `생성결과/rbd_날짜_시간/`에 원시 token/문자열과 추출 결과를 JSON으로 보존한다.
최초 pad/eos 전의 한 개 [LC]를 기준으로 VH/VL을 나누고 표준 아미노산·각 사슬 >=100 aa를 검사한다.
통과한 경우 candidate.fasta를 함께 저장한다. ANARCI 번호 부여·결합 검증은 하지 않는다.

```powershell
$magePython = "C:\5학년 2학기\생성형AI\프로젝트\MAGE_ab_generation\.venv-mage-gpu\Scripts\python.exe"
& $magePython -X utf8 "C:\5학년 2학기\생성형AI\프로젝트\항체실험\generate_one_antibody.py"
```

첫 실행은 VH 120 aa, VL 107 aa를 추출했고 생성 약 13초, peak allocated 1.826 GiB였다.
실제 generate loop는 EOS 없이 256-token 한도에서 종료했다. 첫 pad는 완성된 경쇄 뒤에 있었고
이후 추가 문자가 나왔다. 즉 `termination_found`는 파싱 경계 발견이며 생성 루프의 EOS 종료를 뜻하지 않는다.
첫 pad 뒤를 버리는 논문의 출력 분리 방식을 사용하고 원시 출력은 모두 보존한다.

파서 회귀검사: `python -m unittest discover -s 프로젝트/항체실험 -p test_antibody_output.py`
(프로젝트 루트 생성형AI 폴더에서 전용 환경 Python으로 실행).
