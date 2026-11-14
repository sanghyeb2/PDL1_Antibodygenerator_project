# PD-L1 구조 평가 환경과 실행 기록

이번 단계는 **MAGE 20쌍 + LoRA 20쌍 + 양성 대조군 2쌍**을 비교합니다. 후보 서열은 `공유결과/PDL1_MAGE_LoRA_20261114/분석/*_최종후보.json`에서 가져옵니다. 대조군의 VH/VL 가변 영역은 RCSB 5X8L(atezolizumab), 5X8M(durvalumab)의 polymer 서열을 ANARCI/IMGT로 구분해 추출합니다. 실험 구조 자체를 생성 후보의 도킹 입력이나 정답으로 사용하지 않습니다.

## 실행 환경

- Windows 호스트 + WSL Ubuntu 24.04, Python 3.11.
- GPU: RTX 5060 Laptop 8151 MiB; 드라이버 595.95. WSL 메모리 약 15 GiB.
- ColabFold 1.6.3, AlphaFold-ColabFold 2.3.20, JAX 0.10.1 CUDA12; 정확한 의존성은 `colabfold-lock.txt`.
- HADDOCK3 2026.10.1, PRODIGY 2.3.0; 정확한 의존성은 `docking-lock.txt`.
- 환경 경로 `/root/.cache/pdl1-structure/{colabfold,docking}`. 기존 학습/서열 분석 환경과 독립적입니다.
- HADDOCK CNS Linux 바이너리 정상 구동 확인.
- 2026-11-14 사용자가 공식 MMseqs2 서버 `https://api.colabfold.com`로 후보/대조군/PD-L1 서열 전송을 명시적으로 승인했습니다. 템플릿은 사용하지 않습니다.

## 재현성과 재개

`구조평가.py` 한 파일에 입력 준비, 예측, 도킹, 접촉면 및 PRODIGY 분석 코드가 들어 있습니다. `manifest.json`에 시드 42, 입력 SHA256, 표적 범위와 설정을 저장합니다. 완료 구조는 재사용하며 전체 계산을 재현성 확인만을 위해 반복하지 않습니다.

결과 루트: `항체실험/구조결과/PDL1_40쌍_대조군_20261114_시드42/`.
HADDOCK 대용량 원본 계산은 WSL의 `/root/.cache/pdl1-structure/runs/<profile>/<candidate>/`에 저장되고, 프로젝트의 `docking/.../location.json`이 위치를 기록합니다.

WSL에서 프로젝트의 `항체실험` 디렉터리로 이동한 후 실행합니다:

```bash
# 입력 준비: 기존 ANARCI 환경 필요 (최초 1회)
PATH=/root/.cache/pdl1-analysis/env/bin:$PATH python 구조평가.py prepare

# 공통 PD-L1, 양 그룹 첫 후보, 양성 대조군부터 5개 작업
/root/.cache/pdl1-structure/colabfold/bin/python 구조평가.py fold --limit 5

# 전체 예측: 이미 완료된 항목은 건너뜀
/root/.cache/pdl1-structure/colabfold/bin/python 구조평가.py fold

# 상태와 Fv 구조 신뢰도
/root/.cache/pdl1-structure/docking/bin/python 구조평가.py status
/root/.cache/pdl1-structure/docking/bin/python 구조평가.py fold-summary

# 도킹 자원 측정 파일럿: 본실험 순위로 사용하지 않음
/root/.cache/pdl1-structure/docking/bin/python 구조평가.py dock --profile pilot --sampling 64 --refine 8 --limit 1

# 완료된 도킹의 클러스터/접촉면/PRODIGY/그림
/root/.cache/pdl1-structure/docking/bin/python 구조평가.py analyze --profile pilot
```

본 도킹의 공통 sampling/refine 수는 파일럿의 실제 시간/메모리를 확인한 뒤 기록합니다. 중단된 HADDOCK 실행 디렉터리를 자동 삭제하지 않으며, 해당 로그를 확인한 뒤 HADDOCK의 `--restart` 지점을 정해야 합니다. 동시에 같은 후보를 실행하지 마세요.

## 분석 기준과 해석

- 공통 PD-L1: UniProt Q9NZQ7 19–238, 220 aa. 각 Fv는 VH:VL 2개 체인으로 ColabFold Multimer v3 예측. 표적은 별도 예측 후 모든 도킹에 공통으로 사용합니다.
- Fv chain H/L은 한 도킹 파트너로 유지하고, 공식 `restrain_bodies`로 내부 조립을 보존합니다. H/L 잔기 번호를 충돌 없이 부여하고 대응표를 저장합니다.
- HADDOCK 입력에서는 공식 튜토리얼처럼 Fv의 chain/segid를 B 하나로 표기합니다. **연결 서열을 추가하지 않으며**, `topoaa`가 VH/VL 사이 peptide link를 제거한 것을 확인합니다. 정제 후에는 전역 잔기 대응표로 H/L 표기를 복원해 접촉면·PRODIGY를 계산합니다.
- CDR 중 상대 측쇄 접근성 ≥40%인 잔기와 PD-L1 전체 접근 가능 표면 사이에 AIRs를 생성합니다. 특정 치료용 항체의 epitope로 제한하지 않습니다.
- rigidbody → 상위 포즈 선택 → flexref → emref → FCC clustering(0.6, 최소 4개) → 상위 클러스터/포즈. 최초 파일럿에서는 충분한 클러스터가 생기지 않을 수 있습니다.
- 대표 클러스터의 상위 4개 점수 평균·변동, 전체 클러스터 크기, 항체–표적의 중원자 4 Å 접촉, CDR 접촉 비율, SASA 차이로 계산한 접촉면을 기록합니다. H–L 내부 접촉은 제외합니다.
- PRODIGY: `--selection H,L A`, 25°C, 접촉 cutoff 5.5 Å. 포즈 의존적 ΔG/Kd 추정치이며 실제 결합 여부·실측 친화도가 아닙니다.
- PD-L1 당화 위치 N35/N192/N200/N219의 ND2와 항체 사이 <8 Å, ECD 말단 10잔기와 <8 Å를 경고로 표시합니다. 실제 당쇄/막 충돌 계산을 대체하지 않습니다.
- 정답이 없는 생성 항체에서는 DockQ/native RMSD 계산을 끕니다. 부족한/실패한 후보는 누락 사유와 함께 남기며, MAGE와 LoRA를 임의로 짝지어 통계 처리하지 않습니다.

## 방법 출처

- [ColabFold 공식 코드](https://github.com/sokrypton/ColabFold)
- [HADDOCK3 항체–항원 공식 튜토리얼](https://www.bonvinlab.org/education/HADDOCK3/HADDOCK3-antibody-antigen/)
- [미지 epitope 시나리오](https://www.bonvinlab.org/haddock3-user-manual/docking_scenarios/antibody-antigen.html)
- [PRODIGY 공식 코드](https://github.com/haddocking/prodigy)
- [Ford et al. 2024: ColabFold→HADDOCK→PRODIGY를 사용한 생성 항체 사례](https://pmc.ncbi.nlm.nih.gov/articles/PMC11375674/)

**현재 단계:** 입력 42쌍과 공통 표적 준비 완료. 첫 ColabFold 파일럿 실행 중. 아직 본 도킹/최종 후보 순위 결과는 없습니다.

2026-11-14: 사용자가 Colab GPU로 구조 예측 이전 및 Google Colab에 후보 서열·코드 전송을 승인했습니다. `구조평가_Colab.ipynb`는 현재 코드/manifest/FASTA/기존 표적 MSA를 담은 단독 실행 노트북입니다. GPU 선택 후 셀을 순서대로 실행하고 마지막 셀에서 구조·점수·설정·로그 ZIP을 내려받습니다. 이 노트북은 구조 예측 단계만 실행하며 HADDOCK는 별도 로컬 단계입니다.

별도 양성 대조군 참고값: 공개 실험 구조의 해상된 Fv–PD-L1 IgV를 이용한 접촉면/PRODIGY 분석을 수행했습니다. 전체 220 aa 표적을 사용한 생성 후보 비교와는 범위가 달라 섞어 순위를 만들지 않습니다. `reference-pilot-v3`는 같은 실험 구조로 수행하는 소규모 redocking 자원 확인이며 본 42쌍 비교가 아닙니다.

### GPU 호환성 기록

초기 JAX 0.10.2 실행은 `libcuda.so.1.1`에서 SIGSEGV로 종료됐습니다. 같은 버전에서 `XLA_FLAGS=--xla_gpu_autotune_level=0`를 적용해도 종료돼, 공식 Blackwell 사례의 JAX 0.10.1로 고정했습니다. [ColabFold 공식 이슈 #841](https://github.com/sokrypton/ColabFold/issues/841). 실패 로그는 각 예측 폴더의 `attempts/`에 보존하고 MSA·가중치를 재사용합니다. 아직 호환성 해결 여부는 첫 구조가 완성돼야 확인할 수 있습니다.
