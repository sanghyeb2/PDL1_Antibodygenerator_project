# 항체실험

실행 파일은 **MAGE.py**와 **LoRA.py** 두 개다.

- MAGE.py: 원본 MAGE 생성·NLL 평가.
- LoRA.py: 데이터 분할·LoRA 학습·학습된 모델 생성·NLL 평가·시각화.

각 파일 맨 위에 설정값을 모았고, 처리 코드를 파일 안에 직접 넣었다.
기존 가상환경을 자동 선택한다. 실행 명령은 **시작안내.md**를 확인한다.

최신 진행 상황과 NLL/생성 분석 그림은 [프로젝트 README](../README.md), 완료된 2,000회 생성 결과와 최종 후보는 [공유 결과](공유결과/PDL1_MAGE_LoRA_20261114/README.md)에서 확인한다. 작업 브랜치는 `MAGE_ROLA`이며 `main`은 기준 버전으로 유지한다.

```powershell
python 항체실험/MAGE.py
python 항체실험/LoRA.py
python 항체실험/LoRA.py --mode plot --show
```

앞의 두 명령은 각각 PD-L1 항체 한 쌍을 생성하며, 마지막 명령은 기존 결과 그림을 보여준다.
