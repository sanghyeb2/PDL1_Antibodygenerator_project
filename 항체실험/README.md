# 항체실험

실행 파일은 **MAGE.py**와 **LoRA.py** 두 개다.

- MAGE.py: 원본 MAGE 생성·NLL 평가.
- LoRA.py: 데이터 분할·LoRA 학습·학습된 모델 생성·NLL 평가·시각화.

각 파일 맨 위에 설정값을 모았고, 처리 코드를 파일 안에 직접 넣었다.
기존 가상환경을 자동 선택한다. 실행 명령은 **시작안내.md**를 확인한다.

```powershell
python 항체실험/MAGE.py
python 항체실험/LoRA.py
python 항체실험/LoRA.py --mode plot --show
```

앞의 두 명령은 각각 PD-L1 항체 한 쌍을 생성하며, 마지막 명령은 기존 결과 그림을 보여준다.
