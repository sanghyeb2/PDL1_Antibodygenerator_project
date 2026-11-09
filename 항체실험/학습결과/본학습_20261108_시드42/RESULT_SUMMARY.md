# PD-L1 LoRA 실험 결과 (2026-11-08)

- 완료: 7 epoch, 77 optimizer step, 조기 종료.
- 최적 어댑터: best_adapter/ (epoch 3).
- validation NLL: 0.803423 -> 0.658714 (18.01% 감소).
- validation perplexity: 2.233172 -> 1.932305.
- GPU peak allocated: 3.4282 GiB.
- 실행 시간: 289.9초.
- 분할/학습 seed: 각각 42. 학습 85쌍 / 검증 21쌍.
- 원본 모델 고정 및 저장한 최적 어댑터 재로딩 확인.
- 검증셋은 checkpoint 선택에 사용했다. 독립 test 성능이나 실험적 결합력을 의미하지 않는다.

자세한 설정과 실행 명령: ../../LORA_TRAINING.md
