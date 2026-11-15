# 구조 평가 진행 자료 — 2026-11-15

- ColabFold Fv 42쌍 + 공통 PD-L1 구조 예측 완료, ZIP 로컬 저장 및 입력/설정 검사 통과.
- `folding_summary.json`: 각 후보의 VH/VL/CDR pLDDT, VH–VL 평균 PAE와 대표 구조 SHA256. 이는 PD-L1 결합 신뢰도나 결합력 지표가 아니다.
- `main_settings.json`: 본 도킹의 공통 예산(1,000 rigidbody / 100 refinement), 시드 42와 계산 순서. status는 이 파일을 복사한 시점의 스냅샷이다.
- 본 도킹: 양성 대조군 2개 후 생성 항체 40개. 후보마다 완료 시 접촉면·PRODIGY 분석 및 그림을 로컬 결과 폴더에 저장한다. 본 도킹 점수와 전체 순위는 아직 미완료다.
- 대용량 전체 구조 ZIP과 도킹 원본은 GitHub에 올리지 않는다. 로컬 경로와 재개 방법은 `항체실험/구조평가_환경/재개안내.md`에 기록했다.
