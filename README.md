# art-lab
한양 숨바꼭질 그림 생성 테스트 (무료 SDXL 애니메이션 모델, GitHub 서버 CPU).
생성 결과는 `out/`, 한눈에 보기는 `sheet.jpg`.

## 트라이브 빌리지 (tribez-village) 그림
- 워크플로 `트라이브 빌리지 그림 생성` (`.github/workflows/tribez.yml`) — 무료 SDXL Turbo 계열 모델(기본 DreamShaper XL v2 Turbo)로 8단계 생성 → rembg 로 배경 지움
- 프롬프트: `tribez/prompts.json`, 결과: `tribez/out/`, 한눈에 보기: `tribez/sheet.jpg`
- 묶음 생성: `ids=all` 이면 `tribez/prompts.json` 전부(건물·자연·인물·초상) × 시드. 분류별 모음: `tribez/sheet_buildings.jpg`, `sheet_nature.jpg`, `sheet_characters.jpg`, `sheet_portraits.jpg`
- 게임 크기로 줄인 그림: `tribez/game/` (건물 가로 512, 인물·초상 256)
