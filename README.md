# art-lab
한양 숨바꼭질 그림 생성 테스트 (무료 SDXL 애니메이션 모델, GitHub 서버 CPU).
생성 결과는 `out/`, 한눈에 보기는 `sheet.jpg`.

## 트라이브 빌리지 (tribez-village) 그림
- 워크플로 `트라이브 빌리지 그림 생성` (`.github/workflows/tribez.yml`) — 무료 SDXL Turbo 계열 모델(기본 DreamShaper XL v2 Turbo)로 8단계 생성 → rembg 로 배경 지움
- 프롬프트: `tribez/prompts.json`, 결과: `tribez/out/`, 한눈에 보기: `tribez/sheet.jpg`
- 묶음 생성: `ids=all` 이면 `tribez/prompts.json` 전부(건물·자연·인물·초상) × 시드. 분류별 모음: `tribez/sheet_buildings.jpg`, `sheet_nature.jpg`, `sheet_characters.jpg`, `sheet_portraits.jpg`
- 게임 크기로 줄인 그림: `tribez/game/` (건물 가로 512, 인물·초상 256)

## 엠버폴 (dark-sanctum) 그림
- 워크플로 `엠버폴 그림 생성` (`.github/workflows/ember.yml`) — 트라이브 빌리지와 같은 무료 SDXL Turbo 방식, 디아2풍 다크 판타지 화풍
- 프롬프트: `emberfall/prompts.json` (몬스터·보스·NPC·건물), 결과: `emberfall/out/`, 한눈에 보기: `emberfall/sheet*.jpg`
- 원칙: 디아2의 자리·역할은 따르되 생김새는 새로 — 원작 고유 디자인 복사 금지. 앱에 넣기 전에 먼저 보여 주고 고르기

## 부여전기 (buyeo-rpg) 그림
- 워크플로 `부여전기 그림 생성` (`.github/workflows/buyeo.yml`) — 무료 SDXL Turbo(DreamShaper XL v2 Turbo) → rembg 배경 지움 → 128px 도트풍 가공(색 24개·1px 외곽선)
- 프롬프트: `buyeo/prompts.json` (고구려·부여 고대풍), 원본: `buyeo/out/`, 가공본: `buyeo/game/`, 한눈에 보기: `buyeo/sheet.jpg` (원본 | 가공)
- 원칙: 바람의나라 고유 그림 따라 하지 않기, 앱에 넣기 전에 먼저 보여 주고 고르기
