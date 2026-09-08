# AIdanChem

타겟 단백질에 대한 small-molecule 신약후보 물질 제안을 위한 Agentic Harness. Claude Code나 OpenClaw처럼
대화형으로 쓰는 걸 목표로 한다. 스킬(skill) 구조는 [bioSkills](https://github.com/GPTomics/bioSkills)의
형식을 참고했다.

전체 아키텍처, 파이프라인 설계, 각 모델의 현재 상태와 실측 검증 결과는
**[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)** 에 정리되어 있다.

## 이 저장소를 찾아온 이유가 ADMET 예측 논문 때문이라면

*Bull. Korean Chem. Soc.* 2026;47:756–768 (DOI: [10.1002/bkcs.70177](https://doi.org/10.1002/bkcs.70177))의
Data Availability Statement 관련 안내:

- 논문이 안내하는 코드 위치: **[`models/ADMETpred/`](models/ADMETpred/)**
- 본 저장소는 원래 해당 논문 코드만 포함했음. 이후 에이전틱 신약발굴 솔루션 전체의 컨테이너로 확장됨.
- 과거 별도 컨테이너 저장소 `AIdanMol`이 존재했으나 본 저장소로 통합됨.
- 논문 재현에 필요한 파일 구성은 동일함. 위치만 `models/ADMETpred/`로 이동함.

## 구성 요소

| 구성 요소 | 위치 | 역할 |
|---|---|---|
| ADMETpred | `models/ADMETpred/` | ZINC-DeBERTa 기반 22종 ADMET 엔드포인트 예측 (자체 IP, 논문 출판됨) |
| AIdanFold | `external/AIdanFold/` (git submodule) | ESMFold2 기반 단백질 구조 예측, flow-matching 헤드로 가속 (자체 IP) |
| Skills | `skills/` | bioSkills 스타일 지식 스킬 — 파이프라인의 나머지 단계(도킹, 생성, 합성가능성 등)는 여기서 제3자 도구를 감싼다 |
| Tools | `tools/` | ADMETpred/AIdanFold를 감싸는 CLI 어댑터 (JSON in/out) |

자세한 내용, 각 구성 요소의 목표 계약(target contract)과 현재 상태는 [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)를 볼 것.

## 서브모듈 초기화

```bash
git submodule update --init --recursive
```
