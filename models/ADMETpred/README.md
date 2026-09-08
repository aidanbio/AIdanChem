# ADMETpred

ZINC-사전학습 DeBERTa 기반 SMILES 인코더를 22개 ADMET 엔드포인트에 대해 멀티태스크 파인튜닝한 모델.

> **Note**: 이 디렉토리는 원래 `aidanbio/AIdanChem` 저장소 최상위였다. 그 저장소가 지금은 전체
> 에이전틱 신약발굴 솔루션(AIdanChem)의 컨테이너가 되면서, 이 모델 코드는 `models/ADMETpred/`로
> 옮겨졌다. 아래 논문의 코드/데이터 출처(`github.com/aidanbio/AIdanChem`)를 보고 오셨다면, 정확히
> 이 디렉토리가 그 코드입니다 — 파일 구성은 바뀌지 않았고 위치만 이동했다.

**논문**: Lim JH, Kim M, Han Y, Lee JY. "Improving predictive performance for molecular ADMET
properties using a chemical language model." *Bull. Korean Chem. Soc.* 2026;47:756–768.
DOI: [10.1002/bkcs.70177](https://doi.org/10.1002/bkcs.70177)

전체 솔루션 아키텍처는 저장소 루트의 [`docs/ARCHITECTURE.md`](../../docs/ARCHITECTURE.md) 참고.
