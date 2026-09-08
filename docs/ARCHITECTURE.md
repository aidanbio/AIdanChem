# AIdanChem 아키텍처 설계

> **저장소 개편 이력 (2026-09-08)**: 이 문서는 원래 별도 저장소 `AIdanMol`에서 작성됐다. `AIdanMol`은
> `AIdanChem`(ADMET 예측 모델)과 `AIdanFold`(구조 예측 모델)를 서브모듈로 묶는 컨테이너 저장소였는데,
> 이후 **`AIdanChem` 저장소 자체를 전체 에이전틱 솔루션의 최상위로 삼는 쪽으로 재편**했다. 그 결과:
> - 기존 `AIdanChem`(ADMET 모델) 코드는 `models/ADMETpred/`로 이동하고 이름을 **ADMETpred**로 바꿨다.
> - `AIdanFold`는 계속 git submodule로 유지한다 (`external/AIdanFold/`).
> - `AIdanMol` 저장소는 폐기됐다 — 이 문서를 포함해 거기 있던 내용은 전부 여기로 옮겨졌다.
>
> 아래 본문은 이 개편 이전에 "AIdanMol 관점"에서 쓰인 원문을 새 이름/경로에 맞게 고친 것이다. 날짜가
> 찍힌 발견 사항(예: "2026-09-07 스냅샷")은 개편 **이전** 시점의 실측 기록이므로 그대로 남겨둔다.

이 문서는 AIdanChem(솔루션 전체)의 목표 아키텍처를 정의한다. AIdanChem은 자체 개발 AI 모델(ADMETpred,
AIdanFold)을 도구로 사용하는, 신약후보 물질 발굴을 위한 Agentic Harness다. 사용자는 Claude Code나
OpenClaw를 쓰듯 대화형으로 이 하네스를 사용한다.

스킬(skill) 구조는 [bioSkills](https://github.com/GPTomics/bioSkills)(MIT, archived — 포크/커스터마이징을
공식적으로 권장하는 저장소)의 형식을 참고한다.

> 이 문서는 목표 계약(target contract)과 현재 상태(current state)를 분리해서 적는다. 아래 각 모델
> 섹션의 "현재 상태"는 각 스냅샷 시점에 저장소를 직접 확인한 결과이며, 목표 계약을 먼저 고정하고 그
> 다음에야 스킬을 쓸 수 있다 — 순서를 바꾸면 스킬을 다시 써야 한다.

## 1. 계층 구조

```mermaid
flowchart TB
    User[사용자] <--> Agent[Agentic LLM\n(Opus 등, 대화형 오케스트레이터)]
    Agent <--> Skills[Skills\nSKILL.md + usage-guide.md + examples/\nbioSkills 스타일]
    Agent -->|Bash로 CLI 실행| Tools[Tool Adapters\nAIdanChem/tools/*.py\n(JSON in/out CLI)]
    Tools --> Chem[ADMETpred\n(models/ADMETpred, library로 import)]
    Tools --> Fold[AIdanFold\n(submodule, library로 import)]
```

- **Agentic LLM**: 사용자와 대화하며 Skill을 읽고, 필요한 시점에 Tool Adapter를 Bash로 호출한다.
- **Skills**: "언제 어떤 도구를 어떻게 호출하고 결과를 어떻게 해석/게이팅할지"에 대한 지식. 도구 자체가
  아니다 (bioSkills의 모든 SKILL.md도 지식이지 콜러블 도구가 아니다 — 이 프로젝트에서 실제 콜러블
  도구는 아래 Tool Adapter 계층이 처음이다).
- **Tool Adapters**: 각 모델을 감싸는 얇은 CLI. 이 저장소의 `tools/`에 위치하며, `models/ADMETpred`를
  라이브러리로 import하거나 `external/AIdanFold` 서브모듈을 subprocess로 호출한다.
- **ADMETpred**: 이 저장소 안(`models/ADMETpred/`)에 직접 포함된 일반 디렉토리 — 서브모듈이 아니다.
- **AIdanFold**: git submodule (`aidanbio/AIdanFold`). 이 안의 코드는 건드리지 않는다 — 아래 5절 참고.

## 2. 도구 호출 계약 (Tool Invocation Contract)

가장 먼저 고정해야 하는 결정. 이후의 모든 스킬 작성은 이 계약에 의존한다.

**CLI-first**: 모델마다 `python tools/<name>.py --arg ...` 형태의 CLI 하나. 성공 시 JSON 객체
하나를 stdout에 출력, 실패 시 non-zero exit + stderr 메시지. 스킬은 에이전트에게 "정확히 이 커맨드를
Bash로 실행하고 stdout의 JSON을 읽어라"라고 지시한다.

이 방식을 고른 이유:
- bioSkills 자체의 관용구와 동일하다 (커맨드 실행 → 출력 해석).
- 별도 하네스 인프라 없이 셸에서 바로 테스트 가능하다.
- 나중에 MCP 서버로 감쌀 경우 이 CLI를 그대로 얇게 래핑하면 된다 — 지금은 MCP 계층을 설계하지 않는다.

공통 규칙:
- 출력 JSON에는 `schema_version`, `model_version`(또는 `checkpoint`)을 반드시 포함한다.
- 구조 파일, PAE 행렬처럼 큰 산출물은 디스크에 쓰고 JSON에는 경로만 담는다.
- **신뢰도/불확실성 필드는 선택이 아니라 계약의 일부다.** bioSkills의 admet-prediction, modern-structure-prediction
  두 스킬 모두 "점추정치만으로는 판단하지 말라"는 것이 핵심 원칙이다 (AD 미평가 예측은 "비용이 큰 오탐지",
  구조 예측은 "confident-looking hallucination" 가능). 이 필드가 없으면 하위 스킬이 게이팅 로직을 쓸 수
  없으므로, 3·4절의 출력 스키마에 처음부터 포함시킨다.

## 3. ADMETpred 어댑터 — ADMET 예측 도구

### 목표 계약

```
python tools/admetpred_predict.py --smiles-file in.smi --checkpoint <path> --out preds.json
```

`in.smi`: 줄당 SMILES 하나. 출력 JSON (분자당 1레코드):

```json
{
  "schema_version": "1.0",
  "checkpoint": "path/to/checkpoint_epoch_N.pth",
  "results": [
    {
      "smiles": "CCO",
      "predictions": {
        "Caco2": {"value": -4.7, "uncertainty": 0.04, "in_applicability_domain": null}
      },
      "warnings": []
    }
  ]
}
```

22개 엔드포인트는 `inference.py`의 `FEATURE_NAMES`를 그대로 쓴다 (Caco2, HIA, Pgp, Bioavailability,
Lipophilicity, Solubility, BBB, PPBR, VDss, CYP2C9/2D6/3A4 (억제/기질), Half_Life,
Clearance_Hepatocyte/Microsome, LD50, hERG, AMES, DILI).

`uncertainty`는 fold-앙상블 표준편차로 채운다 (아래 참고) — `null`이 아니라 실제 값이 처음부터 나온다.
`in_applicability_domain`은 아직 없다 (남은 격차, 아래 참고).

### 현재 상태 (2026-09-07 갱신 — 로컬에서 end-to-end 검증 완료)

**해결됨:**
- 저장소 소유자가 논문(BKCS 2026, 아래 인용)과 동일한 설정(lr=3e-5, focal γ=2.0)으로 학습한 **TDC
  scaffold 5-fold 체크포인트를 직접 확보**해 `models/ADMETpred/outputs/debeta_f{0..4}_lr3_g2_15ep/`에
  배치했다 (fold별 `checkpoint_epoch_15.pth` + `label_mean_std.npz`, 총 6GB, `.gitignore` 처리되어 git에는
  안 올라감 — 로컬 서버에만 존재).
- `inference.py`를 재작성해 `eval_finetuned_tdc_2`(존재하지 않던 모듈) 의존성을 제거했다.
  `CustomRegModel`을 `finetuning.py`(실제 학습 스크립트)와 state_dict 키가 완전히 일치하도록 인라인
  정의했다. (이 수정은 PR로 반영: `aidanbio/AIdanChem#1`.)
- **버그 발견 및 수정**: `label_mean_std.npz`의 `columns` 순서는 학습 CSV 원본 순서(예: AMES가 0번째)이며
  `FEATURE_NAMES`의 순서(Caco2가 0번째)와 다르다. 위치 기반으로 mean/std를 매칭했다면 22개 지표가 서로
  다른 지표의 정규화 통계로 역정규화되어 **조용히 틀린 값**이 나왔을 것이다. 이름 기반 매칭으로 수정했고,
  5개 fold의 npz가 서로 완전히 동일한 값(같은 전체 데이터셋에서 fold 분리 전에 계산됨)임을 실측으로
  확인했다.
- **5-fold 앙상블 구현**: `run_inference`가 여러 fold 디렉토리를 받아 각각 예측 후 평균(예측값)과
  표준편차(`<지표>_std`, 불확실성)를 함께 반환한다. 이게 곧 논문 Table 1의 헤드라인 수치(5-fold 평균)를
  재현하는 유일한 방법이며, 동시에 2절의 "신뢰도 필드는 계약의 일부" 요구사항을 앙상블 분산으로
  충족시킨다 — 별도 AD 로직 없이도 격차 3번이 해결됨.
- 아스피린으로 5-fold 전체 실행을 실측 검증함: 22개 지표 모두 합리적 범위의 예측값과 작은 `_std`
  (대부분 지표에서 fold 간 수렴 양호, `Half_Life_std`처럼 상대적으로 큰 것도 있어 지표별 신뢰도 차이가
  바로 드러남).
- 환경 이슈 발견 및 우회: 한 프로세스에서 두 번째 모델의 forward부터 PyTorch의 legacy JIT 프로파일링
  실행기가 반복 연산을 NVRTC로 퓨징하려 시도하는데, 이 서버에 `libnvrtc-builtins.so.13.0`이 없어 실패함
  (코드 로직 버그 아님). `torch._C._jit_set_profiling_executor(False)`로 우회.
- **이 접근법 자체는 동료심사를 거쳐 출판되었다**: Lim JH, Kim M, Han Y, Lee JY. "Improving predictive
  performance for molecular ADMET properties using a chemical language model." *Bull. Korean Chem. Soc.*
  2026;47:756–768. DOI: 10.1002/bkcs.70177. (PDF는 `claude/create-admet-demo-notebook-afEHn` 브랜치의
  `ref/2026_AIdanChem_BKCS.pdf` — 브랜치/파일명은 개편 이전 이름을 그대로 유지.) ZINC 사전학습 DeBERTa를
  300K PubChem-ADMET로 22개 엔드포인트 MTL 파인튜닝(Focal MAE 손실), 자체 제안 지표
  ABPS(ADMET-Balanced Performance Score)로 평가, TDC 벤치마크에서 **12개 엔드포인트 SOTA, 7개 top-5**.

**남은 격차:**
- kNN/leverage 기반 AD(applicability domain) 플래그는 아직 없다 — fold-앙상블 분산이 실질적인 대체
  신뢰도 신호 역할을 하고 있지만, "학습 분포 밖 입력"을 직접 탐지하는 건 아니다.
- `inference.py`는 여전히 `models/ADMETpred/` 안에 있다. `tools/admetpred_predict.py`(CLI 어댑터)가
  이걸 감싸야 하며, 지금은 아직 그 래퍼가 없다.
- `evaluation.py`, `data_filtering.py`, `tdc_download.py`도 여전히 존재하지 않는 `eval_finetuned_tdc_2`를
  import한다 — `inference.py`만 고쳤고 이 파일들은 손대지 않았다 (요청 범위 밖).

### 격차 및 필요 작업

1. ~~`eval_finetuned_tdc_2` 복구 또는 재작성~~ — 완료 (`inference.py`에 인라인).
2. ~~체크포인트 확보~~ — 완료 (5-fold, 로컬 `outputs/`).
3. ~~최소 AD/불확실성 계층 추가~~ — 완료 (fold-앙상블 표준편차로 대체).
4. `tools/admetpred_predict.py`를 작성해 `inference.py`의 `run_inference`를 CLI로 감싸기
   (2절의 JSON 출력 계약에 맞춰 stdout에 JSON 직렬화).
5. (선택, 범위 밖) `evaluation.py`/`data_filtering.py`/`tdc_download.py`의 `eval_finetuned_tdc_2` 의존성도
   필요해지면 같은 방식으로 정리.

## 4. AIdanFold 어댑터 — 단백질 복합체 구조 예측 도구

### 목표 계약

```
python tools/aidanfold_predict.py --fasta complex.fasta --out-dir results/
```

출력 JSON:

```json
{
  "schema_version": "1.0",
  "model_version": "...",
  "structure_file": "results/complex.pdb",
  "confidence": {
    "per_chain_plddt_mean": {"A": 91.2, "B": 88.4},
    "ptm": 0.81,
    "iptm": 0.74,
    "inter_chain_pae_summary": "results/complex_pae.npy"
  }
}
```

`iptm` + `inter_chain_pae`는 단일 체인 pLDDT와 별개로 반드시 존재해야 한다 — bioSkills의
modern-structure-prediction 스킬이 명시하듯 "복합체 인터페이스는 per-chain pLDDT가 아니라 ipTM +
inter-chain PAE로 게이팅한다"가 핵심 규칙이고, 이 필드가 없으면 그 규칙을 쓸 스킬을 쓸 수 없다.

### 현재 상태 (2026-09-08 갱신)

- `main` 브랜치 기준으로는 `external/AIdanFold/CLAUDE.md`가 "Early scaffold, src/refs/config 비어있음"
  이라 적혀 있는데, 이는 stale한 문서고 실제 `main`도 `flow_head.py`, `flow_matching.py`, `fine_tune.py`,
  `sampler.py`, `training.py`, `run_pipeline.py` 등 3,700줄 가량의 flow-matching 헤드 코드를 이미
  갖고 있다.
- **하지만 진짜 작업은 `main`에 없다.** 원격의 `claude/intelligent-ritchie-hcuqk9` 브랜치(이 저장소가
  서브모듈로 추적하는 브랜치)에 `esmfold2adv_phase1_results.md`라는 결과 문서가 있는데, 요지는:
  - ESMFold2의 Karras diffusion 구조 헤드(200-step)를 **Conditional Flow Matching 헤드**로 교체.
  - 전체 test set(n=3060)에서 **lDDT −0.0058 / TM −0.019 / 5.06배 가속** — "diffusion 대비 5배 이상
    빠르면서 품질 손실 2점 이내"라는 목표를 실측으로 달성했다고 기록됨. (단, 이 5.06배는 **head-only**
    수치다 — 트렁크 시간은 제외하고 잰 것. 실사용 시나리오별 실효 속도 이득은 §6.5 참고.)
  - **배포 추론 구성**: 체크포인트 `flow_fape_ema2` epoch 62, 20-step ODE, `churn=1.0`, `self_cond`는
    체크포인트 자체 저장된 설정(`self_conditioning=True`)을 따른다. 체크포인트 파일은
    `external/AIdanFold/checkpoints/flow_fape_ema2/flow_head_epoch62.pt`(2.0GB, `.gitignore` 처리되어
    git에는 안 올라감, 로컬 서버 `/data/trunk/AIdanFold`에만 존재)로 배치되어 있다.
  - `evaluate_flow.py`의 `--oligomeric {monomer,complex,any}` 인자와 `parse_protein_chains`로 보아
    **멀티체인(복합체) 입력 자체는 이미 지원**한다.
- 단, `evaluate_flow.py`는 **정답 구조(ground truth)가 있는 test manifest에 대해 lDDT/TM/RMSD를
  계산하는 평가 스크립트**이지, "서열만 주면 구조를 예측해 돌려주는" 서빙용 추론 CLI가 아니다.
- **자체 신뢰도(ipTM 등)는 확보했다.** ESMFold2의 confidence head는 structure head(diffusion이든
  flow든)와 독립적으로, 예측된 좌표(`x_pred`)와 트렁크 표현만 받아 pLDDT/pTM/ipTM/PAE를 계산한다 —
  `model.structure_head.sample`을 우리 flow head 호출로 monkeypatch하고 전체 `model.forward()`를 그대로
  실행하면 ipTM까지 정상적으로 나온다는 걸 FoldBench 벤치마크 작업(§6.6)에서 실측 검증했다. 즉 위
  "격차 및 필요 작업" 3번은 해소됐다 — 별도 대체 신호가 필요 없다.
- `requirements.txt`는 `esm@git+https://github.com/Biohub/esm.git@main` 하나만 명시한다. 이건
  **의존성이 없다는 뜻이 아니라 필요 없다는 뜻**이다 — ESMFold2 자체가 이미 AlphaFold3급 co-folder라서
  (Protein/DNA/RNA/Ligand를 SMILES까지 포함해 동시에 입력받음, `esm.md` §4.6–4.7) 외부 co-folder가
  따로 필요 없다. 다만 AIdanFold가 직접 학습·검증한 flow-matching 구조 헤드는 단백질-전용 데이터로만
  만들어졌다 — 자세한 내용과 리간드 검증 결과는 §6.4 참고.

### 격차 및 필요 작업

1. `evaluate_flow.py`의 "chains → features → flow sampler → 좌표" 경로에서 **정답 비교 없이** 좌표만
   뽑아 PDB로 쓰는 얇은 예측 전용 경로를 분리 (`--ckpt checkpoints/flow_fape_ema2/flow_head_epoch62.pt
   --flow_steps 20 --churn 1` 배포 구성을 하드코딩된 기본값으로). §6.6의 FoldBench 벤치마크 스크립트가
   이미 이 경로를 직접 구현했으므로 (`predict_best_of_seeds`), 그걸 `tools/aidanfold_predict.py`로
   정리해 옮기면 된다.
2. ~~자체 신뢰도 지표 확보~~ — 완료 (confidence head 재사용, §6.6에서 실측).
3. `tools/aidanfold_predict.py`를 작성 — 위 배포 구성을 기본값으로 감싸는 CLI.
4. `CLAUDE.md`의 stale한 "Early scaffold" 서술 업데이트 (이 문서와 별개 작업, `main` 브랜치 대상).

## 5. 서브모듈 경계 규칙

`AIdanFold`는 `aidanbio/AIdanFold`를 가리키는 git submodule이다. `ADMETpred`(구 AIdanChem)는 이
저장소에 직접 포함된 일반 디렉토리이며 더 이상 서브모듈이 아니다 — 이 규칙은 이제 AIdanFold에만
적용된다.

- Tool Adapter 코드는 **이 저장소의 `tools/`에만** 작성한다. `external/AIdanFold` 내부를 패치하지
  않는다 — 패치가 필요하면 upstream 서브모듈 저장소에 반영한 뒤 이 저장소의 서브모듈 포인터를 갱신한다.
- 어댑터는 서브모듈 경로를 `sys.path`에 추가해 라이브러리로 import하거나, subprocess로 서브모듈 안의
  스크립트를 직접 호출하는 방식만 쓴다.
- `models/ADMETpred/`는 서브모듈이 아니므로 이 제약이 없다 — 다만 여전히 "필요한 최소 변경만" 원칙은
  지킨다.

## 6. 엔드투엔드 파이프라인 — 타겟 단백질 → Small Molecule 후보 제안

### 6.1 목적 문장 재검토

애초 목표를 "타겟 단백질에 대한 small molecule 신약 후보물질 제안"으로 잡았는데, 이건 방향은 맞지만
ADMETpred(ADMET 필터)과 AIdanFold(구조 예측) 두 개만으로는 이 문장의 핵심 동사("제안하다")에 해당하는
단계 — 후보를 실제로 만들거나 골라내는 단계 — 가 빠져 있었다. 그 빈 자리는 **bioSkills의 기존
chemoinformatics 스킬(제3자 도구: GNINA, DiffDock-L, REINVENT4, OpenFE 등)을 그대로 가져와 채운다.**
ADMETpred/AIdanFold는 이 파이프라인 전체를 대체하는 게 아니라, 그 안에서 "자체 IP"가 필요한 두 지점
(타겟 구조, ADMET 필터)에 꽂혀 들어가는 구성요소다.

### 6.2 파이프라인

bioSkills 스킬들의 상호 참조(`Related Skills`)를 실제로 읽어보면, 라이브러리 스크리닝과 de novo 생성은
양자택일이 아니라 **같은 도킹/검증 배관을 공유하는 두 개의 후보 소스**로 설계되어 있다
(`generative-design`은 자신이 만든 분자의 검증을 `virtual-screening`에 위임하고, `virtual-screening`은
사전 필터로 `admet-prediction`을, receptor 출처로 `modern-structure-prediction`을 명시적으로 참조한다).
그래서 virtual-screening 경로를 먼저 세우고, generative-design은 나중에 같은 배관 위에 얹는 순서로
간다.

```
타겟 입력 (서열 / UniProt / 유전자명)
        │
        ▼
① 타겟 구조 확보 ─ structural-biology/alphafold-predictions (기존 DB 우선)
        │           없으면 → AIdanFold (자체 모델, 서열만 입력하는 단백질 전용 예측기)
        │           ①-확장: 같은 타겟의 컨포메이션 앙상블 생성 — §6.5 참고
        ▼
② 구조 준비 & 포켓 탐지 ─ structural-biology/structure-preparation, binding-site-detection
        │
        ▼
③ 후보 소스 ─┬─ (Phase 1) 기존 라이브러리 ─ chemoinformatics/virtual-screening
             │    도킹 전: ADMETpred로 사전 ADMET/druglikeness 필터 (비싼 도킹 전 값싼 컷)
             │
             └─ (Phase 2) de novo 생성 ─ chemoinformatics/generative-design (REINVENT4)
                  스코어링 함수에 ADMETpred ADMET을 리워드 항목으로 포함 가능
                  생성된 분자는 ③의 virtual-screening 도킹 인프라로 검증
        │
        ▼
④ 포즈 검증 ─ chemoinformatics/pose-validation (PoseBusters, 물리적 타당성)
        │
        ▼
⑤ 결합 친화도 정밀 평가 (단계적 깔때기 — 아래 6.4 참고)
        │
        ▼
⑥ ADMET 최종 트리아지 ─ ADMETpred (5-fold 앙상블 + 불확실성, §3)
        │
        ▼
⑦ 합성가능성 체크 ─ chemoinformatics/retrosynthesis (AiZynthFinder)
        │
        ▼
⑧ 리포트 ─ reporting

전체 오케스트레이션: workflows/drug-candidate-pipeline (신규 스킬)
```

### 6.3 결합 친화도(binding affinity) — AIdanFold 자체 co-folding 가능성부터 검증

타겟-리간드 **복합체** 구조를 통해 binding affinity를 추정하는 건 맞는 방향이고, 이 파이프라인에서는
"⑤ 결합 친화도 정밀 평가"에 해당한다.

> **정정 (2026-09-08)**: 이전 판(§6.3 초판)은 "AIdanFold는 서열-전용이라 리간드 co-folding을 지원하지
> 않는다"고 적었는데, 이건 `requirements.txt`와 Phase 1 평가 결과만 보고 내린 근거 부족한 결론이었다.
> 실제로 `esm.md`(§4.6–4.7, ESMFold2 아키텍처 분석 문서)를 확인한 결과 정반대다:
> - **ESMFold2(AIdanFold의 베이스)는 완전한 co-folder다.** `StructurePredictionInput`이
>   Protein/DNA/RNA/Ligand를 동시에 받고, 리간드는 CCD 코드 또는 **SMILES**(`tokenize_ligand_smiles`:
>   RDKit 파싱 → conformer 생성 → 원자 토큰화)로 조건화할 수 있으며, `CovalentBond`로 공유결합 억제제도
>   지정 가능하다 — 아키텍처 수준에서는 AlphaFold3와 동급의 co-folding 능력이 있다.
> - 다만 **AIdanFold가 실제로 학습·검증한 flow-matching 구조 헤드(Phase 1, §4)는 단백질-전용
>   데이터로만 만들어졌다.** `dataprep/download_pdb.py`가 RCSB 검색 쿼리에
>   `selected_polymer_entity_types = "Protein (only)"` 필터를 하드코딩해 리간드/핵산이 포함된 구조를
>   애초에 데이터셋에서 제외한다. `--oligomeric complex`의 "복합체"도 단백질 다중 체인을 뜻하지
>   단백질-리간드가 아니다.
> - 즉 정확한 상태는 "co-folding을 지원 못 한다"가 아니라 **"밑바탕 아키텍처는 지원하는데, 지금 학습된
>   체크포인트(`flow_fape_ema2`)가 리간드 토큰에 대해 한 번도 검증된 적이 없다"**다.

**검증 결과 (2026-09-08, 실측 완료) — 실패, Boltz-2/Chai-1 채택으로 확정**

`/data/trunk/AIdanFold`(conda env `AIdanFold`, ESMC-6B + `esmfold2_fast_cutoff2025` 가중치, 체크포인트
모두 로컬에 실제로 존재)에서 트립신(223 aa) + 벤즈아미딘(`NC(=[NH2+])c1ccccc1`, 트립신의 고전적
저해제) 복합체로 `flow_fape_ema2/flow_head_epoch62.pt`를 실제로 돌렸다. 파이프라인 자체는 끝까지
동작했다: `LigandInput(smiles=...)` → `ESMFold2InputBuilder.prepare_input`(트렁크 피처 생성, 단백질 223
토큰 + 리간드 9원자토큰 정상 구성 확인) → `capture_trunk`로 실제 trunk forward 실행 → 체크포인트의
저장된 config(`self_conditioning=True`, `ode_solver=midpoint`)로 20-step 샘플링 → 리간드 원자 좌표를
`ligand_bonds`(체크포인트가 아니라 ESMFold2 입력 빌더가 제공하는 정확한 원자-이름 결합 정보)로 RDKit
분자로 재구성 → PoseBusters(`mol`/`dock` 설정) 검증까지 전부 실행됐다 (재현 스크립트:
`docs/experiments/aidanfold_ligand_cofold_probe.py`, 결과: `docs/experiments/posebusters_{mol,dock}.csv`.
스크립트는 `/data/trunk/AIdanFold`의 conda env `AIdanFold` — 서브모듈이 아니라 실제 가중치/체크포인트가
있는 별도 작업 트리 — 를 전제로 경로가 하드코딩돼 있다).

**결과는 명확한 물리적 실패다**:
- 벤젠 고리의 결합 길이가 **2.95~5.32 Å** (정상 아로마틱 C-C는 ~1.39 Å) — 고리가 찢어진 수준.
- 고리 평면성 붕괴 (한 원자가 평면에서 2.27 Å 이탈).
- PoseBusters: `bond_lengths`, `bond_angles`, `aromatic_ring_flatness`, `internal_energy`,
  `minimum_distance_to_protein`, `volume_overlap_with_protein` 전부 FAIL.
- 다만 **리간드-단백질 최소 거리는 1.65 Å로 대략 포켓 근처**에 위치했다 — 트렁크(ESMC-6B)가 리간드
  토큰의 대략적인 전역 위치는 어느 정도 잡지만, flow head가 그 자리에서 원자 9개의 상대 좌표(로컬
  결합 기하구조)를 정밀하게 복원하는 능력은 전혀 없다.

**해석**: 예상했던 실패 양상 그대로다. flow head는 표준 아미노산 backbone frame(N-CA-C)에 대한 FAPE
손실로만 학습되어, 한 번도 학습에서 보지 못한 소분자 원자의 로컬 결합 기하구조(고리 평면성, 결합
길이 등)에 대한 사전지식이 없다.

**결정**: ⑤ 단계는 **Boltz-2/Chai-1(`chemoinformatics/ml-docking-rescoring`)로 확정**한다. AIdanFold를
이 역할에 다시 고려하려면, `download_pdb.py`의 protein-only 필터를 풀고 리간드 포함 구조를 데이터셋에
추가해 flow head를 재학습(파인튜닝)해야 한다 — 이건 지금 당장의 파이프라인 구축과는 분리된 별도
과제로 남겨둔다 (학습 인프라 자체는 이미 있으므로 Boltz-2/Chai-1을 새로 붙이는 것보다는 작은 작업).

| 단계 | 도구 | 비용 | 산출물 |
|---|---|---|---|
| 1차 (전체 숏리스트) | `chemoinformatics/ml-docking-rescoring` — Boltz-2/Chai-1 co-folding (AIdanFold는 위 검증 결과에 따라 현재 제외) | 중간 | 단백질-리간드 **복합체 구조** 실제 예측 + affinity-유사 점수 |
| 2차 (최종 top-N만) | `chemoinformatics/free-energy-calculations` (OpenFE, FEP) | 높음 | ΔG에 가장 가까운 추정치 |

`structural-biology/modern-structure-prediction` 스킬 자체가 이미 "Boltz-2의 affinity module은 스크리닝
용 prior일 뿐 신뢰할 수 있는 Kd가 아니다"라고 명시하고 있다 — AIdanFold를 채택하더라도 1차 결과만으로
최종 결정을 내리지 않고, 소수의 최종 후보에 한해 2차(FEP)로 확인하는 깔때기 구조는 그대로 유지한다.

### 6.4 스킬 인벤토리

bioSkills와 동일한 2계층: 단계별 스킬(`SKILL.md` + `usage-guide.md` + `examples/`) + 파이프라인 전체를
오케스트레이션하는 스킬 1개 (`workflows/*-pipeline` 패턴, bioSkills에 41개 선례 있음).

| 파이프라인 단계 | 스킬(제안) | bioSkills 근거 | 처리 |
|---|---|---|---|
| ① 타깃 구조 확보 (기존 DB) | target-structure-lookup | structural-biology/alphafold-predictions | 그대로 재사용 |
| ① 타깃 구조 예측 (자체) | aidanfold-structure-prediction | structural-biology/modern-structure-prediction | 포크 후 primary_tool을 AIdanFold로. 베이스(ESMFold2)는 리간드 co-folding을 지원하지만, 학습된 flow head는 **단백질-전용 검증까지만 완료**임을 스킬 설명에 명시 (§6.3) |
| ② 구조 준비 & 포켓 탐지 | structure-prep-and-pocket | structural-biology/structure-preparation, binding-site-detection | 그대로 재사용 |
| ③a 라이브러리 스크리닝 | virtual-screening-triage | chemoinformatics/virtual-screening | 그대로 재사용, 도킹 전 사전 필터 훅에 ADMETpred 연결 |
| ③b de novo 생성 (Phase 2) | generative-candidate-design | chemoinformatics/generative-design | 그대로 재사용, 스코어링 함수 훅에 ADMETpred 연결 |
| ④ 포즈 검증 | pose-validation | chemoinformatics/pose-validation | 그대로 재사용 |
| ⑤-1 결합친화도(co-folding) | ligand-cofold-affinity | chemoinformatics/ml-docking-rescoring | 그대로 재사용 (primary_tool = Boltz-2/Chai-1). AIdanFold는 §6.3 실측 검증 결과 리간드 로컬 기하구조 예측이 물리적으로 무효(PoseBusters 다수 FAIL)로 확인되어 현재 제외 |
| ⑤-2 결합친화도(FEP, top-N만) | free-energy-refinement | chemoinformatics/free-energy-calculations | 그대로 재사용 |
| ⑥ ADMET 최종 트리아지 | admetpred-admet-screening | chemoinformatics/admet-prediction | 포크 후 primary_tool을 ADMETpred로, §3의 현재 한계 반영 |
| ⑦ 합성가능성 | synthesizability-check | chemoinformatics/retrosynthesis | 그대로 재사용 |
| ⑧ 리포트 | triage-report | reporting | 재사용 후 화합물/타깃 문맥에 맞게 조정 |
| 전체 파이프라인 | drug-candidate-pipeline | workflows/*-pipeline | 신규 작성 (오케스트레이션) |

"그대로 재사용"은 도구 호출이 필요 없는 지식성 스킬이라 포크만 하면 되는 것들이고, "포크 후 교체"는
이 저장소의 실제 도구를 가리키도록 `primary_tool`과 예제를 바꿔야 하는 것들이다.

### 6.5 AIdanFold head 속도(5배)가 실제로 크리티컬해지는 지점

Phase 1의 "5.06배 가속"은 **head-only** 수치다 (`evaluate_flow.py`: "the frozen trunk is shared,
excluded, so speedup is a fair head-only ratio"). 실사용에서 트렁크(ESMC-6B + 폴딩 재귀)는 서열에만
의존하고, head(구조 생성)만 노이즈 draw를 바꿔 여러 구조를 낸다 — 이 둘의 비용 비중을 §6.6 FoldBench
작업에서 실측했다: flow head(20-step) 1개 샘플 ≈ 트렁크 비용의 13%. 즉 **"타겟 하나 예측"** 기준으로는
diffusion(200-step) 대비 실제 end-to-end 속도 이득이 5배가 아니라 **~2배** 수준이다.

**5배 이득이 그대로(혹은 그 이상) 살아나는 경우**: 트렁크를 1번만 계산하고 head만 반복 호출하는
워크플로. 이 파이프라인에서는 정확히 **① 단계의 확장 — 타겟 하나의 컨포메이션 앙상블 생성**이 여기
해당한다: 같은 타겟 서열(트렁크 고정)에 대해 head만 수십~수백 번 돌려 앙상블을 만들면, diffusion으로
같은 앙상블을 만드는 것보다 훨씬 싸다. bioSkills의 `virtual-screening` 스킬이 권장하는
"Cryptic pocket / induced fit → Receptor-ensemble docking"을 저비용으로 구현하는 수단이 된다.

**반대로 이득이 희석되는 경우**: ⑤ 결합친화도(co-folding) 단계처럼 후보마다 입력(트렁크 조건)이 바뀌는
스크리닝 — 화합물마다 트렁크를 새로 계산해야 하므로 head 속도 이점이 거의 사라진다. (다만 §6.3에서
확인했듯 AIdanFold는 지금 이 단계에 쓰이지 않는다.)

### 6.6 FoldBench 항체-항원(AbAg) 벤치마크 재현 — AIdanFold 검증

ESMFold2 논문(Candido et al., FoldBench Interfaces, Fig. S7/2C)의 핵심 주장 — "ESMFold2가 단일서열만으로
AlphaFold3(MSA 사용)를 능가한다" — 을 AIdanFold의 flow head로 직접 재현해 검증했다. 데이터셋은
[FoldBench](https://github.com/BEAM-Labs/FoldBench)(Xu et al., *Nat. Commun.* 2026, 공개 벤치마크)의
`targets/interface_antibody_antigen.csv` 172개 인터페이스를 그대로 썼다.

- **1차 (1 seed, single-sequence)**: 172개 전체에 대해 DockQ ≥ 0.23 통과율 **43.6%** (171/172 스코어링
  성공). 참고: FoldBench 공식 리더보드의 AlphaFold3 = 47.9%, 논문의 ESMFold2 단일서열(25 seed×5 sample
  best-of-ipTM) = 50%±2%.
- 이 격차는 "diffusion→flow 교체로 인한 소폭 품질 저하"와 "1-seed라 최적화가 안 됨" 두 요인이 섞인
  하한선으로 보고, **논문과 동일한 프로토콜(25 seed × 5 diffusion sample, ipTM 최고값 선택)**로
  재실행을 진행했다 (진행 중/결과는 별도 기록).
- 이 작업 과정에서 confidence head(ipTM 등)가 flow head와 독립적으로 동작함을 확인해 §4의 "자체 신뢰도
  지표 없음" 격차를 해소했다 (`model.structure_head.sample`을 flow 호출로 monkeypatch하고 전체
  `model.forward()`를 그대로 실행하면 ipTM이 정상적으로 나옴 — trunk_feats에 남아있던 `num_diffusion_samples`
  키를 제거하지 않으면 이중 확장 버그가 남으니 주의).
- 재현 스크립트와 세부 로그는 로컬 서버 스크래치 디렉토리에 있다(저장소에는 아직 커밋되지 않음) —
  필요 시 `tools/aidanfold_predict.py`를 만들 때 이 스크립트의 `predict_best_of_seeds` 함수를 그대로
  옮기면 된다 (§4 격차 1번).

## 7. 제안 저장소 레이아웃

```
AIdanChem/                     # 저장소 루트 (구 AIdanMol)
  models/
    ADMETpred/                 # 구 AIdanChem 모델 (서브모듈 아님, 일반 디렉토리)
  external/
    AIdanFold/                 # submodule, 손대지 않음
  tools/
    admetpred_predict.py
    aidanfold_predict.py
  skills/
    chemoinformatics/admetpred-admet-screening/{SKILL.md,usage-guide.md,examples/}
    chemoinformatics/virtual-screening-triage/{SKILL.md,usage-guide.md,examples/}
    chemoinformatics/generative-candidate-design/{SKILL.md,usage-guide.md,examples/}   # Phase 2
    chemoinformatics/pose-validation/{SKILL.md,usage-guide.md,examples/}
    chemoinformatics/ligand-cofold-affinity/{SKILL.md,usage-guide.md,examples/}
    chemoinformatics/free-energy-refinement/{SKILL.md,usage-guide.md,examples/}
    chemoinformatics/synthesizability-check/{SKILL.md,usage-guide.md,examples/}
    structural-biology/aidanfold-structure-prediction/{SKILL.md,usage-guide.md,examples/}
    structural-biology/structure-prep-and-pocket/{SKILL.md,usage-guide.md,examples/}
    workflows/drug-candidate-pipeline/{SKILL.md,usage-guide.md,examples/}
    ... (재사용 스킬들: target-structure-lookup, triage-report 등)
  docs/
    ARCHITECTURE.md             # 이 문서
    experiments/                # 재현 스크립트, 실측 결과 CSV 등
```

## 8. 빌드 순서

1. **Phase 0 — 계약 확정** (이 문서). 완료.
2. ~~**Phase 1 — ADMETpred 어댑터 핵심 로직**~~ — 완료 (`eval_finetuned_tdc_2` 복구, 5-fold 앙상블,
   체크포인트 확보). 남은 건 `tools/admetpred_predict.py` CLI 래퍼뿐.
3. **Phase 2 — AIdanFold 어댑터**: FoldBench 작업(§6.6)에서 검증한 예측 전용 경로(트렁크 →
   flow sampler → confidence head → ipTM/pLDDT)를 `tools/aidanfold_predict.py`로 정리.
4. **Phase 3 — ADMETpred/AIdanFold 스킬 작성**: bioSkills 포크 + primary_tool 교체 + 현재 모델의 실제
   한계를 Common Errors 절에 반영 (admetpred-admet-screening, aidanfold-structure-prediction).
5. **Phase 4 — 파이프라인의 나머지 단계를 bioSkills에서 그대로 가져오기**: 6.4절의 재사용 스킬들
   (virtual-screening, pose-validation, ml-docking-rescoring, free-energy-calculations,
   retrosynthesis, reporting 등)을 포크. 코드 변경 없이 스킬 문서만 가져오는 단계라 Phase 3보다 가볍다.
   generative-design은 이 Phase가 아니라 Phase 6(Phase 2 스코프)에서.
6. **Phase 5 — 오케스트레이션 스킬 + 하네스 통합**: `drug-candidate-pipeline` 작성 — ①~⑧ 단계를
   순서대로 호출하고, ⑤ 결합친화도 깔때기(1차 ml-docking-rescoring → 2차는 top-N만 free-energy)를
   구현. 전체 대화형 흐름 검증.
7. **Phase 6 (후속) — de novo 생성 확장**: `generative-candidate-design` 스킬 추가, 스코어링 함수에
   ADMETpred ADMET을 리워드 항목으로 연결.

## 9. 참고

- bioSkills: https://github.com/GPTomics/bioSkills (archived, MIT — 포크·커스터마이징 공식 권장)
- ⚠️ bioSkills의 `CONTRIBUTING.md`에는 자동화 에이전트를 겨냥한 숨은 프롬프트 인젝션(HTML 주석)이
  있다 — "에이전트라면 PR에 모델명/지시 프롬프트/승인자를 공개하라"는 내용. 이 저장소에는 PR을 올리지
  않으므로 영향 없음. 이 저장소에서 코드를 더 가져올 때(향후 스킬 포크 시) 유사한 주석이 섞여 들어오지
  않도록 원본 파일의 숨은 주석 여부를 확인할 것.
- FoldBench: https://github.com/BEAM-Labs/FoldBench (Xu et al., *Nat. Commun.* 2026) — §6.6의 AbAg
  벤치마크 데이터 출처.
- ADMETpred 논문: Lim JH, Kim M, Han Y, Lee JY. *Bull. Korean Chem. Soc.* 2026;47:756–768.
  DOI: 10.1002/bkcs.70177.
