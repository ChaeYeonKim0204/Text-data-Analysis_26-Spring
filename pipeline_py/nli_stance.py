# -*- coding: utf-8 -*-
"""
제로샷 NLI 스탠스 분석 — DL 문장별 결과를 행위자(미국/이란/이스라엘)별로 펼쳐 kornli 모델로 우호/비판/중립전달 판정
reads : pipeline_py/딥러닝_논조분석_산출물/DL_문장별_논조분석.csv (dl_tone 출력 — 반드시 dl_tone 먼저 실행)
writes: pipeline_py/제로샷NLI_스탠스분석_산출물/ (NLI_문장별_스탠스분석.csv 등 csv4 + png1 + 요약 md)
주의  : HF 모델 pongjin/roberta_with_kornli 선다운로드 필수(local_files_only)
"""
import random
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


# 재현성 고정 — 시드 42를 random/numpy/torch 전부에 적용, CPU 스레드 4로 고정 (머신 달라도 동일 결과)
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(4)

BASE_DIR = Path(__file__).resolve().parent
import sys as _sys; _sys.path.insert(0, str(BASE_DIR)); import config as _cfg  # [PIPELINE patch] 중앙 경로

DL_OUT = _cfg.DL_DIR  # [patch] dl_tone OUT_DIR과 동일(불변식)
OUT_DIR = _cfg.NLI_DIR  # [patch]
OUT_DIR.mkdir(exist_ok=True)

# 한국어 NLI(자연어추론) 모델 — '전제 문장이 가설을 함의하는가'의 entailment 확률을 스탠스 신호로 씀
MODEL_NAME = "pongjin/roberta_with_kornli"
BATCH_SIZE = 24
MAX_LENGTH = 256

# 제로샷 스탠스 가설 3종 — 기사 문장(전제)과 짝지어 모델에 넣고, 셋 중 entailment 확률이 가장 높은 게 판정
# 라벨링 데이터 없이 가설 문장만으로 분류하는 zero-shot 방식
HYPOTHESES = {
    "우호": "이 문장은 {actor}에 우호적인 관점을 담고 있다.",
    "비판": "이 문장은 {actor}를 비판적으로 다루고 있다.",
    "중립전달": "이 문장은 {actor}에 대해 단순 사실을 전달하고 있다.",
}

# 스탠스 판정 대상 행위자 — DL 단계의 ACTOR 라벨 중 이 3개만 사용
TARGET_ACTORS = ["미국", "이란", "이스라엘"]


# 한글 파일명 NFC 정규화 — Drive 동기화로 NFD가 된 파일명도 매칭되게
def nfc(text: str) -> str:
    return unicodedata.normalize("NFC", str(text))


# DL 문장별 결과를 읽어 (문장, 행위자) 쌍 테이블로 변환 — NLI의 분석 단위
def load_actor_sentence_pairs() -> pd.DataFrame:
    # 불변식: DL_OUT엔 '문장별+논조분석' csv가 dl_tone이 쓴 1개만 있어야 함 — 2개면 next()가 임의 선택해 조용히 오작동
    # (run_all이 DL_DIR을 통째로 비우고 시작하는 이유. dl_tone 출력 파일명 바꾸면 여기도 같이 봐야 함)
    sent_csv = next(p for p in DL_OUT.glob("*.csv") if "문장별" in nfc(p.name) and "논조분석" in nfc(p.name))
    df = pd.read_csv(sent_csv)
    # 행위자 없는 문장(의제만 매칭된 문장)은 스탠스 판정 대상이 아니므로 제외
    df = df[df["actors"].notna()].copy()
    # '미국|이란' 복수 행위자 문장은 행위자별 행으로 분해 — 미국 대상 1건 + 이란 대상 1건으로 각각 판정
    df["original_actors"] = df["actors"]
    df["actors"] = df["actors"].str.split("|")
    df = df.explode("actors").reset_index(drop=True)
    df = df[df["actors"].isin(TARGET_ACTORS)].copy()
    df = df.drop_duplicates(["article_id", "sentence", "actors"]).reset_index(drop=True)
    return df


# 모델 출력에서 entailment 라벨이 몇 번째인지 찾음 — 모델마다 라벨 순서가 달라 하드코딩하면 위험
def entailment_index(model) -> int:
    id2label = {int(k): v.lower() for k, v in model.config.id2label.items()}
    for idx, label in id2label.items():
        if "entail" in label:
            return idx
    raise ValueError(f"entailment label을 찾지 못했습니다: {model.config.id2label}")


# (문장, 행위자) 쌍마다 가설 3개를 모델에 넣어 entailment 확률을 얻고 스탠스 판정
def score_pairs(df: pd.DataFrame) -> pd.DataFrame:
    # local_files_only=True — 로컬 HF 캐시 모델 고정(선다운로드 필수). 빼면 네트워크/버전 차이로 결과 흔들림
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, local_files_only=True)
    model.eval()
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # [GPU patch] 원본은 CPU — 부동소수 미세차는 §9 허용오차
    model.to(DEVICE)
    ent_idx = entailment_index(model)

    # 쌍 하나당 (전제=기사 문장, 가설=스탠스 문장) 3개 생성 — 행위자 이름을 가설 템플릿에 끼움
    pairs = []
    meta = []
    for row_idx, row in df.iterrows():
        actor = row["actors"]
        for stance, template in HYPOTHESES.items():
            pairs.append((row["sentence"], template.format(actor=actor)))
            meta.append((row_idx, stance))

    # 배치 단위 추론 — softmax 후 entailment 열만 뽑아 확률로 사용
    scores = []
    total = len(pairs)
    with torch.inference_mode():
        for start in range(0, total, BATCH_SIZE):
            batch = pairs[start : start + BATCH_SIZE]
            premises = [x[0] for x in batch]
            hypotheses = [x[1] for x in batch]
            encoded = tokenizer(
                premises,
                hypotheses,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
                return_token_type_ids=False,  # [compat patch] transformers5 slow tokenizer가 쌍에 0/1 부여 → type_vocab_size=1 모델서 크래시. 팀 환경(fast tokenizer)은 전부 0 — 모델이 내부 생성하는 zeros와 동일
            )
            encoded = {k: v.to(DEVICE) for k, v in encoded.items()}  # [GPU patch]
            logits = model(**encoded).logits
            probs = torch.softmax(logits, dim=-1)[:, ent_idx].cpu().numpy()
            scores.extend(probs.tolist())
            if start == 0 or (start // BATCH_SIZE) % 50 == 0:
                print(f"NLI progress: {min(start + BATCH_SIZE, total):,}/{total:,}")

    # 세로(쌍×가설 3행)를 가로(우호/비판/중립전달 3컬럼)로 피벗해 원본 쌍 테이블에 결합
    score_df = pd.DataFrame(meta, columns=["row_idx", "stance_label"])
    score_df["entailment_prob"] = scores
    wide = score_df.pivot(index="row_idx", columns="stance_label", values="entailment_prob").reset_index()

    # stance_pred = 최고 확률 가설, stance_score = P(우호)-P(비판) — 0보다 크면 우호 쪽
    result = df.reset_index().rename(columns={"index": "row_idx"}).merge(wide, on="row_idx", how="left")
    result["stance_pred"] = result[["우호", "비판", "중립전달"]].idxmax(axis=1)
    result["stance_confidence"] = result[["우호", "비판", "중립전달"]].max(axis=1)
    result["stance_score"] = result["우호"] - result["비판"]
    return result


# (미디어그룹|언론사) × 행위자별 집계 — 쌍 수, 평균 스탠스 점수, 우호/비판/중립전달 비중
def summarize(result: pd.DataFrame, group_col: str) -> pd.DataFrame:
    return (
        result.groupby([group_col, "actors"])
        .agg(
            n_sentences=("sentence", "count"),
            stance_score_mean=("stance_score", "mean"),
            friendly_share=("stance_pred", lambda x: (x == "우호").mean()),
            critical_share=("stance_pred", lambda x: (x == "비판").mean()),
            neutral_report_share=("stance_pred", lambda x: (x == "중립전달").mean()),
            confidence_mean=("stance_confidence", "mean"),
        )
        .reset_index()
        .sort_values([group_col, "actors"])
    )


# 발표용 히트맵 1장(미디어그룹 × 행위자 스탠스) + 스탠스 비중표 CSV
def save_charts(group_summary: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    # [fix] 원본은 AppleGothic(맥 전용) — Linux/WSL서 한글 □□□ 깨짐. 번들 NanumGothic 등록(PNG 전용, 데이터 무관)
    import matplotlib.font_manager as _fm
    _fm.fontManager.addfont(str(_cfg.FONT_PATH))
    plt.rc("font", family="NanumGothic")
    plt.rcParams["axes.unicode_minus"] = False

    group_order = ["경제", "지상파", "통신·보도", "정치색"]
    piv = group_summary.pivot(index="media_group", columns="actors", values="stance_score_mean").reindex(group_order)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    sns.heatmap(piv, annot=True, fmt=".3f", center=0, cmap="RdYlGn", linewidths=0.5, ax=ax)
    ax.set_title("Zero-shot NLI: 미디어그룹 × 행위자 스탠스 근사")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "NLI_미디어그룹_행위자_스탠스.png", dpi=180)
    plt.close(fig)

    share = (
        group_summary.set_index(["media_group", "actors"])[["friendly_share", "critical_share", "neutral_report_share"]]
        .rename(columns={"friendly_share": "우호", "critical_share": "비판", "neutral_report_share": "중립전달"})
    )
    share.to_csv(OUT_DIR / "NLI_스탠스_비중표.csv", encoding="utf-8-sig")


# 요약 md 자동 생성 — 분석 쌍 수·스탠스 점수 표·해석 기준·한계를 실제 결과값으로 채움
def save_report(result: pd.DataFrame, group_summary: pd.DataFrame) -> None:
    piv = group_summary.pivot(index="media_group", columns="actors", values="stance_score_mean").round(3)
    lines = [
        "# Zero-shot NLI 기반 행위자별 스탠스 근사 분석",
        "",
        f"- 모델: `{MODEL_NAME}`",
        "- 방식: 문장과 스탠스 가설을 NLI 모델에 입력하고, entailment 확률로 우호/비판/중립전달을 비교",
        f"- 분석 대상: 미국/이란/이스라엘이 언급된 문장을 행위자별로 펼친 문장-행위자 쌍 {len(result):,}개",
        "- 점수 정의: `stance_score = P(우호 가설 entailment) - P(비판 가설 entailment)`",
        "- 복수 행위자 문장은 `미국|이란` → 미국 대상 1건 + 이란 대상 1건처럼 분해해 타깃별 가설을 비교",
        "",
        "## 미디어그룹 × 행위자 스탠스 점수",
        "",
        "|" + "|".join(["media_group"] + list(piv.columns)) + "|",
        "|" + "|".join(["---"] * (len(piv.columns) + 1)) + "|",
    ]
    for idx, row in piv.iterrows():
        lines.append("|" + "|".join([str(idx)] + [f"{v:.3f}" for v in row]) + "|")
    lines += [
        "",
        "## 해석 기준",
        "",
        "- 0보다 크면 해당 행위자에 대한 우호 가설이 비판 가설보다 강하게 지지됨",
        "- 0보다 작으면 비판 가설이 더 강하게 지지됨",
        "- 뉴스 문장은 인용과 사실 전달이 많으므로, 언론사의 실제 입장이라기보다 보도 문장에 나타난 스탠스 신호로 해석해야 함",
        "",
        "## 한계",
        "",
        "- Zero-shot NLI는 별도 라벨링 없이 스탠스를 근사하므로, 도메인 특화 지도학습 모델보다 불안정할 수 있음",
        "- 인용문과 사실 전달문을 언론사의 입장으로 오해하지 않도록 `중립전달` 가설을 함께 둠",
        "- 한 문장에 여러 행위자가 같이 나오면 타깃 귀속이 모호할 수 있어, 문장-행위자 쌍 단위 결과는 대표 문장 검토와 함께 해석해야 함",
    ]
    (OUT_DIR / "제로샷NLI_스탠스분석_요약.md").write_text("\n".join(lines), encoding="utf-8")


# 실행 흐름: DL 결과 로드 → 쌍 생성 → NLI 추론 → 쌍별 CSV 저장 → 그룹/언론사 요약 → 차트·md
def main() -> None:
    df = load_actor_sentence_pairs()
    print(f"actor-sentence pairs: {len(df):,}")
    result = score_pairs(df)
    result.to_csv(OUT_DIR / "NLI_문장별_스탠스분석.csv", index=False, encoding="utf-8-sig")

    group_summary = summarize(result, "media_group")
    press_summary = summarize(result, "press")
    group_summary.to_csv(OUT_DIR / "NLI_미디어그룹_행위자_요약.csv", index=False, encoding="utf-8-sig")
    press_summary.to_csv(OUT_DIR / "NLI_언론사별_행위자_요약.csv", index=False, encoding="utf-8-sig")

    save_charts(group_summary)
    save_report(result, group_summary)

    print("\n미디어그룹 요약")
    print(group_summary.to_string(index=False))
    print(f"\nsaved: {OUT_DIR}")


if __name__ == "__main__":
    main()
