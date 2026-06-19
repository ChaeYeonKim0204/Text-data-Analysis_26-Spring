# -*- coding: utf-8 -*-
"""
DL 문장별 결과에서 미국, 이란, 이스라엘 문장 선별, 우호/비판/중립전달 분류
입력: pipeline_py/딥러닝_논조분석_산출물/DL_문장별_논조분석.csv 사용, dl_tone 선행 필요
출력: pipeline_py/제로샷NLI_스탠스분석_산출물/에 문장별 CSV, 요약 CSV, 그래프, 요약 md 저장
주의: 모델 파일 사전 준비 필요
"""
import random
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


# 같은 조건에서 비슷한 결과가 나오도록 숫자 생성 기준 고정
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(4)

BASE_DIR = Path(__file__).resolve().parent
import sys as _sys; _sys.path.insert(0, str(BASE_DIR)); import config as _cfg

DL_OUT = _cfg.DL_DIR  # dl_tone 결과 폴더
OUT_DIR = _cfg.NLI_DIR  # 스탠스 분석 결과 저장 폴더
OUT_DIR.mkdir(exist_ok=True)

# 문장과 가설을 비교해 어느 설명이 더 맞는지 보는 한국어 모델
MODEL_NAME = "pongjin/roberta_with_kornli"
BATCH_SIZE = 24   # 문장과 가설을 함께 넣기 때문에 한 번에 처리하는 개수를 조금 줄임
MAX_LENGTH = 256

# 기사 문장과 비교할 세 가지 가설
HYPOTHESES = {
    "우호": "이 문장은 {actor}에 우호적인 관점을 담고 있다.",
    "비판": "이 문장은 {actor}를 비판적으로 다루고 있다.",
    "중립전달": "이 문장은 {actor}에 대해 단순 사실을 전달하고 있다.",
}

# 스탠스 판정 대상 행위자 — DL 단계의 ACTOR 라벨 중 이 3개만 사용
TARGET_ACTORS = ["미국", "이란", "이스라엘"]


# 한글 파일명을 비교하기 위해 이름을 한 번 정리
def nfc(text: str) -> str:
    return unicodedata.normalize("NFC", str(text))


# DL 문장별 결과를 읽어 문장과 행위자를 한 줄씩 정렬
def load_actor_sentence_pairs() -> pd.DataFrame:
    # dl_tone에서 만든 문장별 분석 파일 입력
    sent_csv = next(p for p in DL_OUT.glob("*.csv") if "문장별" in nfc(p.name) and "논조분석" in nfc(p.name))
    df = pd.read_csv(sent_csv)
    # 행위자 없는 문장(의제만 매칭된 문장)은 스탠스 판정 대상 제외
    df = df[df["actors"].notna()].copy()
    # `미국|이란` 복수 행위자 문장은 행위자별 행으로 분해 — 미국 대상 1건 + 이란 대상 1건으로 각각 판정
    df["original_actors"] = df["actors"]   # 행 분해 전 원래 행위자 표시 보관
    df["actors"] = df["actors"].str.split("|")
    df = df.explode("actors").reset_index(drop=True)
    df = df[df["actors"].isin(TARGET_ACTORS)].copy()
    # 같은 기사·같은 문장·같은 행위자 쌍은 1개만 유지
    df = df.drop_duplicates(["article_id", "sentence", "actors"]).reset_index(drop=True)
    return df


# 모델 점수 중에서 맞는 점수 칸 선택
def entailment_index(model) -> int:
    id2label = {int(k): v.lower() for k, v in model.config.id2label.items()}
    for idx, label in id2label.items():
        if "entail" in label:
            return idx
    raise ValueError(f"가설 점수 label을 찾지 못했습니다: {model.config.id2label}")


# 문장과 행위자별 우호/비판/중립전달 근접도 계산
def score_pairs(df: pd.DataFrame) -> pd.DataFrame:
    # 미리 받아 둔 모델 파일 사용
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, local_files_only=True)
    model.eval()
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(DEVICE)
    ent_idx = entailment_index(model)

    # 기사 문장 하나마다 우호/비판/중립전달 가설 3개 생성·비교
    pairs = []
    meta = []
    for row_idx, row in df.iterrows():
        actor = row["actors"]
        for stance, template in HYPOTHESES.items():
            pairs.append((row["sentence"], template.format(actor=actor)))
            meta.append((row_idx, stance))

    # 여러 문장을 묶어 모델 입력, 각 가설 점수 계산
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
                return_token_type_ids=False,
            )
            encoded = {k: v.to(DEVICE) for k, v in encoded.items()}
            logits = model(**encoded).logits
            probs = torch.softmax(logits, dim=-1)[:, ent_idx].cpu().numpy()
            scores.extend(probs.tolist())
            # 오래 걸리는 작업이라 처음과 50묶음마다 처리 건수 표시
            if start == 0 or (start // BATCH_SIZE) % 50 == 0:
                print(f"처리 건수: {min(start + BATCH_SIZE, total):,}/{total:,}")

    # 가설별 점수를 우호/비판/중립전달 컬럼으로 정리
    score_df = pd.DataFrame(meta, columns=["row_idx", "stance_label"])
    score_df["entailment_prob"] = scores
    wide = score_df.pivot(index="row_idx", columns="stance_label", values="entailment_prob").reset_index()

    # stance_pred = 점수가 가장 높은 가설, stance_score = 우호 점수 - 비판 점수
    # 0보다 크면 우호 쪽, 0보다 작으면 비판 쪽 해석
    result = df.reset_index().rename(columns={"index": "row_idx"}).merge(wide, on="row_idx", how="left")
    result["stance_pred"] = result[["우호", "비판", "중립전달"]].idxmax(axis=1)
    result["stance_confidence"] = result[["우호", "비판", "중립전달"]].max(axis=1)
    result["stance_score"] = result["우호"] - result["비판"]
    return result


# 미디어그룹이나 언론사별 행위자 문장 수와 평균 점수 요약
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


# 발표에 쓸 미디어그룹별 스탠스 그래프와 스탠스 비중표 저장
def save_charts(group_summary: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    # 그래프 한글 표시용 폰트
    import matplotlib.font_manager as _fm
    _fm.fontManager.addfont(str(_cfg.FONT_PATH))
    plt.rc("font", family="NanumGothic")
    plt.rcParams["axes.unicode_minus"] = False

    group_order = ["경제", "지상파", "통신·보도", "정치색"]
    piv = group_summary.pivot(index="media_group", columns="actors", values="stance_score_mean").reindex(group_order)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    sns.heatmap(piv, annot=True, fmt=".3f", center=0, cmap="RdYlGn", linewidths=0.5, ax=ax)
    ax.set_title("Zero-shot NLI: 미디어그룹 × 행위자 스탠스")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "NLI_미디어그룹_행위자_스탠스.png", dpi=180)
    plt.close(fig)

    share = (
        group_summary.set_index(["media_group", "actors"])[["friendly_share", "critical_share", "neutral_report_share"]]
        .rename(columns={"friendly_share": "우호", "critical_share": "비판", "neutral_report_share": "중립전달"})
    )
    share.to_csv(OUT_DIR / "NLI_스탠스_비중표.csv", encoding="utf-8-sig")


# 분석 결과를 요약 파일로 저장
def save_report(result: pd.DataFrame, group_summary: pd.DataFrame) -> None:
    piv = group_summary.pivot(index="media_group", columns="actors", values="stance_score_mean").round(3)
    lines = [
        "# Zero-shot NLI 기반 행위자별 스탠스 분석",
        "",
        f"- 모델: `{MODEL_NAME}`",
        "- 방식: 문장과 스탠스 가설 3개를 모델에 넣고, 점수가 높은 쪽으로 판정",
        f"- 분석 대상: 미국/이란/이스라엘이 언급된 문장을 행위자별로 펼친 문장-행위자 쌍 {len(result):,}개",
        "- 점수 정의: `stance_score = 우호 가설 점수 - 비판 가설 점수`",
        "- 복수 행위자 문장은 `미국|이란` → 미국 대상 1건 + 이란 대상 1건처럼 나누어 비교",
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
        "## 점수 보는 방법",
        "",
        "- 0보다 크면 그 행위자에 대해 우호 쪽 설명이 더 맞게 나온 것임",
        "- 0보다 작으면 비판 쪽 설명이 더 맞게 나온 것임",
        "- 뉴스에는 인용문과 사실 전달문이 많아서, 이 점수를 언론사의 실제 입장이라고 바로 보면 안 됨",
        "",
        "## 주의할 점",
        "",
        "- 사람이 직접 정답을 붙인 데이터로 학습한 것은 아니라 결과가 흔들릴 수 있음",
        "- 인용문과 사실 전달문을 언론사의 입장으로 착각하지 않도록 `중립전달`도 함께 사용",
        "- 한 문장에 여러 행위자가 같이 나오면 어느 대상에 대한 표현인지 애매할 수 있어, 대표 문장 확인과 함께 해석",
    ]
    (OUT_DIR / "제로샷NLI_스탠스분석_요약.md").write_text("\n".join(lines), encoding="utf-8")


# 실행 흐름: DL 결과 불러오기 → 행위자별 문장 만들기 → 모델 분석 → CSV·차트 저장
def main() -> None:
    df = load_actor_sentence_pairs()
    print(f"행위자별 문장 수: {len(df):,}")
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
    print(f"\n저장 폴더: {OUT_DIR}")


if __name__ == "__main__":
    main()
