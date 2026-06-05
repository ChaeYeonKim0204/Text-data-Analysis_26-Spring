import random
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


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

MODEL_NAME = "pongjin/roberta_with_kornli"
BATCH_SIZE = 24
MAX_LENGTH = 256

HYPOTHESES = {
    "우호": "이 문장은 {actor}에 우호적인 관점을 담고 있다.",
    "비판": "이 문장은 {actor}를 비판적으로 다루고 있다.",
    "중립전달": "이 문장은 {actor}에 대해 단순 사실을 전달하고 있다.",
}

TARGET_ACTORS = ["미국", "이란", "이스라엘"]


def nfc(text: str) -> str:
    return unicodedata.normalize("NFC", str(text))


def load_actor_sentence_pairs() -> pd.DataFrame:
    sent_csv = next(p for p in DL_OUT.glob("*.csv") if "문장별" in nfc(p.name) and "논조분석" in nfc(p.name))
    df = pd.read_csv(sent_csv)
    df = df[df["actors"].notna()].copy()
    df["original_actors"] = df["actors"]
    df["actors"] = df["actors"].str.split("|")
    df = df.explode("actors").reset_index(drop=True)
    df = df[df["actors"].isin(TARGET_ACTORS)].copy()
    df = df.drop_duplicates(["article_id", "sentence", "actors"]).reset_index(drop=True)
    return df


def entailment_index(model) -> int:
    id2label = {int(k): v.lower() for k, v in model.config.id2label.items()}
    for idx, label in id2label.items():
        if "entail" in label:
            return idx
    raise ValueError(f"entailment label을 찾지 못했습니다: {model.config.id2label}")


def score_pairs(df: pd.DataFrame) -> pd.DataFrame:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, local_files_only=True)
    model.eval()
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # [GPU patch] 원본은 CPU — 부동소수 미세차는 §9 허용오차
    model.to(DEVICE)
    ent_idx = entailment_index(model)

    pairs = []
    meta = []
    for row_idx, row in df.iterrows():
        actor = row["actors"]
        for stance, template in HYPOTHESES.items():
            pairs.append((row["sentence"], template.format(actor=actor)))
            meta.append((row_idx, stance))

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

    score_df = pd.DataFrame(meta, columns=["row_idx", "stance_label"])
    score_df["entailment_prob"] = scores
    wide = score_df.pivot(index="row_idx", columns="stance_label", values="entailment_prob").reset_index()

    result = df.reset_index().rename(columns={"index": "row_idx"}).merge(wide, on="row_idx", how="left")
    result["stance_pred"] = result[["우호", "비판", "중립전달"]].idxmax(axis=1)
    result["stance_confidence"] = result[["우호", "비판", "중립전달"]].max(axis=1)
    result["stance_score"] = result["우호"] - result["비판"]
    return result


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


def save_charts(group_summary: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.rc("font", family="AppleGothic")
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
