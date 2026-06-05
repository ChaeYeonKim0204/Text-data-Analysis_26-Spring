import json
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(4)

BASE_DIR = Path(__file__).resolve().parent
import sys as _sys; _sys.path.insert(0, str(BASE_DIR)); import config as _cfg  # [PIPELINE patch] 중앙 경로

DATA_PATH = _cfg.DATA_DIR / "전처리_본문_언론사_260505_260511.csv"  # [patch] 디렉토리만 교체
OUT_DIR = _cfg.DL_DIR  # [patch]
OUT_DIR.mkdir(exist_ok=True)

MODEL_NAME = "snunlp/KR-FinBert-SC"
ISSUE_PATTERN = re.compile(r"호르무즈|이란")
SENT_SPLIT = re.compile(r"(?<=[.!?。！？])\s+|(?<=[다요죠음])\.(?=\s)|\n+")

ACTOR = {
    "미국": ["미국", "워싱턴", "트럼프", "백악관", "미군", "펜타곤", "국무부"],
    "이란": ["이란", "테헤란", "혁명수비대", "이란군", "이란산"],
    "이스라엘": ["이스라엘", "네타냐후"],
}

ASPECT = {
    "원유·에너지": ["원유", "유가", "기름", "석유", "에너지", "정유", "휘발유"],
    "봉쇄·항로": ["봉쇄", "호르무즈", "해협", "항로", "통항", "차단", "선박"],
    "군사·충돌": ["공격", "교전", "미사일", "폭격", "전쟁", "무력", "타격", "보복", "도발", "군사"],
    "외교·제재": ["협상", "외교", "회담", "제재", "핵", "합의", "대화", "중재"],
}


def split_sentences(text: str) -> list[str]:
    if not isinstance(text, str):
        return []
    text = re.sub(r"\s+", " ", text).strip()
    parts = SENT_SPLIT.split(text)
    sentences = []
    for part in parts:
        part = part.strip()
        if 25 <= len(part) <= 350:
            sentences.append(part)
    return sentences


def find_labels(sentence: str, lexicon: dict[str, list[str]]) -> list[str]:
    return [label for label, words in lexicon.items() if any(word in sentence for word in words)]


def load_issue_sentences() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    df = df[~df["is_weather"] & ~df["is_closing"] & ~df["is_within_press_dup"]].copy()
    issue = df[df["text"].fillna("").str.contains(ISSUE_PATTERN)].copy()

    rows = []
    for _, row in issue.iterrows():
        for sent in split_sentences(row["text"]):
            actors = find_labels(sent, ACTOR)
            aspects = find_labels(sent, ASPECT)
            if not actors and not aspects:
                continue
            rows.append(
                {
                    "article_id": row["article_id"],
                    "press": row["press"],
                    "media_group": row["media_group"],
                    "date": row["date"],
                    "title_cleaned": row["title_cleaned"],
                    "sentence": sent,
                    "actors": "|".join(actors),
                    "aspects": "|".join(aspects),
                    "n_actors": len(actors),
                    "n_aspects": len(aspects),
                }
            )

    sent_df = pd.DataFrame(rows).drop_duplicates(["article_id", "sentence"]).reset_index(drop=True)
    return issue, sent_df


def run_deep_learning_sentiment(sent_df: pd.DataFrame) -> pd.DataFrame:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, local_files_only=True)
    clf = pipeline("text-classification", model=model, tokenizer=tokenizer, device=-1, top_k=None)

    sentences = sent_df["sentence"].tolist()
    outputs = clf(sentences, batch_size=32, truncation=True, max_length=256)

    rows = []
    for output in outputs:
        score_map = {item["label"]: float(item["score"]) for item in output}
        rows.append(
            {
                "prob_negative": score_map.get("negative", 0.0),
                "prob_neutral": score_map.get("neutral", 0.0),
                "prob_positive": score_map.get("positive", 0.0),
                "dl_sentiment": max(score_map, key=score_map.get),
                "dl_score": score_map.get("positive", 0.0) - score_map.get("negative", 0.0),
            }
        )
    return pd.concat([sent_df, pd.DataFrame(rows)], axis=1)


def explode_summary(df: pd.DataFrame, column: str, label_name: str, exclusive_actor_only: bool = False) -> pd.DataFrame:
    work = df[df[column].fillna("").ne("")].copy()
    if exclusive_actor_only:
        work = work[work["n_actors"] == 1].copy()
    work[label_name] = work[column].str.split("|")
    work = work.explode(label_name)
    return work


def summarize(work: pd.DataFrame, index_col: str, target_col: str) -> pd.DataFrame:
    return (
        work.groupby([index_col, target_col])
        .agg(
            n_sentences=("sentence", "count"),
            dl_score_mean=("dl_score", "mean"),
            negative_share=("dl_sentiment", lambda x: (x == "negative").mean()),
            neutral_share=("dl_sentiment", lambda x: (x == "neutral").mean()),
            positive_share=("dl_sentiment", lambda x: (x == "positive").mean()),
        )
        .reset_index()
        .sort_values([index_col, "dl_score_mean"], ascending=[True, False])
    )


def save_charts(actor_summary: pd.DataFrame, aspect_summary: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    try:
        plt.rc("font", family="AppleGothic")
    except Exception:
        pass
    plt.rcParams["axes.unicode_minus"] = False

    group_order = ["경제", "지상파", "통신·보도", "정치색"]

    actor_piv = actor_summary.pivot(index="media_group", columns="actor", values="dl_score_mean").reindex(group_order)
    plt.figure(figsize=(8, 4.8))
    sns.heatmap(actor_piv, annot=True, fmt=".3f", center=0, cmap="RdYlGn", linewidths=0.5)
    plt.title("딥러닝 감성 모델: 미디어그룹 × 행위자 논조")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "DL_미디어그룹_행위자_논조.png", dpi=180)
    plt.close()

    aspect_piv = aspect_summary.pivot(index="media_group", columns="aspect", values="dl_score_mean").reindex(group_order)
    plt.figure(figsize=(9, 4.8))
    sns.heatmap(aspect_piv, annot=True, fmt=".3f", center=0, cmap="RdYlGn", linewidths=0.5)
    plt.title("딥러닝 감성 모델: 미디어그룹 × 의제 논조")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "DL_미디어그룹_의제_논조.png", dpi=180)
    plt.close()


def save_report(issue: pd.DataFrame, sent_scored: pd.DataFrame, actor_summary: pd.DataFrame, aspect_summary: pd.DataFrame) -> None:
    actor_table = actor_summary.pivot(index="media_group", columns="actor", values="dl_score_mean").round(3)
    aspect_table = aspect_summary.pivot(index="media_group", columns="aspect", values="dl_score_mean").round(3)

    lines = [
        "# 딥러닝 기반 언론사별 논조 분석 고도화",
        "",
        f"- 모델: `{MODEL_NAME}` (`negative / neutral / positive` 3분류 Sequence Classification)",
        f"- 재현성: seed={SEED}, 로컬 캐시 모델 사용, 유료 LLM/API 미사용",
        f"- 분석 대상: 비기사 필터 후 전체 기사 중 `호르무즈|이란` 포함 기사 {len(issue):,}건",
        f"- 문장 단위 분석: 행위자/의제 키워드가 포함된 문장 {len(sent_scored):,}개",
        "- 점수 정의: `dl_score = P(positive) - P(negative)`, 0보다 크면 긍정 톤, 작으면 부정 톤",
        "",
        "## 미디어그룹 × 행위자 논조",
        "",
        actor_table.to_markdown(),
        "",
        "## 미디어그룹 × 의제 논조",
        "",
        aspect_table.to_markdown(),
        "",
        "## 발표용 해석",
        "",
        "기존 KNU 감성사전 기반 ABSA는 특정 대상이 등장한 문장의 주변 어휘를 사전 점수로 환산했다. 이번 고도화는 같은 문장 단위를 딥러닝 감성 분류 모델에 직접 입력해 긍정/중립/부정 확률을 얻었다.",
        "",
        "따라서 발표에서는 `사전 기반 ABSA로 1차 패턴을 잡고, 딥러닝 문장 분류로 같은 패턴을 재검증했다`고 설명하면 된다. 단, 이 모델은 문장 전체 감성 분류기이므로 엄밀한 의미의 stance detection은 아니며, 한 문장에 여러 행위자가 동시에 나오면 감성이 특정 행위자 하나에만 귀속된다고 보기 어렵다는 한계는 함께 명시한다.",
        "",
        "## 산출 파일",
        "",
        "- `DL_문장별_논조분석.csv`: 문장별 딥러닝 감성 확률과 점수",
        "- `DL_행위자_요약.csv`: 미디어그룹/행위자별 요약",
        "- `DL_의제_요약.csv`: 미디어그룹/의제별 요약",
        "- `DL_미디어그룹_행위자_논조.png`, `DL_미디어그룹_의제_논조.png`: 발표용 히트맵",
    ]
    (OUT_DIR / "딥러닝_논조분석_요약.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    issue, sent_df = load_issue_sentences()
    print(f"issue articles: {len(issue):,}")
    print(f"target/aspect sentences: {len(sent_df):,}")

    sent_scored = run_deep_learning_sentiment(sent_df)
    sent_scored.to_csv(OUT_DIR / "DL_문장별_논조분석.csv", index=False, encoding="utf-8-sig")

    actor_work = explode_summary(sent_scored, "actors", "actor", exclusive_actor_only=True)
    aspect_work = explode_summary(sent_scored, "aspects", "aspect")

    actor_summary = summarize(actor_work, "media_group", "actor")
    aspect_summary = summarize(aspect_work, "media_group", "aspect")
    press_actor_summary = summarize(actor_work, "press", "actor")

    actor_summary.to_csv(OUT_DIR / "DL_행위자_요약.csv", index=False, encoding="utf-8-sig")
    aspect_summary.to_csv(OUT_DIR / "DL_의제_요약.csv", index=False, encoding="utf-8-sig")
    press_actor_summary.to_csv(OUT_DIR / "DL_언론사별_행위자_요약.csv", index=False, encoding="utf-8-sig")

    save_charts(actor_summary, aspect_summary)
    save_report(issue, sent_scored, actor_summary, aspect_summary)

    print("\nactor summary")
    print(actor_summary.to_string(index=False))
    print("\naspect summary")
    print(aspect_summary.to_string(index=False))
    print(f"\nsaved: {OUT_DIR}")


if __name__ == "__main__":
    main()
