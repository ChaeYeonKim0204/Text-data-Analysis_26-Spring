# -*- coding: utf-8 -*-
"""
딥러닝 논조 분석. 호르무즈나 이란이 들어간 기사 문장 추출, KR-FinBert-SC로 긍정/중립/부정 분류
읽는 파일: data/news/전처리_본문_언론사_260505_260511.csv
저장 파일: pipeline_py/딥러닝_논조분석_산출물/ (CSV 4개, PNG 2개, 요약 md 저장)
주의: 모델 파일 사전 준비 필요. 다음 단계 nli_stance가 이 결과 사용
"""
import json
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline


# 같은 조건에서 비슷한 결과가 나오도록 숫자 생성 기준 고정
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(4)

BASE_DIR = Path(__file__).resolve().parent
import sys as _sys; _sys.path.insert(0, str(BASE_DIR)); import config as _cfg

DATA_PATH = _cfg.DATA_DIR / "전처리_본문_언론사_260505_260511.csv"  # 전처리된 기사 파일
OUT_DIR = _cfg.DL_DIR  # 딥러닝 분석 결과 저장 폴더
OUT_DIR.mkdir(exist_ok=True)

# 문장을 부정/중립/긍정으로 나누는 한국어 감성 모델
MODEL_NAME = "snunlp/KR-FinBert-SC"
# 호르무즈 의제 기사를 고를 때 쓰는 키워드 — 한글 매칭이라 외국어 전용 기사는 자연히 안 걸림
ISSUE_PATTERN = re.compile(r"호르무즈|이란")
# 문장 끝 부호나 종결어미를 기준으로 본문을 문장 단위로 나눔
SENT_SPLIT = re.compile(r"(?<=[.!?。！？])\s+|(?<=[다요죠음])\.(?=\s)|\n+")

# 행위자/의제 단어 목록. 문장에 아래 단어가 들어 있으면 해당 이름 부여
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


# 기사 본문 문장 단위 분리 — 25자보다 짧은 문장과 350자보다 긴 문장은 분석 제외
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


# 문장에 사전 단어가 하나라도 들어있는 라벨 반환 — ACTOR/ASPECT 공용
def find_labels(sentence: str, lexicon: dict[str, list[str]]) -> list[str]:
    return [label for label, words in lexicon.items() if any(word in sentence for word in words)]


# 전처리본에서 호르무즈 의제 기사 선택, 분석 대상 문장 테이블 생성
def load_issue_sentences() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    # 비기사(날씨/클로징)·매체 내 중복·외국어 전용 제외 — 교차 매체 중복은 언론사 비교용 유지
    df = df[~df["is_weather"] & ~df["is_closing"] & ~df["is_within_press_dup"] & ~df["is_foreign"]].copy()
    issue = df[df["text"].fillna("").str.contains(ISSUE_PATTERN)].copy()

    # 기사를 문장으로 나누고, 행위자나 의제 단어가 없는 문장은 제외 — 논조 확인 가능한 문장만 유지
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
                    "actors": "|".join(actors),    # 여러 행위자는 한 칸에 묶어 저장
                    "aspects": "|".join(aspects),  # 여러 의제도 같은 방식으로 저장
                    "n_actors": len(actors),
                    "n_aspects": len(aspects),
                }
            )

    # 같은 기사 안에서 같은 문장이 중복 추출되면 1개만 — (기사, 문장) 쌍이 분석 단위
    sent_df = pd.DataFrame(rows).drop_duplicates(["article_id", "sentence"]).reset_index(drop=True)
    return issue, sent_df


# 문장 전체를 모델에 넣어 감성 확률 3개(neg/neu/pos) 획득, 점수 컬럼 부착
def run_deep_learning_sentiment(sent_df: pd.DataFrame) -> pd.DataFrame:
    # 미리 받아 둔 모델 파일 사용
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, local_files_only=True)
    DEVICE = 0 if torch.cuda.is_available() else -1
    # 부정/중립/긍정 확률 모두 수집
    clf = pipeline("text-classification", model=model, tokenizer=tokenizer, device=DEVICE, top_k=None)

    # 문장을 한 번에 32개씩 모델 입력
    sentences = sent_df["sentence"].tolist()
    outputs = clf(sentences, batch_size=32, truncation=True, max_length=256)

    # 긍정 확률에서 부정 확률을 뺀 점수 계산
    # 0보다 크면 긍정 쪽, 0보다 작으면 부정 쪽 해석
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


# `미국|이란`처럼 묶인 값을 행으로 분해 — 행위자별/의제별 분리 집계용
# 옵션을 켜면 행위자가 하나만 나온 문장만 사용
def explode_summary(df: pd.DataFrame, column: str, label_name: str, exclusive_actor_only: bool = False) -> pd.DataFrame:
    work = df[df[column].fillna("").ne("")].copy()
    if exclusive_actor_only:
        work = work[work["n_actors"] == 1].copy()
    work[label_name] = work[column].str.split("|")
    work = work.explode(label_name)
    return work


# 미디어그룹 또는 언론사별 행위자/의제 점수 요약 — 문장 수, 평균 점수, 부정/중립/긍정 비율 계산
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


# 발표용 히트맵 2장 — 미디어그룹 × 행위자 / 미디어그룹 × 의제 (값 = dl_score 평균)
def save_charts(actor_summary: pd.DataFrame, aspect_summary: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    # 그래프 한글 표시용 폰트
    import matplotlib.font_manager as _fm
    _fm.fontManager.addfont(str(_cfg.FONT_PATH))
    plt.rc("font", family="NanumGothic")
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


# 분석 건수와 점수표를 요약 파일로 저장
def save_report(issue: pd.DataFrame, sent_scored: pd.DataFrame, actor_summary: pd.DataFrame, aspect_summary: pd.DataFrame) -> None:
    actor_table = actor_summary.pivot(index="media_group", columns="actor", values="dl_score_mean").round(3)
    aspect_table = aspect_summary.pivot(index="media_group", columns="aspect", values="dl_score_mean").round(3)

    lines = [
        "# 딥러닝 기반 언론사별 논조 분석",
        "",
        f"- 모델: `{MODEL_NAME}` (부정/중립/긍정 3분류)",
        f"- 실행 설정: seed={SEED}, 미리 받아 둔 모델 사용",
        f"- 분석 대상: 비기사 제외 뒤 전체 기사 중 `호르무즈|이란` 포함 기사 {len(issue):,}건",
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
        "## 해석 참고",
        "",
        "앞에서는 KNU 감성사전으로 단어 점수를 더해서 봤고, 여기서는 같은 문장을 딥러닝 감성 모델에 넣어 긍정/중립/부정 확률을 확인함.",
        "",
        "발표에서는 `사전 방식으로 먼저 보고 딥러닝 문장 분류로 한 번 더 확인했다` 정도로 설명 가능. 다만 문장 전체 분위기를 보는 방식이라, 한 문장에 여러 나라가 같이 나오면 어느 쪽에 대한 감정인지 헷갈릴 수 있음.",
        "",
        "## 저장된 파일",
        "",
        "- `DL_문장별_논조분석.csv`: 문장별 딥러닝 감성 확률과 점수",
        "- `DL_행위자_요약.csv`: 미디어그룹/행위자별 요약",
        "- `DL_의제_요약.csv`: 미디어그룹/의제별 요약",
        "- `DL_미디어그룹_행위자_논조.png`, `DL_미디어그룹_의제_논조.png`: 발표용 히트맵",
    ]
    (OUT_DIR / "딥러닝_논조분석_요약.md").write_text("\n".join(lines), encoding="utf-8")


# 실행 흐름: 문장 추출 → 모델 분석 → CSV 저장 → 요약표·차트 저장
def main() -> None:
    issue, sent_df = load_issue_sentences()
    print(f"호르무즈/이란 관련 기사 수: {len(issue):,}")
    print(f"분석에 쓸 문장 수: {len(sent_df):,}")

    sent_scored = run_deep_learning_sentiment(sent_df)
    sent_scored.to_csv(OUT_DIR / "DL_문장별_논조분석.csv", index=False, encoding="utf-8-sig")

    # 행위자 요약은 행위자가 하나만 나온 문장만 사용, 의제 요약은 전체 사용
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

    print("\n행위자별 요약")
    print(actor_summary.to_string(index=False))
    print("\n의제별 요약")
    print(aspect_summary.to_string(index=False))
    print(f"\n저장 폴더: {OUT_DIR}")


if __name__ == "__main__":
    main()
