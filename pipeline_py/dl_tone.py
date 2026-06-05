# -*- coding: utf-8 -*-
"""
딥러닝 논조 분석 — 호르무즈|이란 기사에서 행위자/의제 문장을 뽑아 KR-FinBert-SC로 긍/중/부 분류
reads : data/news/전처리_본문_언론사_260505_260511.csv
writes: pipeline_py/딥러닝_논조분석_산출물/ (DL_문장별_논조분석.csv 등 csv4 + png2 + 요약 md)
주의  : HF 모델 snunlp/KR-FinBert-SC 선다운로드 필수(local_files_only) — 다음 단계 nli_stance가 이 출력 csv를 읽음
"""
import json
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline


# 재현성 설정 — 시드 42를 random/numpy/torch에 적용, CPU 연산 스레드 4로 고정 (GPU 사용 시 부동소수 미세차 가능)
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(4)

BASE_DIR = Path(__file__).resolve().parent
import sys as _sys; _sys.path.insert(0, str(BASE_DIR)); import config as _cfg  # [PIPELINE patch] 중앙 경로

DATA_PATH = _cfg.DATA_DIR / "전처리_본문_언론사_260505_260511.csv"  # [patch] 전처리 입력 — config 중앙 경로로 run_all과 동일 파일 사용
OUT_DIR = _cfg.DL_DIR  # [patch] DL 산출물 폴더 — 후속 nli_stance가 같은 위치를 읽음
OUT_DIR.mkdir(exist_ok=True)

# 한국어 금융 감성 분류 모델 — 문장을 negative/neutral/positive 3분류
MODEL_NAME = "snunlp/KR-FinBert-SC"
# 호르무즈 의제 기사 선별용 키워드 — 한글 매칭이라 외국어 전용 기사는 자연히 안 걸림
ISSUE_PATTERN = re.compile(r"호르무즈|이란")
# 문장 분리 정규식 — 종결부호 뒤 공백에서 자르고, 한국어 종결어미(다/요/죠/음) 뒤 마침표는 마침표 자체를 구분자로 소비
# (\n+ 분기는 split_sentences가 분리 전에 공백을 뭉개므로 실제로는 안 탐 — 원본 유지용)
SENT_SPLIT = re.compile(r"(?<=[.!?。！？])\s+|(?<=[다요죠음])\.(?=\s)|\n+")

# 행위자/의제 사전 — 문장에 아래 단어가 포함되면 해당 라벨 부여 (한 문장에 여러 라벨 가능)
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


# 기사 본문을 문장 단위로 쪼갬 — 25자 미만(단편·잡음)과 350자 초과(분리 실패 덩어리)는 버림
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


# 문장에 사전 단어가 하나라도 들어있는 라벨들을 반환 — ACTOR/ASPECT 공용
def find_labels(sentence: str, lexicon: dict[str, list[str]]) -> list[str]:
    return [label for label, words in lexicon.items() if any(word in sentence for word in words)]


# 전처리본에서 호르무즈 의제 기사를 골라 분석 대상 문장 테이블 생성
def load_issue_sentences() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    # 비기사(날씨/클로징)·매체 내 중복·외국어 전용 제외 — 교차 매체 중복은 언론사 비교에 필요해서 유지
    df = df[~df["is_weather"] & ~df["is_closing"] & ~df["is_within_press_dup"] & ~df["is_foreign"]].copy()
    issue = df[df["text"].fillna("").str.contains(ISSUE_PATTERN)].copy()

    # 기사를 문장으로 쪼개고 행위자/의제 키워드가 하나도 없는 문장은 제외 — 논조 귀속 대상이 있는 문장만 남김
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

    # 같은 기사 안에서 같은 문장이 중복 추출되면 1개만 — (기사, 문장) 쌍이 분석 단위
    sent_df = pd.DataFrame(rows).drop_duplicates(["article_id", "sentence"]).reset_index(drop=True)
    return issue, sent_df


# 문장 전체를 모델에 넣어 감성 확률 3개(neg/neu/pos)를 얻고 점수 컬럼으로 붙임
def run_deep_learning_sentiment(sent_df: pd.DataFrame) -> pd.DataFrame:
    # local_files_only=True — 로컬 HF 캐시 모델 고정(선다운로드 필수), 빼면 네트워크/버전 차이로 결과 흔들림
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, local_files_only=True)
    DEVICE = 0 if torch.cuda.is_available() else -1  # [GPU patch] 원본은 device=-1(CPU) — 부동소수 미세차는 §9 허용오차
    # top_k=None — 최상위 1개가 아니라 3개 라벨 확률을 전부 받음 (dl_score 계산에 필요)
    clf = pipeline("text-classification", model=model, tokenizer=tokenizer, device=DEVICE, top_k=None)

    # 배치 32 / 최대 256토큰 — 초과분은 잘림 (문장 단위라 256이면 충분)
    sentences = sent_df["sentence"].tolist()
    outputs = clf(sentences, batch_size=32, truncation=True, max_length=256)

    # dl_sentiment = 최고 확률 라벨, dl_score = P(긍정)-P(부정) — 0보다 크면 긍정 톤
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


# '미국|이란'처럼 |로 묶인 라벨을 행으로 펼침 — 라벨별 집계 전 단계
# exclusive_actor_only=True면 행위자가 1명뿐인 문장만 사용 — 여러 행위자가 섞이면 감성 귀속이 모호해지므로
def explode_summary(df: pd.DataFrame, column: str, label_name: str, exclusive_actor_only: bool = False) -> pd.DataFrame:
    work = df[df[column].fillna("").ne("")].copy()
    if exclusive_actor_only:
        work = work[work["n_actors"] == 1].copy()
    work[label_name] = work[column].str.split("|")
    work = work.explode(label_name)
    return work


# (미디어그룹|언론사) × (행위자|의제)별 집계 — 문장 수, 평균 점수, 긍/중/부 비중
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

    # [fix] 원본은 AppleGothic(맥 전용) — Linux/WSL서 한글 □□□ 깨짐. 번들 NanumGothic 등록(PNG 전용, 데이터 무관)
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


# 요약 md 자동 생성 — 분석 건수·점수 표·발표용 해석 문구를 실제 결과값으로 채움
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


# 실행 흐름: 문장 추출 → 모델 추론 → 문장별 CSV 저장 → 행위자/의제별 요약 → 차트·md
def main() -> None:
    issue, sent_df = load_issue_sentences()
    print(f"issue articles: {len(issue):,}")
    print(f"target/aspect sentences: {len(sent_df):,}")

    sent_scored = run_deep_learning_sentiment(sent_df)
    sent_scored.to_csv(OUT_DIR / "DL_문장별_논조분석.csv", index=False, encoding="utf-8-sig")

    # 행위자 요약은 단일 행위자 문장만(귀속 명확), 의제 요약은 전체 사용
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
