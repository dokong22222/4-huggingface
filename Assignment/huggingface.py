# @title 단계 1. 환경 설정
# 필요한 라이브러리 설치
!pip install -q "transformers<5.0" "tokenizers<0.21" datasets accelerate

# 라이브러리 임포트
from datasets import load_dataset, Dataset
from transformers import pipeline
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# GPU 사용 가능 여부 확인
device = 0 if torch.cuda.is_available() else -1
print(f"사용 가능한 디바이스: {'GPU' if device == 0 else 'CPU'}")

# @title 단계 2. 데이터셋 로드 및 필터링

# tweet_eval 데이터셋 로드 (sentiment 서브셋)
raw_dataset = load_dataset("tweet_eval", "sentiment", split="train")

# Pandas DataFrame으로 변환
df = pd.DataFrame(raw_dataset)

print("전체 데이터셋 크기:", len(df))
print("\n데이터 샘플 확인:")
display(df.head(3))
print("\n컬럼 목록:", df.columns.tolist())

# @title 단계 2-2. iPhone / Galaxy 필터링

# 키워드 필터링
iphone_mask = df['text'].str.contains('iphone|apple', case=False, na=False)
galaxy_mask = df['text'].str.contains('galaxy|samsung', case=False, na=False)

# 브랜드 라벨 추가
df_iphone = df[iphone_mask].copy()
df_iphone['brand'] = 'iPhone'

df_galaxy = df[galaxy_mask].copy()
df_galaxy['brand'] = 'Galaxy'

# 결과 확인
print(f"iPhone 관련 트윗 수: {len(df_iphone)}")
print(f"Galaxy 관련 트윗 수: {len(df_galaxy)}")

print("\n iPhone 트윗 샘플:")
print(df_iphone['text'].values[0])
print("\n Galaxy 트윗 샘플:")
print(df_galaxy['text'].values[0])

# @title 단계 2-3. 합치기 & 샘플링

# 두 브랜드 합치기
combined_df = pd.concat([df_iphone, df_galaxy]).reset_index(drop=True)

# 100개 샘플링 (브랜드 비율 유지)
sampled_df = combined_df.groupby('brand', group_keys=False).apply(
    lambda x: x.sample(min(len(x), int(100 * len(x) / len(combined_df))), random_state=42)
).reset_index(drop=True)

# label 숫자 → 텍스트로 변환
label_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
sampled_df['original_label'] = sampled_df['label'].map(label_map)

# Hugging Face Dataset으로 변환
sampled_dataset = Dataset.from_pandas(sampled_df[['text', 'brand', 'original_label']].reset_index(drop=True))

print(f"최종 샘플 수: {len(sampled_dataset)}")
print(f"\n브랜드별 샘플 수:")
print(sampled_df['brand'].value_counts())
print(f"\n데이터 샘플 확인:")
display(sampled_df[['text', 'brand', 'original_label']].head(5))

# @title 단계 3. 감정 분석 모델 파이프라인 로드

# 트위터 전용 감정 분석 모델 로드
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
    device=device
)

print("감정 분석 파이프라인 로드 완료!")

# 테스트
test_result = sentiment_pipeline("iPhone camera is absolutely amazing!")
print(f"\n테스트 결과: {test_result}")

# @title 단계 4. map 함수로 감정 분석 적용

# 감정 분석 함수 정의
def analyze_sentiment(example):
    result = sentiment_pipeline(
        example['text'],
        truncation=True,    # 트윗이 너무 길면 자동으로 잘라줌
        max_length=512
    )
    example['predicted_sentiment'] = result[0]['label']
    example['sentiment_score'] = round(result[0]['score'], 4)
    return example

print("감정 분석 시작... (잠시 기다려주세요)")

# map 함수로 전체 데이터셋에 적용
final_dataset = sampled_dataset.map(analyze_sentiment)

print("감정 분석 완료!")

# 결과 확인
final_df = pd.DataFrame(final_dataset)
print(f"\n결과 샘플:")
display(final_df[['brand', 'text', 'predicted_sentiment', 'sentiment_score']].head(5))

# @title 한글 폰트 설치 (맨 처음에 한 번만 실행)
import subprocess
subprocess.run(['apt-get', 'install', '-y', 'fonts-nanum'], capture_output=True)

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 폰트 캐시 강제 초기화
fm.fontManager.addfont('/usr/share/fonts/truetype/nanum/NanumGothic.ttf')
plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False

# 적용 확인
print("현재 폰트:", plt.rcParams['font.family'])
fig, ax = plt.subplots(figsize=(4, 2))
ax.set_title('한글 테스트 - 아이폰 갤럭시')
plt.show()

# @title 단계 5. 시각화 (한글 폰트 수정 + 비율 분석)

# 한글 폰트 설치
import subprocess
subprocess.run(['apt-get', 'install', '-y', 'fonts-nanum'], capture_output=True)

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
fm._load_fontmanager(try_read_cache=False)

# 나눔 폰트 설정
plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False

sentiment_order = ['positive', 'neutral', 'negative']
colors = {'positive': '#4CAF50', 'neutral': '#2196F3', 'negative': '#F44336'}

# ── 그래프 1~3 ──────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('iPhone vs Galaxy 트윗 감정 분석 결과', fontsize=16, fontweight='bold')

# 그래프 1. 브랜드별 감정 분포 (Count)
sentiment_counts = final_df.groupby(['brand', 'predicted_sentiment']).size().unstack(fill_value=0)
sentiment_counts = sentiment_counts.reindex(columns=sentiment_order, fill_value=0)
sentiment_counts.plot(
    kind='bar', ax=axes[0],
    color=[colors[s] for s in sentiment_order],
    edgecolor='white', width=0.6
)
axes[0].set_title('브랜드별 감정 분포', fontsize=13)
axes[0].set_xlabel('브랜드')
axes[0].set_ylabel('트윗 수')
axes[0].set_xticklabels(['Galaxy', 'iPhone'], rotation=0)
axes[0].legend(title='감정')

# 그래프 2. iPhone 감정 비율 파이차트
iphone_df = final_df[final_df['brand'] == 'iPhone']
iphone_counts = iphone_df['predicted_sentiment'].value_counts().reindex(sentiment_order, fill_value=0)
axes[1].pie(iphone_counts, labels=sentiment_order,
            colors=[colors[s] for s in sentiment_order],
            autopct='%1.1f%%', startangle=90)
axes[1].set_title(f'iPhone 감정 비율\n(총 {len(iphone_df)}개)', fontsize=13)

# 그래프 3. Galaxy 감정 비율 파이차트
galaxy_df = final_df[final_df['brand'] == 'Galaxy']
galaxy_counts = galaxy_df['predicted_sentiment'].value_counts().reindex(sentiment_order, fill_value=0)
axes[2].pie(galaxy_counts, labels=sentiment_order,
            colors=[colors[s] for s in sentiment_order],
            autopct='%1.1f%%', startangle=90)
axes[2].set_title(f'Galaxy 감정 비율\n(총 {len(galaxy_df)}개)', fontsize=13)

plt.tight_layout()
plt.show()

# ── 그래프 4. 브랜드별 감정 비율 비교 막대그래프 ──────
fig2, ax = plt.subplots(figsize=(10, 5))

ratio_df = sentiment_counts.div(sentiment_counts.sum(axis=1), axis=0) * 100
ratio_df.plot(kind='bar', ax=ax,
              color=[colors[s] for s in sentiment_order],
              edgecolor='white', width=0.5)

ax.set_title('브랜드별 감정 비율 비교 (%)', fontsize=14, fontweight='bold')
ax.set_xlabel('브랜드')
ax.set_ylabel('비율 (%)')
ax.set_xticklabels(['Galaxy', 'iPhone'], rotation=0)
ax.legend(title='감정')
ax.set_ylim(0, 100)

# 막대 위에 퍼센트 표시
for container in ax.containers:
    ax.bar_label(container, fmt='%.1f%%', fontsize=9)

plt.tight_layout()
plt.show()

# ── 수치 요약 ──────────────────────────────────────
print("\n📊 브랜드별 감정 비율 요약 (%):")
ratio_summary = ratio_df.round(1).astype(str) + '%'
display(ratio_summary)

print("\n📌 인사이트:")
for brand in ['iPhone', 'Galaxy']:
    row = ratio_df.loc[brand]
    print(f"\n{brand}")
    print(f"  긍정: {row['positive']:.1f}%  중립: {row['neutral']:.1f}%  부정: {row['negative']:.1f}%")
    dominant = row.idxmax()
    print(f"  → 가장 많은 감정: {dominant}")


