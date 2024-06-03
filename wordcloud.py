# -*- coding: utf-8 -*-
"""wordcloud_model.ipynb"""

# pip install matplotlib wordcloud pandas nltk

import pandas as pd
import matplotlib.pyplot as plt
import os
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# nltk의 불용어(stopwords)를 다운로드합니다.
nltk.download('stopwords')

# CSV 파일에서 데이터를 불러옵니다. 파일 경로와 열 이름을 적절하게 수정하세요.
df1 = pd.read_csv('C:\Users\yoonj\gaming\my-nextjs-project\public\test.csv')

# 열 이름 출력
df1.columns

# 열 이름 한국어로 변경
df1.rename(columns={"title":"게임_제목", "user_review":"리뷰_텍스트"}, inplace=True)

# 불필요한 열 삭제하기
df1.drop(['review_id', 'year'], axis = 'columns', inplace=True)

df1

# 게임 제목을 기준으로 리뷰 데이터를 합칩니다.
merged_data = df1.groupby('게임_제목')['리뷰_텍스트'].apply(lambda x: ' '.join(x)).reset_index()

# 불용어를 제거합니다.
stop_words = set(stopwords.words('english'))
merged_data['리뷰_텍스트'] = merged_data['리뷰_텍스트'].apply(lambda x: ' '.join([word.lower() for word in x.split() if word.isalpha() and word.lower() not in stop_words]))

# 사용자 정의 불용어 목록 (불용어 추가하려면 여기에 추가하세요!)
custom_stopwords = ["game", "play", "character", "people", "gg", "really", "great", "time", "playing", "one",
                    "time", "dont", "good", "best", "thing", "played", "come", "well", "think", "need", "yes",
                    "right", "nice", "way", "going", "thank", "real", "made", "less", "sometime", "better",
                    "nothing", "without", "full", 'tell', "clicker", 'added', "already", "review", "see", "many",
                    "look", "lot", "want", "player", "pay","still", "even", "bit", "steam", "without", "actually",
                    "thing", "played", "come", "well", "think", "need", "yes", "right", "nice", "way", "going",
                    "thank", "real", "made", "less", "sometime", "better", "say", "seem", "much"]

# 게임 제목을 기준으로 워드클라우드를 생성합니다.
for index, row in merged_data.iterrows():
    game_title = row['게임_제목']
    review_text = row['리뷰_텍스트']

    # 사용자 정의 불용어를 제거합니다.
    review_text = ' '.join([word for word in review_text.split() if word not in custom_stopwords])

    # 워드클라우드 생성
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(review_text)

    # 워드클라우드를 시각화하고 저장합니다.
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(game_title)
    plt.axis('off')
    plt.savefig(f'{game_title}_wordcloud.png')
    plt.show()

# TF-IDF 벡터화
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(merged_data['리뷰_텍스트'])

# 사용자로부터 키워드 입력 받기
user_keyword = input("유사한 워드클라우드를 찾을 키워드를 입력하세요: ")

# 입력된 키워드에 대한 TF-IDF 벡터 계산
user_keyword_tfidf = tfidf_vectorizer.transform([user_keyword])

# 코사인 유사도 계산
cosine_sim = cosine_similarity(user_keyword_tfidf, tfidf_matrix)

# 유사한 워드클라우드 목록 생성
similar_wordclouds = list(merged_data['게임_제목'].loc[cosine_sim.argsort()[0][::-1][:5]])

# 결과 출력
print(f"\n{user_keyword}와(과) 유사한 워드클라우드 목록 (상위 5개):")
for idx, wordcloud_title in enumerate(similar_wordclouds, 1):
    print(f"{idx}. {wordcloud_title}")

# 게임 제목을 기준으로 리뷰 데이터를 합칩니다.
merged_data = df1.groupby('게임_제목')['리뷰_텍스트'].apply(lambda x: ' '.join(x)).reset_index()

# TF-IDF 벡터화
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(merged_data['리뷰_텍스트'])

# 사용자로부터 키워드 입력 받기
user_keyword = input("유사한 워드클라우드를 찾을 키워드를 입력하세요: ")

# 입력된 키워드에 대한 TF-IDF 벡터 계산
user_keyword_tfidf = tfidf_vectorizer.transform([user_keyword])

# 코사인 유사도 계산
cosine_sim = cosine_similarity(user_keyword_tfidf, tfidf_matrix)

# 코사인 유사도를 내림차순으로 정렬
similar_wordcloud_indices = cosine_sim.argsort()[0][::-1]

# 2위와 마지막 순위의 유사한 워드클라우드 목록 생성
second_similar_wordcloud = merged_data['게임_제목'].loc[similar_wordcloud_indices[1]]
last_similar_wordcloud = merged_data['게임_제목'].loc[similar_wordcloud_indices[-1]]

# 결과 출력
print(f"{user_keyword}와(과) 유사한 워드클라우드 2위: {second_similar_wordcloud}")
print(f"{user_keyword}와(과) 유사한 워드클라우드 마지막 순위: {last_similar_wordcloud}")

