import random
import numpy as np
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.metrics import mean_squared_error, r2_score

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

regions = ["종로구", "중구", "용산구", "성동구", "광진구", "동대문구", "중랑구", "성북구", "강북구",
           "도봉구", "노원구", "은평구", "서대문구", "마포구", "양천구", "강서구", "구로구", "금천구",
           "영등포구", "동작구", "관악구", "서초구", "강남구", "송파구", "강동구"]
sex = ['남자', '여자']
time_slots = ['오전', '오후', '종일']
jobs = [
    "카페 알바", "편의점 알바", "마트 계산원", "서빙", "주방보조", "배달", "택배 분류", "물류센터", "콜센터", "사무보조",
    "서점 직원", "문구점 직원", "독서실/스터디카페 관리", "주유소 직원", "세차장 직원", "청소 알바", "건설 현장 보조",
    "인테리어 보조", "전단지 배포", "샘플 배포", "이벤트 스태프", "행사 도우미", "결혼식 도우미", "피팅모델",
    "웹디자인 보조", "영상 편집 보조", "사진 촬영 보조", "프로그래밍 보조", "IT 개발 보조", "SNS 관리",
    "블로그 글쓰기", "유튜브 편집", "온라인 쇼핑몰 운영 보조", "상품 포장", "창고 정리", "매장 진열", "고객 응대",
    "방문판매", "전화 영업", "리서치 조사원", "서베이 조사", "미스터리 쇼퍼", "영화관 스태프", "놀이공원 스태프",
    "수영장 안전요원", "헬스장 데스크", "PC방 알바", "노래방 알바", "볼링장 알바", "키즈카페 스태프",
    "놀이방 교사 보조", "어린이집 보조교사", "학원 보조교사", "학습지 교사", "과외", "외국어 과외",
    "통역 알바", "번역 알바", "원어민 회화 도우미", "전시회 도우미", "갤러리 안내원", "도서 정리", "논문 교정",
    "출판 보조", "도서관 사서 보조", "약국 보조", "병원 접수", "병원 안내", "간병 도우미", "마사지샵 보조",
    "네일샵 보조", "미용실 보조", "웨딩샵 피팅 보조", "렌터카 안내", "주차 요원", "발렛파킹", "호텔 리셉션",
    "게스트하우스 관리", "여행사 보조", "공항 안내", "면세점 스태프", "보안요원", "경비원", "주차관리",
    "사무실 청소", "학교 급식 보조", "공장 단순노무", "전자부품 조립", "식품 포장", "분식점 알바",
    "제과점 알바", "아이스크림 가게", "패스트푸드점", "치킨집", "피자집", "고깃집", "횟집", "뷔페", "술집", "호프집"
]

def make_workers_uniform(n=100000):
    return [{
        "name": f"worker_{i}",
        "phone": f"010-0000-{i:04d}",
        "sex": random.choice(sex),
        "age": random.randint(20, 60),
        "region": random.choice(regions),
        "position": random.choice(jobs),
        "time": random.choice(time_slots),
        "experience": random.choice([True, False])
    } for i in range(n)]

def rule_score(worker, request):
    score = 0
    if worker["region"] == request["region"]: score += 20
    if worker["position"] == request["job"]: score += 20
    if worker["sex"] == request["sex"]: score += 10
    if request["age"][0] <= worker["age"] <= request["age"][1]: score += 10
    if worker["time"] == request["time"]: score += 20
    if worker["experience"] == request["experience_required"]: score += 20
    return score

def sample_request(prefer_job=None):
    job = prefer_job if (prefer_job and random.random() < 0.3) else random.choice(jobs)
    return {
        "job": job,
        "sex": random.choice(sex),
        "age": sorted([random.randint(20, 60) for _ in range(2)]),
        "region": random.choice(regions),
        "time": random.choice(time_slots),
        "experience_required": random.choice([True, False]),
    }

def build_training_data(workers, n_pairs=30000):
    rows, y = [], []
    for _ in range(n_pairs):
        w = random.choice(workers)
        req = sample_request(prefer_job=w["position"])
        rows.append({
            "worker_sex": w["sex"],
            "request_sex": req["sex"],
            "age_in_range": int(req["age"][0] <= w["age"] <= req["age"][1]),
            "worker_region": w["region"],
            "request_region": req["region"],
            "worker_position": w["position"],
            "request_job": req["job"],
            "worker_time": w["time"],
            "request_time": req["time"],
            "experience_match": int(w["experience"] == req["experience_required"]),
            "region_match": int(w["region"] == req["region"]),
            "job_match": int(w["position"] == req["job"]),
            "sex_match": int(w["sex"] == req["sex"]),
            "time_match": int(w["time"] == req["time"]),
        })
        y.append(rule_score(w, req))
    return pd.DataFrame(rows), np.array(y, dtype=float)

workers = make_workers_uniform(100000)
pd.DataFrame(workers).to_csv("data/workers.csv", index=False, encoding="utf-8-sig")

X_raw, y = build_training_data(workers, n_pairs=30000)
df_train = X_raw.copy()
df_train["score"] = y
df_train.to_csv("data/train_data.csv", index=False, encoding="utf-8-sig")

cat_cols = ["worker_sex","request_sex","worker_region","request_region",
            "worker_position","request_job","worker_time","request_time"]
num_cols = ["age_in_range","experience_match","region_match","job_match","sex_match","time_match"]

preprocess = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ("num", "passthrough", num_cols),
])

pipe = Pipeline([
    ("prep", preprocess),
    ("model", RandomForestRegressor(random_state=SEED, n_jobs=-1))
])

param_distributions = {
    "model__n_estimators": [300, 500, 700],
    "model__max_depth": [None, 12, 16, 20],
    "model__min_samples_split": [2, 5],
    "model__min_samples_leaf": [1, 2],
    "model__max_features": ["sqrt", "log2"],
}

X_train, X_test, y_train, y_test = train_test_split(X_raw, y, test_size=0.2, random_state=SEED)
cv = KFold(n_splits=3, shuffle=True, random_state=SEED)
search = RandomizedSearchCV(pipe, param_distributions=param_distributions,
                             n_iter=10, scoring="neg_mean_squared_error",
                             cv=cv, random_state=SEED, n_jobs=-1)
search.fit(X_train, y_train)
best_pipe = search.best_estimator_

joblib.dump(best_pipe, "model/matching_model.pkl")

y_pred = best_pipe.predict(X_test)
print("Best Params:", search.best_params_)
print("MSE:", mean_squared_error(y_test, y_pred))
print("R²:", r2_score(y_test, y_pred))
