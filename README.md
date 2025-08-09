# 하루핏 (HARUFIT)  
> **"하루 안에 딱 맞는 알바를, 빠르고 정확하게"**

<div align=center><h1>📚 STACKS</h1></div>

<div align=center>
<img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white">
<img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white">
<img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white">
<img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white">
<img src="https://img.shields.io/badge/joblib-9C27B0?style=for-the-badge">
<img src="https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=matplotlib&logoColor=white">
<img src="https://img.shields.io/badge/Seaborn-0099CC?style=for-the-badge">
</div>



## 프로젝트 소개
이 프로젝트는 근무자 데이터베이스(DB)와 사용자 요청(request) 간의 매칭 점수를 학습한 모델을 활용해,
요청 조건에 가장 적합한 근무자를 실시간으로 추천하는 기능을 제공합니다.

- 랜덤포레스트 회귀모델로 요청 조건과 근무자 정보를 점수화
- 대규모 근무자 DB(100,000명) 기반의 빠른 추천
- 성별, 나이대, 지역, 직무, 근무시간, 경력여부 등 다양한 조건 반영
- 높은 예측 성능: R² 0.98 이상


## 주요 기능
1. **근무자 DB 생성**  
   - 100,000명 규모
   - 성별, 나이, 지역, 직무, 근무시간, 경력 여부 포함
2. **학습 데이터 생성**  
   - 요청-근무자 매칭 페어 30,000건 생성
   - 규칙 기반 점수를 타깃으로 사용
3. **모델 학습 및 저장**  
   - `RandomForestRegressor` + `OneHotEncoder`
   - 하이퍼파라미터 튜닝 (`RandomizedSearchCV`)
4. **실시간 추천**  
   - 요청 조건에 따라 Top-K 근무자 추천


## 기술 스택
- **Language** : Python 3.11
- **ML Library** : scikit-learn (RandomForestRegressor)
- **Data Handling** : Pandas, NumPy
- **Model I/O** : joblib
- **Visualization** : Matplotlib, Seaborn (개발용)

---

## 폴더 구조
```python
project/
│
├── data/
│   ├── workers.csv
│   └── train_data.csv
├── model/
│   └── matching_model.pkl
├── src/
│   ├── train_model.py
│   └── recommend.py
├── README.md
└── requirements.txt
```


## 빠른 시작
1) 데이터 생성 & 모델 학습
아래 스크립트 실행 시 자동으로 데이터 생성 → 학습 → 모델 저장까지 진행됩니다.

python train_model.py
실행 후 생성물

- data/workers.csv — 100,000명 근무자 DB (무작위 생성) 
- data/train_data.csv — 요청-근무자 페어 30,000건 + 규칙 점수 레이블 
- model/matching_model.pkl — OHE+RandomForest 파이프라인(튜닝 적용) 

터미널에는 베스트 하이퍼파라미터, MSE, R²가 출력됩니다.


## 동작 개요
###데이터
근무자(예시 100k): 성별, 나이, 지역, 직무, 시간대, 경력 여부 등 무작위 생성. 
학습 페어(30k): 무작위 요청을 샘플링하고, 사전 정의된 **규칙 점수(rule_score)**를 타깃으로 사용. 

### 규칙 점수(rule_score)
지역/직무/시간/경력/성별/나이 범위 일치 여부를 가중합(총 100점 만점)으로 산정. 

### 모델
파이프라인: OneHotEncoder(handle_unknown="ignore") + RandomForestRegressor
튜닝: RandomizedSearchCV (n_iter=10, 3-Fold KFold, MSE 기준) 

## 요청 스키마 (recommend)

recommend.py 내 함수 시그니처:

```python
def recommend(request, workers, model, topk=10): ...
```

request 예시:
```python
{
  "job": "고객 응대",
  "sex": "남자",
  "age": [25, 35],
  "region": "서초구",
  "time": "오후",
  "experience_required": True
}
```
* 동작:
  1. 요청 × 근무자 조합에 대해 특징을 생성
  2. 학습된 모델로 점수 예측
  3. 점수 내림차순으로 Top-K 반환
  4. 스크립트 예시는 Top 10을 콘솔로 출력

## 특징(Features)
학습/추론에 공통으로 사용하는 입력 특징:

* 범주형: worker_sex, request_sex, worker_region, request_region,
worker_position, request_job, worker_time, request_time

* 수치형: age_in_range, experience_match, region_match,
job_match, sex_match, time_match 


## 출력 예시
```python
=== 추천된 알바생 Top 3 ===
worker_46130 / 79점 / {'name': 'worker_46130', 'phone': '010-0004-6130', 'sex': '남자', 'age': 51, 'region': '서초구', 'position': '고객 응대', 'time': '오후', 'experience': True}
worker_29565 / 76점 / {'name': 'worker_29565', 'phone': '010-0002-9565', 'sex': '남자', 'age': 28, 'region': '동작구', 'position': '고객 응대', 'time': '오후', 'experience': True}
worker_36713 / 76점 / {'name': 'worker_36713', 'phone': '010-0003-36713', 'sex': '남자', 'age': 32, 'region': '동작구', 'position': '고객 응대', 'time': '오후', 'experience': True}
```

