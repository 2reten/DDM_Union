# 하루핏 (HARUFIT)
> "하루 안에 딱 맞는 알바를, 빠르고 정확하게"

<p align="center">
  <img src="https://raw.githubusercontent.com/2reten/DDM_Union/main/assets/Harufit_logo" alt="Harufit Logo" width="200"/>
</p>
<br>

<div align=center><h1> STACKS</h1></div>

<div align=center>
<img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white">
<img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white">
<img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white">
<img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white">
<br>
<img src="https://img.shields.io/badge/joblib-9C27B0?style=for-the-badge">
<img src="https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=matplotlib&logoColor=white">
<img src="https://img.shields.io/badge/Seaborn-0099CC?style=for-the-badge">
</div>

<br>
<br>
<br>

## 프로젝트 소개

기존의 아르바이트 채용은 [공고 등록 → 지원 대기 → 이력서 확인 → 연락 및 조율] 이라는 복잡하고 시간 소모적인 과정이었습니다.

하루핏(HARUFIT)은 이러한 비효율을 해결하기 위해 탄생했습니다. 더 이상 지원자를 기다리고 선별할 필요 없이, 채용 요청 즉시 가장 적합한 인재를 능동적으로 찾아주는 지능형 매칭 솔루션입니다.

머신러닝 모델이 10만 명의 인재풀에서 요청 조건과의 적합도를 실시간으로 분석하여 최적의 근무자를 추천합니다. 이력서를 일일이 검토하는 단계를 과감히 생략하고, 데이터 기반의 '매칭 점수'로 객관적인 판단을 돕습니다.

이를 통해 사용자는 채용에 드는 시간과 노력을 획기적으로 줄이고, 필요한 인력을 빠르고 정확하게 확보할 수 있습니다.

<br>

## 주요 기능

1.  대규모 근무자 DB 구축
    - 100,000명 규모의 가상 근무자 데이터 생성
    - 프로필 정보: 성별, 나이, 주소, 선호 직무, 가능 시간, 경력 유무 등

2.  규칙 기반 학습 데이터 생성
    - 사용자 요청과 근무자 간의 매칭 시나리오 30,000건 생성
    - 자체 정의된 규칙 기반 점수(Rule-based Score)를 타겟(Label)으로 활용

3.  매칭 모델 학습 및 최적화
    - `RandomForestRegressor`와 `OneHotEncoder`를 결합한 파이프라인 구축
    - `RandomizedSearchCV`를 이용한 하이퍼파라미터 튜닝으로 모델 성능 극대화

4.  실시간 Top-K 근무자 추천
    - 새로운 아르바이트 요청에 대해 DB 내 모든 근무자와의 매칭 점수를 실시간으로 예측
    - 가장 높은 점수를 받은 Top-K 근무자 목록을 즉시 반환

5.  자동 알림 기능 (구현 예정)
    - 추천 목록에서 사전에 설정된 기준 점수(Threshold)를 넘는 근무자에게 자동으로 알림을 발송
    - 구인자와 구직자 간의 신속한 연결을 유도하여 매칭 효율성 증대

<br>

## 기술 스택

-   Language: Python 3.11
-   ML/Data: scikit-learn, Pandas, NumPy
-   Model I/O: joblib
-   Visualization: Matplotlib, Seaborn (개발 및 분석용)

---

## 폴더 구조
```python
project/
│
├── data/
│   ├── workers.csv         # 근무자 DB (10만명)
│   └── train_data.csv      # 학습용 매칭 페어 (3만건)
│
├── model/
│   └── matching_model.pkl  # 학습된 RandomForest 모델
│
├── src/
│   ├── train_model.py      # 데이터 생성 및 모델 학습 스크립트
│   └── recommend.py        # 실시간 추천 및 결과 반환 스크립트
│
├── README.md
└── requirements.txt
```

<br>


## 빠른 시작

#### 1. 데이터 생성 및 모델 학습
아래 스크립트를 실행하면 `data` 폴더와 `model` 폴더에 필요한 파일이 모두 생성됩니다.
> 데이터 생성 → 학습 데이터 구축 → 모델 학습 및 저장까지 한번에 진행됩니다.

python src/train_model.py

-   실행 후 생성 파일
    -   `data/workers.csv`: 100,000명의 무작위 근무자 DB
    -   `data/train_data.csv`: 30,000건의 요청-근무자 페어 및 규칙 점수
    -   `model/matching_model.pkl`: 최적화된 Scikit-learn 파이프라인 모델
-   터미널 출력
    -   탐색된 최적의 하이퍼파라미터와 테스트셋에 대한 MSE, R² 성능 지표가 출력됩니다.

#### 2. 추천 기능 실행
`recommend.py` 파일 내의 `request_demo` 값을 수정하여 원하는 조건으로 테스트할 수 있습니다.

python src/recommend.py

<br>

## 동작 개요

### 데이터 처리
-   근무자(Worker): 성별, 나이, 지역, 직무, 시간대, 경력 여부 등 10만 명의 데이터를 무작위로 생성하여 현실적인 DB를 모사합니다.
-   학습 데이터(Training Pair): 무작위 요청(Request)과 근무자 정보를 조합하고, 자체 규칙 점수(Rule-based Score)를 계산하여 모델이 학습할 정답 데이터로 사용합니다.

### 규칙 점수 (Rule-based Score)
모델이 학습할 목표 값으로, 각 조건의 중요도에 따라 가중치를 부여하여 100점 만점으로 계산됩니다.
-   주요 평가 항목: `지역`, `직무`, `시간`, `경력`, `성별`, `나이대` 일치 여부

### 모델링
-   파이프라인: 범주형 데이터를 처리하기 위한 `OneHotEncoder`와 `RandomForestRegressor` 모델을 결합하여 데이터 전처리 및 학습을 동시에 수행합니다.
    -  `handle_unknown="ignore"` 옵션으로 새로운 입력 값에 대한 에러 방지
-   하이퍼파라미터 튜닝: `RandomizedSearchCV`를 사용하여 최적의 모델 파라미터를 탐색하고, 3-Fold 교차 검증을 통해 일반화 성능을 높입니다.

### 추천 및 알림 프로세스
`recommend.py` 스크립트는 다음 순서로 동작합니다.
1.  사용자 요청(Request) 수신
2.  전체 근무자 DB와 요청을 결합하여 모델 입력용 특징(Feature) 생성
3.  학습된 모델(`matching_model.pkl`)을 로드하여 각 근무자별 매칭 점수 예측
4.  예측 점수를 기준으로 내림차순 정렬하여 Top-K명의 근무자 반환
5.  (구현 예정) 기준 점수를 초과하는 근무자에게 자동으로 알림 발송

<br>

---

## 요청 스키마 (Request Schema)

`recommend.py`의 `recommend` 함수는 아래와 같은 형식의 `request` 딕셔너리를 입력받습니다.

<br>

## recommend.py 내 요청 예시
```python
request_demo = {
  "job": "고객 응대",
  "sex": "남자",
  "age": [25, 35],
  "region": "서초구",
  "time": "오후",
  "experience_required": True
}
```
<br>

## 특징 (Features)

모델 학습 및 추론에 공통으로 사용되는 입력 특징입니다.
```python
범주형 (Categorical): `worker_sex`, `request_sex`, `worker_region`, `request_region`, `worker_position`, `request_job`, `worker_time`, `request_time`
수치형 (Numerical): `age_in_range`, `experience_match`, `region_match`, `job_match`, `sex_match`, `time_match`
```
<br>

## 출력 예시
```python
=== 추천된 알바생 Top 3 ===
worker_46130 / 79점 / {'name': 'worker_46130', 'phone': '010-0004-6130', 'sex': '남자', 'age': 51, 'region': '서초구', 'position': '고객 응대', 'time': '오후', 'experience': True, ...}
worker_29565 / 76점 / {'name': 'worker_29565', 'phone': '010-0002-9565', 'sex': '남자', 'age': 28, 'region': '동작구', 'position': '고객 응대', 'time': '오후', 'experience': True, ...}
worker_36713 / 76점 / {'name': 'worker_36713', 'phone': '010-0003-36713', 'sex': '남자', 'age': 32, 'region': '동작구', 'position': '고객 응대', 'time': '오후', 'experience': True, ...}

* 위 근무자들에게는 기준 점수 충족 시 매칭 알림이 자동으로 발송될 수 있습니다.
```
