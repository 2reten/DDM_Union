import pandas as pd
import joblib

def recommend(request, workers, model, topk=10):
    feat_list, payload = [], []
    for w in workers:
        payload.append(w)
        feat_list.append({
            "worker_sex": w["sex"],
            "request_sex": request["sex"],
            "age_in_range": int(request["age"][0] <= w["age"] <= request["age"][1]),
            "worker_region": w["region"],
            "request_region": request["region"],
            "worker_position": w["position"],
            "request_job": request["job"],
            "worker_time": w["time"],
            "request_time": request["time"],
            "experience_match": int(w["experience"] == request["experience_required"]),
            "region_match": int(w["region"] == request["region"]),
            "job_match": int(w["position"] == request["job"]),
            "sex_match": int(w["sex"] == request["sex"]),
            "time_match": int(w["time"] == request["time"]),
        })
    df = pd.DataFrame(feat_list)
    scores = model.predict(df)
    ranked = sorted(zip(payload, scores), key=lambda x: x[1], reverse=True)[:topk]
    return ranked

workers_df = pd.read_csv("data/workers.csv")
workers = workers_df.to_dict(orient="records")
model = joblib.load("model/matching_model.pkl")

request_demo = {
    'job': '고객 응대',
    'sex': '남자',
    'age': [25, 35],
    'region': '서초구',
    'time': '오후',
    'experience_required': True
}

top_candidates = recommend(request_demo, workers, model, topk=10)
print("\n=== 추천된 알바생 Top 10 ===")
for w, s in top_candidates:
    print(f"{w['name']} / {int(round(s))}점 / {w}")
