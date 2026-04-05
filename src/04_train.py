import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score,roc_auc_score

#mlflow does not work with databricks git folder, hence path is defined outside
mlflow.set_experiment("/Users/varun1995jain@gmail.com/wine_quality_experiment")

train_pdf =spark.table("train_set").toPandas()
test_pdf = spark.table("test_set").toPandas()

target="label"
drop_cols =["quality","label"]
features=[c for c in train_pdf.columns if c not in drop_cols]
X_train = train_pdf[features]
X_test = test_pdf[features]
y_train = train_pdf[target]
y_test = test_pdf[target]

models=[("Logistic Regression",LogisticRegression(max_iter=1000)),
        ("Random Forest",RandomForestClassifier(n_estimators=200,random_state=42))]

best_model_name= None
best_f1 = -1.0
for model_name,model in models:
  with mlflow.start_run(run_name=model_name):
    model.fit(X_train,y_train)
    preds=model.predict(X_test)

    if hasattr(model,"predict_proba"):
      probs=model.predict_proba(X_test)[:,1]
      auc = roc_auc_score(y_test,probs)
    else:
      auc=None

    f1=f1_score(y_test,preds)
    
    mlflow.log_param("model_name",model_name)
    mlflow.log_metric("f1",f1)

    if auc is not None:
        mlflow.log_metric("roc_auc",auc)
    
    # local_model_path =f"/tmp/{model_name}.joblib"
    # joblib.dump(model,local_model_path)
    # mlflow.log_artifact(local_model_path,artifact_path="model_artifacts")
    if f1>best_f1:
      best_f1=f1
      best_model_name=model_name

print(f"Best Model: {best_model_name}")
print(f"Best F1: {best_f1}")
    