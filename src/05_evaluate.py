import json
import mlflow
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

train_pdf=spark.table("train_set").toPandas()
test_pdf=spark.table("test_set").toPandas()

target="label"
drop_cols=["quality","label"]
features=[c for c in train_pdf.columns if c not in drop_cols]

X_train=train_pdf[features]
y_train=train_pdf[target]

X_test=test_pdf[features]
y_test=test_pdf[target]

model=RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)
preds= model.predict(X_test)

report=classification_report(y_test,preds,output_dict=True)
matrix=confusion_matrix(y_test,preds).tolist()

metrics={"classification_report_json":json.dumps(report),"confusion_matrix_json":json.dumps(matrix)}

metrics_pdf=pd.DataFrame([metrics])
metrics_df=spark.createDataFrame(metrics_pdf)
#dbutils.fs.put("dbfs:/tmp/wine_quality_metrics.json",json.dumps(metrics,indent=2),overwrite=True)

print("Saved evaluation metrics to table: evaluation_metrics")