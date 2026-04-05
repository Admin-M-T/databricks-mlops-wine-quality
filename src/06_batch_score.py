from sklearn.ensemble import RandomForestClassifier

train_pdf=spark.table("train_set").toPandas()
test_pdf=spark.table("test_set").toPandas()

target="label"
drop_cols=["quality","label"]
features=[c for c in train_pdf.columns if c not in drop_cols]

X_train=train_pdf[features]
y_train=train_pdf[target]

X_test=test_pdf[features]

model = RandomForestClassifier(n_estimators=200,random_state=42)
model.fit(X_train, y_train)
preds = model.predict(X_test)
X_test["prediction"]=preds

pred_spark=spark.createDataFrame(X_test)
pred_spark.write.mode("overwrite").format("delta").saveAsTable("batch_predictions")

print("Saved predictions to batch_predictions")