df =spark.table("silver_wine")

train_df,test_df=df.randomSplit([0.8,0.2],seed=42)

train_df.write.mode("overwrite").format("delta").saveAsTable("train_set")
test_df.write.mode("overwrite").format("delta").saveAsTable("test_set")

print(f"Train rows: {train_df.count()}")
print(f"Test rows: {test_df.count()}")