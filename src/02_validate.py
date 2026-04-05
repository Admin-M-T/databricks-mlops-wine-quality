from pyspark.sql import functions as F

df=spark.table("bronze_wine").dropDuplicates()

required_cols=["fixed_acidity","volatile_acidity","citric_acid","quality"]

for c in required_cols:
  df=df.filter(F.col(c).isNotNull())

df=df.filter((F.col("quality")>=0) & (F.col("quality")<=10))
df = df.withColumn("label",F.when(F.col("quality")>=7,1).otherwise(0)) #creates a binary target column called label


df.write.mode("overwrite").format("delta").saveAsTable("silver_wine")

print(f"Validated rows: {df.count()}")