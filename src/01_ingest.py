import os
from pyspark.sql import functions as F
import pandas as pd

csv_path=os.path.abspath(os.path.join(os.getcwd(),"..","data","raw","winequality-red.csv"))

input_path=f"file://{csv_path}"

print (f"Loading data from {input_path}")

pdf=pd.read_csv(csv_path,sep=";")

df=spark.createDataFrame(pdf)

for c in df.columns:
    new_c=c.strip().lower().replace(" ","_")
    df=df.withColumnRenamed(c,new_c)
    
df.write.mode("overwrite").format("delta").saveAsTable("bronze_wine")

print(f"Loaded {df.count()} rows into bronze_wine")