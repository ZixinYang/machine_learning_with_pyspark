import pyspark
from pyspark.sql import SparkSession
from pyspark.ml.regression import LinearRegression
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler

spark = SparkSession.builder.appName('LR').getOrCreate()

data = spark.read.csv("data/Ecommerce_Customers.csv", inferSchema=True, header=True)

assembler = VectorAssembler(
    inputCols = [
        "Avg Session Length", "Time on App",
        "Time on Website", "Length of Membership"],
    outputCol = "features")

output = assembler.transform(data)
final_data = output.select("features", "Yearly Amount Spent")
train_data, test_data = final_data.randomSplit([0.7, 0.3])

lr = LinearRegression(labelCol='Yearly Amount Spent')
lrModel = lr.fit(train_data)
test_results = lrModel.evaluate(test_data)

print("RMSE: {}".format(test_results.rootMeanSquaredError))
print("MSE: {}".format(test_results.meanSquaredError))
