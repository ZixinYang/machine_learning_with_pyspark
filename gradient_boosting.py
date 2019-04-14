from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StringIndexer
from pyspark.ml.classification import DecisionTreeClassifier,GBTClassifier,RandomForestClassifier
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

spark = SparkSession.builder.appName('GB').getOrCreate()

data = spark.read.csv('data/College.csv', inferSchema=True, header=True)

assembler = VectorAssembler(
    inputCols=['Apps', 'Accept', 'Enroll', 'Top10perc',
               'Top25perc', 'F_Undergrad', 'P_Undergrad',
               'Outstate', 'Room_Board', 'Books',
               'Personal', 'PhD', 'Terminal',
               'S_F_Ratio', 'perc_alumni', 'Expend',
               'Grad_Rate'],
    outputCol="features")

output = assembler.transform(data)
indexer = StringIndexer(inputCol="Private", outputCol="PrivateIndex")
output_fixed = indexer.fit(output).transform(output)
final_data = output_fixed.select("features", 'PrivateIndex')
train_data,test_data = final_data.randomSplit([0.7,0.3])

gbt = GBTClassifier(labelCol='PrivateIndex', featuresCol='features')
gbt_model = gbt.fit(train_data)
gbt_predictions = gbt_model.transform(test_data)
acc_evaluator = MulticlassClassificationEvaluator(labelCol="PrivateIndex", predictionCol="prediction", metricName="accuracy")
gbt_acc = acc_evaluator.evaluate(gbt_predictions)
print('A ensemble using GBT had an accuracy of: {0:2.2f}%'.format(gbt_acc*100))