from pyspark.sql import SparkSession
from pyspark.ml.feature import (VectorAssembler,VectorIndexer,
                                OneHotEncoder,StringIndexer)
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator

spark = SparkSession.builder.appName('classifier').getOrCreate()

data = spark.read.csv('data/titanic.csv', inferSchema=True, header=True)

my_cols = data.select(['Survived', 'Pclass', 'Sex', 'Age', 'SibSp',
                       'Parch', 'Fare', 'Embarked'])
my_final_data = my_cols.na.drop()

gender_indexer = StringIndexer(inputCol='Sex', outputCol='SexIndex')
gender_encoder = OneHotEncoder(inputCol='SexIndex', outputCol='SexVec')
embark_indexer = StringIndexer(inputCol='Embarked', outputCol='EmbarkIndex')
embark_encoder = OneHotEncoder(inputCol='EmbarkIndex', outputCol='EmbarkVec')
assembler = VectorAssembler(inputCols=['Pclass', 'SexVec', 'Age', 'SibSp',
                                       'Parch', 'Fare', 'EmbarkVec'],
                            outputCol='features')
log_reg_titanic = LogisticRegression(featuresCol='features', labelCol='Survived')

pipeline = Pipeline(stages=[gender_indexer, embark_indexer,
                            gender_encoder, embark_encoder,
                            assembler, log_reg_titanic])

train_titanic_data, test_titanic_data = my_final_data.randomSplit([0.7, 0.3])
fit_model = pipeline.fit(train_titanic_data)
results = fit_model.transform(test_titanic_data)
my_eval = BinaryClassificationEvaluator(rawPredictionCol='prediction', labelCol='Survived')
AUC = my_eval.evaluate(results)

print(AUC)