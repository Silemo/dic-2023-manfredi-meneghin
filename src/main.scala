// Databricks notebook source
// MAGIC %md #Titanic: Spark and Machine Learning from disaster
// MAGIC ---
// MAGIC Author: Giovanni Manfredi, Sebastiano Meneghin
// MAGIC
// MAGIC Email: gioman@kth.se, meneghin@kth.se
// MAGIC
// MAGIC Last update: 21st October 2023
// MAGIC
// MAGIC Github Repository: https://github.com/Silemo/dic-2023-manfredi-meneghin

// COMMAND ----------

// DBTITLE 1,Project Library Import
// Packages for ML
import org.apache.spark.{ml => spml}
import org.apache.spark.{mllib => spmlib}
import spml.classification.{LogisticRegression, LogisticRegressionModel}
import spml.classification.{DecisionTreeClassifier, DecisionTreeClassificationModel}
import spml.classification.{RandomForestClassifier, RandomForestClassificationModel}
import spml.classification.{NaiveBayes, NaiveBayesModel}
import spml.feature.{VectorIndexer, IndexToString, StringIndexer, VectorAssembler}
import spml.linalg.Vectors
import spml.{Pipeline, PipelineModel}
import spml.evaluation.MulticlassClassificationEvaluator
import spmlib.evaluation.MulticlassMetrics
import spmlib.util.MLUtils

// Packages for Data Analysis
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.catalog
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{DataType, FloatType, IntegerType, BooleanType, NullType}

// Packages for Data Manipulation
import scala.util.matching.Regex
import scala.util.matching.UnanchoredRegex
import scala.collection.mutable
import spark.implicits._
import java.util.regex.Pattern




// COMMAND ----------

// Data are here loaded from Databricks Datasets, in the folder "default"
val dataSet  = spark.read.table("default.dataset")

// Cast type and rename columns according to the .csv provided by the competition, if there are any problem with the import of the csv to a table
//val testDataCast = testData.withColumnRenamed("_c0", "PassengerId").withColumnRenamed("_c1", "PClass").withColumnRenamed("_c2", "Name").withColumnRenamed("_c3","Sex").withColumnRenamed("_c4", "Age").withColumnRenamed("_c5", "SibSp").withColumnRenamed("_c6", "Parch").withColumnRenamed("_c7", "Ticket").withColumnRenamed("_c8", "Fare").withColumnRenamed("_c9", "Cabin").withColumnRenamed("_c10", "Embarked").withColumn("PassengerId", col("PassengerId").cast(IntegerType)).withColumn("PClass", col("PClass").cast(IntegerType)).withColumn("Age", col("Age").cast(FloatType)).withColumn("SibSp", col("SibSp").cast(IntegerType)).withColumn("Parch", col("Parch").cast(IntegerType)).withColumn("Fare", col("Fare").cast(FloatType))
//val trainDataCast = trainData.withColumnRenamed("_c0", "PassengerId").withColumnRenamed("_c1", "Survived").withColumnRenamed("_c2", "PClass").withColumnRenamed("_c3", "Name").withColumnRenamed("_c4","Sex").withColumnRenamed("_c5", "Age").withColumnRenamed("_c6", "SibSp").withColumnRenamed("_c7", "Parch").withColumnRenamed("_c8", "Ticket").withColumnRenamed("_c9", "Fare").withColumnRenamed("_c10", "Cabin").withColumnRenamed("_c11", "Embarked").withColumn("PassengerId", col("PassengerId").cast(IntegerType)).withColumn("Survived", col("Survived").cast(IntegerType)).withColumn("PClass", col("PClass").cast(IntegerType)).withColumn("Age", col("Age").cast(FloatType)).withColumn("SibSp", col("SibSp").cast(IntegerType)).withColumn("Parch", col("Parch").cast(IntegerType)).withColumn("Fare", col("Fare").cast(FloatType))

// COMMAND ----------

// DataSet Data Visualisation
// Here the data are analysed, trying to find valuable information about them, like pattern, unknown data, trends.
dataSet.describe().select("summary", "PassengerId", "Survived", "PClass", "Age", "SibSp", "Parch", "Fare", "Embarked", "Cabin").show(false)

// We can see from above that there are some rows in the table which not contains a value for the attribute, as Age, Embarked and Cabin
dataSet.createOrReplaceTempView("initialView")
spark.sql("SELECT count(*) AS NullAgeRows      FROM initialView WHERE Age      IS NULL").show()
spark.sql("SELECT count(*) AS NullEmbarkedRows FROM initialView WHERE Embarked IS NULL").show()
spark.sql("SELECT count(*) AS NullCabinRows    FROM initialView WHERE Cabin    IS NULL").show()

// COMMAND ----------

// MAGIC %md 
// MAGIC After those short analyses, **we have to consider that:**
// MAGIC - Data have different types, such as strings, float, int
// MAGIC - Some rows of the tables have missing data
// MAGIC - We have both Categorical and Numerical data
// MAGIC - The feature "Name" contains not only the name of the people, but also their "title", which might effect their survival chance
// MAGIC - SibSp and Parch relate both to the number of people related to the passenger (number of siblings, spouses, parents and children)
// MAGIC
// MAGIC **Looking at the statistics**, we highlight that:
// MAGIC - Fares varies significantly
// MAGIC - All the names are unique
// MAGIC - Cabin values have a lot of duplicates, meaning that passengers shared cabins
// MAGIC - Embarked most common value is S, whereas there are three possible value (S, C, Q)
// MAGIC
// MAGIC Said that and looked to the data, the **future steps are:**
// MAGIC - Derive from the feature "Name" the "Title" of the passengers
// MAGIC - Complete Age feature and Embarked features
// MAGIC - Drop some features that contains duplicates, are full of nulls, or we think will not contribute to survival, such as PassengerId, Cabin, Ticket
// MAGIC - Understand how the features correlate with survival chances.
// MAGIC - Create a feature which can containts the aggregated information of SibSp and Parch
// MAGIC - Remove continuos data like Age and Fare with something categorial as AgeGroup and FareLevel.
// MAGIC
// MAGIC Lastly, according to the [History of Titanic](https://en.wikipedia.org/wiki/Titanic), the survivor where mostly women and children from the upper-class passengers, which were also the people closest to the lifeboats. We should add this assumption to the problem description later.
// MAGIC

// COMMAND ----------

// The assumption about correlation between survival and upper-class or sex are confirmed by the following result, where we cannot ignora a significant correlation 0.63% survival rate of people travelling in the first class and a 0.74% for women, against a 0.38% survival rate of the passenger in the training set. Thus PClass and Sex must be used in our prediction model!
spark.sql("SELECT PClass, cast(avg(Survived) as numeric(36,2)) AS SurvivalRate FROM initialView GROUP BY PClass ORDER BY PClass ").show()
spark.sql("SELECT Sex, cast(avg(Survived) as numeric(36,2)) AS SurvivalRate FROM initialView GROUP BY Sex").show()

// COMMAND ----------

// Analysing the age in AgeGroups, we can see that young children, under five years old, had incresed survival rate, as for elderly people, while lowest survival rate are the people in the group 16-35 year old. Thus, Age must be used in our prediction model!
val histogram = spark.sql(f"""
with hist as (
  select 
    width_bucket(Age, 0, 100, 20) as bucket,
    cast(avg(Survived) as numeric(36,2)) as cnt
  from initialView
  group by bucket
  ),
buckets as (
  select id+1 as bucket from range(20)
)
select
    bucket, (bucket) * 5 as value,
    nvl(cnt, 0) as SurvivalRate
from hist right outer join buckets using(bucket)
order by bucket
""")

histogram.show()


// COMMAND ----------

// Analysing the feature Embarked, we can see that the SurvivalRate changes depending on the embarkment. Also, depending on the embarkment, women and men have more or less pronounced difference in their survival rate. Thus, Embarked must be used in our prediction model!
spark.sql("SELECT Embarked, Sex, cast(avg(Survived) as numeric(36,2)) as SurvivalRate FROM initialView WHERE Embarked IS NOT NULL GROUP BY Embarked, Sex ORDER BY Embarked DESC, Sex ASC").show


// COMMAND ----------

// Analysing the Fare paid by different passenger, we can see how much difference there is among different fares paid. Thus, Fare must be used in our prediction model!
val histogram = spark.sql(f"""
with hist as (
  select 
    width_bucket(Fare, 0, 150, 20) as bucket,
    cast(avg(Survived) as numeric(36,2)) as cnt,
    count(*) as counter
  from initialView
  group by bucket
  ),
buckets as (
  select id+1 as bucket from range(15)
)
select
    bucket, (bucket) * 10 as value,
    nvl(cnt, 0) as SurvivalRate,
    counter as PassNum
from hist right outer join buckets using(bucket)
where counter > 0
order by bucket
""")

histogram.show()

// COMMAND ----------

// Analysing the features Parch and SibSp we can see that being an alone passenger decrease the survival rate. Thus, this information must be used in our prediction model!
spark.sql("SELECT cast(avg(Survived) as numeric(36,2)) AS AloneSurvival FROM initialView WHERE Parch = 0 AND SibSp = 0").show()
spark.sql("SELECT cast(avg(Survived) as numeric(36,2)) AS NotAloneSurvival FROM initialView WHERE Parch + SibSp > 0").show()

// COMMAND ----------

// As said above, the feature Name contains another information, which is the "Title". This feature can be retrieved by applying a RegEx to the value of the attribute "Name", looking for the word ending with a period. We remove here also the feature Name, PassengerId, Cabin and Ticket.

// The title are unified in 5 groups (Mr, Miss, Mrs, Master and Uncommon), to which are then assigned five different numbers, to ease the following the prediction model, which are respectively (1, 2, 3, 4, 5)

// Furthermore, the Sex is converted into a numerical value (Men 0, Women 1)

val tempDS1 = dataSet.withColumn("Name", regexp_extract($"Name","([A-Za-z]+\\.)",1)) 
                        .withColumnRenamed("Name", "Title")
                        .withColumn("Title", 
                           when(col("Title") === "Lady.",     5)
                          .when(col("Title") === "Countess.", 5)
                          .when(col("Title") === "Capt.",     5)
                          .when(col("Title") === "Col.",      5)
                          .when(col("Title") === "Don.",      5)
                          .when(col("Title") === "Dr.",       5)
                          .when(col("Title") === "Major.",    5)
                          .when(col("Title") === "Rev.",      5)
                          .when(col("Title") === "Sir.",      5)
                          .when(col("Title") === "Jonkheer.", 5)
                          .when(col("Title") === "Dona.",     5)
                          .when(col("Title") === "Mlle.",     2)
                          .when(col("Title") === "Ms.",       2)
                          .when(col("Title") === "Miss.",     2)
                          .when(col("Title") === "Mme.",      3)
                          .when(col("Title") === "Mrs.",      3)
                          .when(col("Title") === "Master.",   4)
                          .when(col("Title") === "Mr.",       1))
                        .withColumn("Title", col("Title").cast(IntegerType))
                        .drop("Name", "Cabin", "Ticket")
                        .withColumn("Sex",
                         when(col("Sex") === "male",   0)
                        .when(col("Sex") === "female", 1))

// COMMAND ----------

// We need to complete the feature Age in all the rows. We have previously seen how the age is correlated with PClass and Sex. So we calculate the median values for each group "(PClass, Sex)" in order to insert that value into "Age" when needed.

// Each age then is associated to a specific AgeGroup, labelled by a integer
tempDS1.createOrReplaceTempView("tempDS1View")

val medianP1M = spark.sql("SELECT percentile_cont(0.5) WITHIN GROUP  (ORDER BY tempDS1View.Age) FROM tempDS1View WHERE tempDS1View.PClass == 1 AND Sex = 0 ").first().getDouble(0).toInt
val medianP2M = spark.sql("SELECT percentile_cont(0.5) WITHIN GROUP  (ORDER BY tempDS1View.Age) FROM tempDS1View WHERE tempDS1View.PClass == 2 AND Sex = 0 ").first().getDouble(0).toInt
val medianP3M = spark.sql("SELECT percentile_cont(0.5) WITHIN GROUP  (ORDER BY tempDS1View.Age) FROM tempDS1View WHERE tempDS1View.PClass == 3 AND Sex = 0 ").first().getDouble(0).toInt
val medianP1W = spark.sql("SELECT percentile_cont(0.5) WITHIN GROUP  (ORDER BY tempDS1View.Age) FROM tempDS1View WHERE tempDS1View.PClass == 1 AND Sex = 1 ").first().getDouble(0).toInt
val medianP2W = spark.sql("SELECT percentile_cont(0.5) WITHIN GROUP  (ORDER BY tempDS1View.Age) FROM tempDS1View WHERE tempDS1View.PClass == 2 AND Sex = 1 ").first().getDouble(0).toInt
val medianP3W = spark.sql("SELECT percentile_cont(0.5) WITHIN GROUP  (ORDER BY tempDS1View.Age) FROM tempDS1View WHERE tempDS1View.PClass == 3 AND Sex = 1 ").first().getDouble(0).toInt

val tempDS2 = tempDS1.withColumn("Age", 
                   when(col("Age").isNull and col("PClass") === 1 and col("Sex") === 0, medianP1M)
                  .when(col("Age").isNull and col("PClass") === 2 and col("Sex") === 0, medianP2M)
                  .when(col("Age").isNull and col("PClass") === 3 and col("Sex") === 0, medianP3M)
                  .when(col("Age").isNull and col("PClass") === 1 and col("Sex") === 1, medianP1W)
                  .when(col("Age").isNull and col("PClass") === 2 and col("Sex") === 1, medianP2W)
                  .when(col("Age").isNull and col("PClass") === 3 and col("Sex") === 1, medianP3W)
                  .otherwise($"Age"))
                .withColumn("Age",
                   when(col("Age") <= 14,                      1)
                  .when(col("Age") >  14 and col("Age") <= 28, 2)
                  .when(col("Age") >  28 and col("Age") <= 42, 3)
                  .when(col("Age") >  42 and col("Age") <= 56, 4)
                  .when(col("Age") >  56,                      5)
                  .otherwise(1))
                .withColumn("Age", col("Age").cast(IntegerType))


// COMMAND ----------

// We transform Parch and SibSp in a single boolean variable called BeingAlone. This is gonna be added as a feature to our prediction model.

val tempDS3 = tempDS2.withColumn("BeingAlone", 
                              when(col("Parch") + col("SibSp") > 0, 0)
                              .when(col("Parch") + col("SibSp") === 0, 1))
                            .withColumn("BeingAlone", col("BeingAlone").cast(IntegerType))
                            .drop("Parch", "SibSp")

// COMMAND ----------

// We take care of the missing information present on the dataset about Embarked, where two value are missing. We replace them with the most occurent value.
// Then, we assing to each embarkment label a number, from 1 to 3

tempDS3.createOrReplaceTempView("tempDS3View")
val embMostOcc = spark.sql("SELECT Embarked, count(*) AS TotalEmbarked FROM tempDS3View WHERE Embarked IS NOT NULL GROUP BY Embarked ORDER BY count(*) DESC").first().getString(0)

val tempDS4 = tempDS3.withColumn("Embarked", 
                               when(col("Embarked").isNull, embMostOcc)
                              .otherwise($"Embarked"))
                            .withColumn("Embarked",
                               when(col("Embarked") === "Q", 1)
                              .when(col("Embarked") === "C", 2)
                              .when(col("Embarked") === "S", 3))

// COMMAND ----------

// We complete now the feature Fare, by adding the median to the missing value. Then we divide the Fare into FareLevel, selected on the quartiles of the Fare distribution.

tempDS4.createOrReplaceTempView("tempDS4View")
val fareOne   = spark.sql("SELECT percentile_cont(0.25) WITHIN GROUP  (ORDER BY tempDS4View.Fare) FROM tempDS4View ").first().getDouble(0)
val fareTwo   = spark.sql("SELECT percentile_cont(0.50) WITHIN GROUP  (ORDER BY tempDS4View.Fare) FROM tempDS4View ").first().getDouble(0)
val fareThree = spark.sql("SELECT percentile_cont(0.75) WITHIN GROUP  (ORDER BY tempDS4View.Fare) FROM tempDS4View ").first().getDouble(0)

val tempDS5 = tempDS4.withColumn("Fare",
                               when(col("Fare").isNull, fareTwo)
                               .otherwise($"Fare"))
                            .withColumn("Fare",
                               when(col("Fare") <= fareOne,                                 1)
                              .when(col("Fare") >  fareOne    and col("Fare") <= fareTwo,   2)
                              .when(col("Fare") >  fareTwo    and col("Fare") <= fareThree, 3)
                              .when(col("Fare") >  fareThree,                               4))
                            .withColumn("Fare", col("Fare").cast(IntegerType))
                            

// COMMAND ----------

// We split now the set into the final TrainSet and TestSet, according to the 80:20 Rule.
val splits = tempDS5.randomSplit(Array(0.8, 0.2))
val (trainingDataSet, testDataSet) = (splits(0), splits(1))

// COMMAND ----------

// Select the attributes I want to train the model on.
val columns = Array("PassengerId", "Survived", "Pclass", "Title", "Sex", "Age", "Fare", "Embarked", "BeingAlone")

// Remove the feature Survived from the VectorAssembler column, since it is gonna be the column "label"
val colomnsOfPred = columns.drop(2)
val realTestSet = testDataSet.drop("Survived")
testDataSet.createOrReplaceTempView("testSetView")

// The rumber of row contained in the realTestSet is counted and then will be used for the accuracy calculation
val totalRows   = spark.sql("SELECT count(*) FROM testSetView").first().getLong(0)

// Create assembler and indexer to give and prepare data for the model training
val assembler = new VectorAssembler().setInputCols(colomnsOfPred).setOutputCol("features")
val indexer = new StringIndexer().setInputCol("Survived").setOutputCol("label")


// MODEL TRAINING - For each model (Decision Tree, Random Forest, Logistic Regression and Naive Bayes), a new Classifier is instanciated and inserted in a dedicated pipeline with the data managed my assembler and indexer. Then each model is trained and tested on the "realTestSet"

// Decision Tree
val dectree = new DecisionTreeClassifier()
val pipelineDT = new Pipeline().setStages(Array(assembler, indexer, dectree))
val modelDT = pipelineDT.fit(trainingDataSet)
val predictionsDT = modelDT.transform(realTestSet)

// Random Forest
val rf = new RandomForestClassifier()
val pipelineRF = new Pipeline().setStages(Array(assembler, indexer, rf))
val modelRF = pipelineRF.fit(trainingDataSet)
val predictionsRF = modelRF.transform(realTestSet)

// Logistic Regression
val lr = new LogisticRegression()
val pipelineLR = new Pipeline().setStages(Array(assembler, indexer, lr))
val modelLR = pipelineLR.fit(trainingDataSet)
val predictionsLR = modelLR.transform(realTestSet)

// Naive Bayes
val nb = new NaiveBayes()
val pipelineNB = new Pipeline().setStages(Array(assembler, indexer, nb))
val modelNB = pipelineNB.fit(trainingDataSet)
val predictionsNB = modelNB.transform(realTestSet)



// MODEL EVALUATION - For each model XX, a view of the predictionsXX is created. This is used to understand how many correct predictions has been made by the model XX, comparing them with the "TestDataSet". Thus, the accuracy is calculated and displayed for each method.

// Decision Tree
predictionsDT.createOrReplaceTempView("dtView")
val wellGuessedDT = spark.sql("SELECT count(*) FROM dtView DT JOIN testSetView T ON DT.PassengerId == T.PassengerId WHERE survived == prediction").first().getLong(0)
val accuracyDT : Double = (wellGuessedDT).toDouble/(totalRows).toDouble
println(s"The accuracy of Decision Tree is: " + accuracyDT)

// Random Forest
predictionsDT.createOrReplaceTempView("rfView")
val wellGuessedRF = spark.sql("SELECT count(*) FROM rfView RF JOIN testSetView T ON RF.PassengerId == T.PassengerId WHERE survived == prediction").first().getLong(0)
val accuracyRF : Double = (wellGuessedRF).toDouble/(totalRows).toDouble
println(s"The accuracy of Random Forest is: " + accuracyRF)

// Logistic Regression
predictionsLR.createOrReplaceTempView("lrView")
val wellGuessedLR = spark.sql("SELECT count(*) FROM lrView LR JOIN testSetView T ON LR.PassengerId == T.PassengerId WHERE survived == prediction").first().getLong(0)
val accuracyLR : Double = (wellGuessedLR).toDouble/(totalRows).toDouble
println(s"The accuracy of Logistic Regression is: " + accuracyLR)

// Naive Bayes
predictionsNB.createOrReplaceTempView("nbView")
val wellGuessedNB = spark.sql("SELECT count(*) FROM nbView NB JOIN testSetView T ON NB.PassengerId == T.PassengerId WHERE survived == prediction").first().getLong(0)
val accuracyNB : Double = (wellGuessedNB).toDouble/(totalRows).toDouble
println(s"The accuracy of Naive Bayes is: " + accuracyNB)
