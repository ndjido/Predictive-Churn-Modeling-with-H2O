#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  H2O Demo : Predictive Churn Model
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

library(h2o)

#---------------------------------------
# Start H2o cluster
#---------------------------------------
h2o.init()

#---------------------------------------
# Loading data
#---------------------------------------
# hint: can also load data from a FOLDER or from HDFS with "h2o.importFile"
df_churn.hex <- h2o.importFile("churn.all.csv")
summary(df_churn.hex)

# Convert to R Data.Frame => to work with R functions
# limitation: R can't handle a very large dataset
df_churn <- as.data.frame(df_churn.hex)

# Summary on a large Dataset 
h2o.table(df_churn.hex[,c(5,6)])

# Feature Extraction
df_churn.hex[, "churned"] <- as.factor(df_churn.hex[, "churned"])
df_churn.hex[, "international_plan"] <- as.factor(df_churn.hex[, "international_plan"])
df_churn.hex[, "voice_mail_plan"] <- as.factor(df_churn.hex[, "voice_mail_plan"])

summary(df_churn.hex)

#---------------------------------------
# Split Dataset
#---------------------------------------
df_churn.split = h2o.splitFrame(data=df_churn.hex, ratio=.7)
df_churn.train = df_churn.split[[1]]
df_churn.test = df_churn.split[[2]]

summary(df_churn.train)
summary(df_churn.test)

#---------------------------------------
# Bulding Random Forest modele
#---------------------------------------
model <- h2o.randomForest(
          y = "churned",
          x = c("account_length", "international_plan", "voice_mail_plan", 
                "number_vmail_messages", "total_day_minutes", "total_day_calls", "total_day_charge", 
                "total_eve_minutes", "total_eve_calls", "total_eve_charge", "total_night_minutes", 
                "total_night_calls", "total_night_charge", "total_intl_minutes", "total_intl_calls", 
                "total_intl_charge", "number_customer_service_calls"),
          training_frame = df_churn.train,
          ntrees = 100,
          balance_classes = TRUE,
          stopping_metric = "AUC",
          categorical_encoding = "Enum")

summary(model)

#---------------------------------------
# Perf. on train set
#---------------------------------------
(perf.train <- h2o.performance(model, df_churn.train))
h2o.auc(perf.train)

#---------------------------------------
# Perf. on test set
#---------------------------------------
(perf.test <- h2o.performance(model, df_churn.test))
h2o.auc(perf.test)

#---------------------------------------
# Retrieving Scores for test set
#---------------------------------------
(pred.test <- h2o.predict(model, df_churn.test))

# Lift extraction
## Lift on train set
h2o.gainsLift(model, df_churn.train)

## Lift on test set
h2o.gainsLift(model, df_churn.test)

# Saving th H2OModel
h2o.saveModel(model, path = "rf_model.hex", force = TRUE)

#---------------------------------------
# Download model as a POJO
#---------------------------------------
sink("rf_pojo.java")
h2o.download_pojo(model) # hint: use path param if the file happens to be large
sink()

