

dat <- read.csv("BankChurners.csv")
dat <- dat[, -22:-23] # removing junk columns as instructed on source website

#
library(highcharter)
library(dplyr)
library(viridis)
library(readr)
library(purrr)
library(class)
library(randomForest)
library(e1071)
library(pROC)
library(caret)

#### Quality Checking ####

#visualising variables

str(dat) # matches what is expected, may be worth turning 'Attrition_Flag'
# binary, as this is the outcome variable we are going to predict
dat$CLIENTNUM <- NULL

##Variables Explained:
# CLIENTNUM                : Unique Identifier for customer holding account
# Attrition_Flag           : If customer is 'Existing' or 'Attrited' (left)
# Customer_age             : Customers age in years
# Gender                   : M = Male, F = Female
# Dependant_count          : Number of dependants
# Education_Level          : Education level of the account holder
# Marital_Status           : Marital status of account holder
# Income_Category          : Annual income category of the account holder
# Card_Category            : Product variable - type of card
# Months_on_book           : Period of relationship with the bank
# Total_Relationship_Count : Total no. of products held by customer
# Months_Inactive_12_mon   : No. of months inactive in the last 12 months
# Contacts_Count_12_mon    : No. of contacts in the last 12 months
# Credit_Limit             : Credit limit on the credit card
# Total_Revolving_Bal      : Total revolving balance on the credit card
# Avg_Open_To_Buy          : Open to buy credit line (avg of last 12 months)
# Total_Amt_Chng_Q4_Q1     : Change in transaction amount (Q4 over Q1)
# Total_Trans_Amt          : Total transaction amount (last 12 months)
# Total_Trans_Ct           : Total transaction count (last 12 months)
# Total_Ct_Chng_Q4_Q1      : Change in transaction count (Q4 over Q1)
# Avg_Utilization_Ratio    : Average card utilisation ratio

summary(dat) # useful for the numerical columns

colSums(is.na(dat)) # no NA values to clean

# classifying the character variables as factors
dat$Attrition_Flag <- ifelse(dat$Attrition_Flag == "Attrited Customer", "Yes", "No")
dat$Attrition_Flag <- as.factor(dat$Attrition_Flag)
# applying to the rest of the character columns
dat[sapply(dat, is.character)] <- lapply(dat[sapply(dat, is.character)], as.factor)

str(dat)

cor_spearman <- cor(dat[, sapply(dat, is.numeric)], method = 'spearman')

# Visualising Spearman's Correlation Coefficients with a heatmap
as.matrix(data.frame(cor_spearman)) %>% 
  round(3) %>% # All values to 3 d.p.
  hchart() %>% 
  hc_add_theme(hc_theme_smpl()) %>%
  hc_title(text = "Spearman's correlation coefficients", align = "center") %>% 
  hc_legend(align = "center") %>% 
  hc_colorAxis(stops = color_stops(colors = viridis::inferno(10))) %>%
  hc_plotOptions(
    series = list(
      borderWidth = 0,
      dataLabels = list(enabled = TRUE)))



#
par(mfrow=c(2,2)) # Set up a 2x2 plotting grid

hist(dat$Customer_Age, 
     main="Histogram of Customer Age", 
     xlab="Age", col="lightblue", border="white")

hist(dat$Total_Trans_Amt, 
     main="Histogram of Trans Amount", 
     xlab="Amount", col="lightgreen", border="white")

hist(dat$Total_Trans_Ct, 
     main="Histogram of Trans Count", 
     xlab="Count", col="salmon", border="white")

hist(dat$Credit_Limit, 
     main="Histogram of Credit Limit", 
     xlab="Credit Limit", col="gray", border="white")

par(mfrow=c(1,2))

boxplot(Total_Trans_Ct ~ Attrition_Flag, data=dat,
        main="Trans Count by Attrition",
        ylab="Total Transaction Count",
        col=c("orange", "lightblue"))

boxplot(Total_Revolving_Bal ~ Attrition_Flag, data=dat,
        main="Revolving Bal by Attrition",
        ylab="Total Revolving Balance",
        col=c("orange", "lightblue"))


# removing Avg_Open_To_Buy
dat$Avg_Open_To_Buy <- NULL 
#creating new dataset for no very high correlation variables
dat_no_corr <- dat
# removing Customer_Age to be used for a no-multicollinearity model
dat_no_corr$Customer_Age <- NULL
# combining both transaction variables into one predictor for a no-multicollinearity model
dat_no_corr$Avg_Trans_Amt <- dat$Total_Trans_Amt / dat$Total_Trans_Ct
# removing the other transaction variables
dat_no_corr$Total_Trans_Amt <- NULL
dat_no_corr$Total_Trans_Ct <- NULL


#Splitting the data

set.seed(123) 
train_index <- sample(1:nrow(dat), 0.7 * nrow(dat)) 

#Set 1: Original Data (For Trees, RF, SVM)
train_data <- dat[train_index, ]
test_data <- dat[-train_index, ]
test_y <- test_data$Attrition_Flag

#Set 2: No-Correlation Data (For GLM, LDA, etc)
train_data_nc <- dat_no_corr[train_index, ]
test_data_nc <- dat_no_corr[-train_index, ]
test_y_nc <- test_data_nc$Attrition_Flag

#GLM

#Fitting the model on the non-correlated data
glm_fit <- glm(Attrition_Flag ~ ., 
               data = train_data_nc, 
               family = binomial)

glm_probs <- predict(glm_fit, test_data_nc, type = "response")

#Figuring out which label is 0 and which is 1
level_neg <- levels(test_y_nc)[1] 
level_pos <- levels(test_y_nc)[2] 

#Standard 0.5 cutoff
pred_class <- ifelse(glm_probs > 0.5, level_pos, level_neg)

#Make sure everything is a factor before feeding to confusionMatrix
actual_factor <- factor(test_y_nc)
predicted_factor <- factor(pred_class, levels = levels(actual_factor))

#Check results
cm_glm <- confusionMatrix(data = predicted_factor, 
                          reference = actual_factor, 
                          positive = pos_label)
print(cm_glm)

#Trying a stricter threshold (0.3) to catch more churners
threshold <- 0.3
pred_class_sensitive <- ifelse(glm_probs > threshold, level_pos, level_neg)
predicted_factor_sens <- factor(pred_class_sensitive, levels = levels(actual_factor))

cm_sens <- confusionMatrix(data = predicted_factor_sens, 
                           reference = actual_factor, 
                           positive = pos_label)
print(cm_sens)

#Checking AUC
roc_glm <- roc(test_y_nc, glm_probs)
auc(roc_glm)
plot(roc_glm, main="ROC Curve - GLM")


#The output matrix tells us that the model correctly predicts 183 customers to have left
#and correctly predicts 2499 customers to be existing. However it falsely predicts 62 customers
#to have left but are actually existing. Furthermore it predicts 295 customers to be existing
#whereas they have actually left, this is higher than would be desired.
#Nevertheless the model has an overall accuracy of 88.3% which is high. 
#However this could just be largely down to the data consisting or primarily non churners
#and the model will tend to predict the majority class.

#KNN
#KNN needs numbers, not factors, so we convert everything
train_x_knn <- model.matrix(Attrition_Flag ~ ., data = train_data_nc)[,-1]
test_x_knn <- model.matrix(Attrition_Flag ~ ., data = test_data_nc)[,-1]
train_y_knn <- train_data_nc$Attrition_Flag

set.seed(1)

#Simple loop to see which K value works best
accuracy_results <- numeric(20)
for(i in 1:20){
  knn_temp <- knn(train_x_knn, test_x_knn, train_y_knn, k = i)
  accuracy_results[i] <- mean(knn_temp == test_y_nc)
}

best_k <- which.max(accuracy_results)

#Comparing the "best" K vs a generic K=3
knn_best_pred <- knn(train_x_knn, test_x_knn, train_y_knn, k = best_k)
knn_3 <- knn(train_x_knn, test_x_knn, train_y_knn, k = 3)

#Confusion Matrix for K=3
predicted_factor_knn <- factor(knn_3, levels = levels(actual_factor))

cm_knn <- confusionMatrix(data = predicted_factor_knn, 
                          reference = actual_factor, 
                          positive = pos_label)

print(cm_knn)

#The model chose k=19 as the best for accuracy (83.8%), this resulted in the model correctly identifying
#66 customers leaving and missing 412. 
#If we compare to k=3, the accuracy falls (81.7%), but the model correctly predicted 177 customers
#leaving and missed 361, which is better. It did however result in more false positives with 196.



#Random Forest

set.seed(1)
#Growing 500 trees using the original dataset
rf_fit <- randomForest(Attrition_Flag ~ ., 
                       data = train_data, 
                       mtry = 4, 
                       importance = TRUE)
print(rf_fit)

rf_pred_class <- predict(rf_fit, test_data, type = "class")
predicted_factor_rf <- factor(rf_pred_class, levels = levels(actual_factor))

cm_rf <- confusionMatrix(data = predicted_factor_rf, 
                         reference = actual_factor, 
                         positive = pos_label)
print(cm_rf)

#Checking which variables actually matter
varImpPlot(rf_fit, main="Random Forest: Variable Importance")

#ROC / AUC for Random Forest
#Need to be careful to grab the right column for probabilities
rf_prob_matrix <- predict(rf_fit, test_data, type = "prob")

#Finding the column that matches our 'pos_label' (Churn)
churn_col <- intersect(pos_label, colnames(rf_prob_matrix))
if(length(churn_col) == 0) churn_col <- colnames(rf_prob_matrix)[1]

rf_probs <- rf_prob_matrix[, churn_col]
roc_rf <- roc(test_data$Attrition_Flag, rf_probs)

print(auc(roc_rf))
plot(roc_rf, main = paste("ROC Curve - Random Forest (AUC =", round(auc(roc_rf), 3), ")"))

#SVM

#Initial run to see how it does
svm_fit <- svm(Attrition_Flag ~ ., 
               data = train_data, 
               kernel = "radial", 
               cost = 1, 
               scale = TRUE)

svm_pred_class <- predict(svm_fit, test_data)
predicted_factor_svm <- factor(svm_pred_class, levels = levels(actual_factor))

cm_svm <- confusionMatrix(data = predicted_factor_svm, 
                          reference = actual_factor, 
                          positive = pos_label)
print(cm_svm)

#Retraining with probability turned on so we can do AUC
svm_fit_prob <- svm(Attrition_Flag ~ ., 
                    data = train_data, 
                    kernel = "radial", 
                    cost = 1, 
                    scale = TRUE,
                    probability = TRUE)

#Grabbing the probabilities
svm_pred_prob <- predict(svm_fit_prob, test_data, probability = TRUE)
prob_matrix <- attr(svm_pred_prob, "probabilities")

#Finding the right column again
match_col <- intersect(pos_label, colnames(prob_matrix))
target_col <- if(length(match_col) > 0) match_col[1] else colnames(prob_matrix)[1]

churn_prob <- prob_matrix[, target_col]

roc_svm <- roc(test_data$Attrition_Flag, churn_prob)
cat("SVM AUC Score:", auc(roc_svm), "\n")
plot(roc_svm, main = paste("ROC Curve - SVM (AUC =", round(auc(roc_svm), 3), ")"))





