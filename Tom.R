

dat <- read.csv("BankChurners.csv")
dat <- dat[, -22:-23] # removing junk columns as instructed on source website

#
library(highcharter)
library(dplyr)
library(viridis)
library(readr)
library(purrr)
library(class)

#### Quality Checking ####

# visualising variables

str(dat) # matches what is expected, may be worth turning 'Attrition_Flag'
# binary, as this is the outcome variable we are going to predict
dat$CLIENTNUM <- NULL

## Variables Explained:
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
      boderWidth = 0,
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

# Set 1: Original Data (For Trees, RF, SVM)
train_data <- dat[train_index, ]
test_data <- dat[-train_index, ]
test_y <- test_data$Attrition_Flag

# Set 2: No-Correlation Data (For GLM, LDA, etc)
train_data_nc <- dat_no_corr[train_index, ]
test_data_nc <- dat_no_corr[-train_index, ]
test_y_nc <- test_data_nc$Attrition_Flag

cat("Training Data Size:", nrow(train_data), "\n")
cat("Test Data Size:", nrow(test_data), "\n")
cat("Training Data Size:", nrow(train_data_nc), "\n")
cat("Test Data Size:", nrow(test_data_nc), "\n")


#GLM

glm_fit <- glm(Attrition_Flag ~ ., 
               data = train_data_nc, 
               family = binomial)

glm_probs <- predict(glm_fit, test_data_nc, type = "response")

# Determine which class corresponds to probability > 0.5
# R's glm models the probability of the SECOND factor level.
# We dynamically get the level names from the data to avoid "Accuracy: 0" due to label mismatches
# (e.g., if your data has "No/Yes" but we predicted "Attrited/Existing")
level_neg <- levels(test_y_nc)[1] # The reference class (Prob < 0.5)
level_pos <- levels(test_y_nc)[2] # The target class (Prob > 0.5)

pred_class <- ifelse(glm_probs > 0.5, level_pos, level_neg)

cat("Confusion Matrix (GLM):\n")
table(Predicted = pred_class, Actual = test_y_nc)
cat("Accuracy:", mean(pred_class == as.character(test_y_nc)), "\n")

#The output matrix tells us that the model correctly predicts 183 customers to have left
#and correctly predicts 2499 customers to be existing. However it falsely predicts 62 customers
#to have left but are actually existing. Furthermore it predicts 295 customers to be existing
#whereas they have actually left, this is higher than would be desired.
#Nevertheless the model has an overall accuracy of 88.3% which is high. 
#However this could just be largely down to the data consisting or primarily non churners
#and the model will tend to predict the majority class.

#KNN

# KNN requires NUMERIC input only. 
# We use model.matrix to turn factors into numbers. 
# [,-1] removes the Intercept column automatically created by model.matrix
train_x_knn <- model.matrix(Attrition_Flag ~ ., data = train_data_nc)[,-1]
test_x_knn <- model.matrix(Attrition_Flag ~ ., data = test_data_nc)[,-1]
train_y_knn <- train_data_nc$Attrition_Flag

set.seed(1)

# Loop to find the BEST K (from 1 to 20)
accuracy_results <- numeric(20)
for(i in 1:20){
  knn_temp <- knn(train_x_knn, test_x_knn, train_y_knn, k = i)
  accuracy_results[i] <- mean(knn_temp == test_y_nc)
}

# Identify best K
best_k <- which.max(accuracy_results)
cat("Best K found:", best_k, "with Accuracy:", accuracy_results[best_k], "\n")

# Run Final Model with Best K
knn_best_pred <- knn(train_x_knn, test_x_knn, train_y_knn, k = best_k)

table(Predicted = knn_best_pred, Actual = test_y_nc)
cat("Final KNN Accuracy:", mean(knn_best_pred == test_y_nc), "\n")

knn_3 <- knn(train_x_knn, test_x_knn, train_y_knn, k = 3)

table(Predicted = knn_3, Actual = test_y_nc)
cat("Final KNN Accuracy:", mean(knn_3 == test_y_nc), "\n")

#The model chose k=19 as the best for accuracy (83.8%), this resulted in the model correctly identifying
#66 customers leaving and missing 412. 
#If we compare to k=3, the accuracy falls (81.7%), but the model correctly predicted 177 customers
#leaving and missed 361, which is better. It did however result in more false positives with 196.

