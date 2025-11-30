#### Install and Load Required Packages ####
install.packages("tree")
install.packages("gbm")
library(janitor)     # Tabulation (tabyl)
library(ggplot2)     # Data visualisation
library(patchwork)   # Multiple plots
library(gridExtra)   # Arranging multiple plots
library(corrplot)    # Correlation analysis (Chap 2, Slide 37)
library(dplyr)       # Functions
library(tidyr)       # Neat plots
library(MASS)        # Standard package for LDA in R
library(caret)       # For data splitting
library(pROC)        # For ROC curves
library(tree)        # For CART
library(gbm)         # For Boosting

dat <- read.csv("BankChurners.csv")
dat <- dat[, -22:-23] # removing junk columns as instructed on source website

#### Quality Checking ####

# visualising variables

str(dat) # matches what is expected, may be worth turning 'Attrition_Flag'
# binary, as this is the outcome variable we are going to predict
dat$CLIENTNUM <- NULL # just personal identifier, not useful in analysis

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
# Avg_Open_To_Buy          : Avg amount of credit available (Credit Limit - Current Balance) over last 12 mos
# Total_Amt_Chng_Q4_Q1     : Change in transaction amount (Q4 over Q1)
# Total_Trans_Amt          : Total transaction amount (last 12 months)
# Total_Trans_Ct           : Total transaction count (last 12 months)
# Total_Ct_Chng_Q4_Q1      : Change in transaction count (Q4 over Q1)
# Avg_Utilization_Ratio    : Average card utilisation ratio (Proportion of credit limit used each month)

summary(dat) # useful as a snapshot for each variable
# we can see that there is a proportion of 'Unknown' education levels, However it is not significant so we will leave it in

colSums(is.na(dat)) # no NA values to clean

# classifying the character variables as factors
dat$Attrition_Flag <- ifelse(dat$Attrition_Flag == "Attrited Customer", "Yes", "No")
dat$Attrition_Flag <- as.factor(dat$Attrition_Flag)
# applying it generally to the rest of the character columns
dat[sapply(dat, is.character)] <- lapply(dat[sapply(dat, is.character)], as.factor)

#### EDA ####

## Outcome variable overview ##
tabyl(dat, Attrition_Flag) # we see that only 16% of the dataset is attrited. This does make 
                           # it harder to predict, as there is less information about 'Attrited', however due
                           # to a fairly large amount of observations this will still be suitable

## OUTCOME VARIABLE DISTRIBUTION PLOT ##
ggplot(dat, aes(x = Attrition_Flag)) +     
  geom_bar(fill = "steelblue") +
  theme_minimal() +
  labs(x = "Attrited (0=no, 1=yes)", y = "count",
       title = "Class balance: Attrition")
# visualises the tabyl in an easier way

## NUMERICAL VARIABLES: BOXPLOT ##

# Splitting up into numerical data, to be used later as well
numeric_data <- dat %>% 
  dplyr::select(Attrition_Flag, where(is.numeric)) %>%  # need to mention 'dplyr' otherwise it gets confused with 'MASS'
  pivot_longer(cols = -Attrition_Flag, names_to = "Variable", values_to = "Value")
                                       # naming generically as 'variables' and 'values' to make one, neat plot

p_boxplot <- ggplot(numeric_data, aes(x = Attrition_Flag, y = Value, fill = Attrition_Flag)) +
  geom_boxplot(outlier.colour = "red", outlier.size = 1, alpha = 0.7) +
  scale_fill_manual(values = c("steelblue", "orange"), name = "Attrition Status") +
  facet_wrap(~ Variable, scales = "free", ncol = 4) + # Separate panels, free scales for plot neatness
  theme_minimal() +
  theme(legend.position = "top", # Single legend at top
        strip.text = element_text(face = "bold", size = 10)) +
  labs(title = "Boxplots of Numeric Variables by Attrition Status",
       x = "Attrition Status", y = "Value")

print(p_boxplot)
# we see that with a lot of the variables there is not much difference between, however
# columns like 'Avg_Utilization_Ratio', 'Months_Inactive_12_months', 'Total_Relationship_Count',
# and 'Total_Revolving_Balance' show significant differences between people who have attrited
# and people which haven't. This shows these may be useful predictors when modelling

## NUMERICAL VARIABLES: OVERLAPPING DENSITY PLOTS ##

# Create density plot
p_density <- ggplot(numeric_data, aes(x = Value, fill = Attrition_Flag)) +
  geom_density(alpha = 0.5, color = NA) + # Transparent overlap to see difference
  scale_fill_manual(values = c("steelblue", "orange"), name = "Attrition Status") +
  facet_wrap(~ Variable, scales = "free", ncol = 4) + # Separate panels, free scales again
  theme_minimal() +
  theme(legend.position = "top", # Single legend at top
        axis.text.y = element_blank(), # Remove y-axis text (density values vary)
        axis.ticks.y = element_blank(),
        strip.text = element_text(face = "bold", size = 10)) +
  labs(title = "Distribution of Numeric Variables by Attrition Status",
       x = "Value", y = "Density")

print(p_density)
# we see once again that attrited individuals have much lower transaction amounts and
# transaction counts, which is expected, as people using their acount less would generally
# be more likely to leave. We also see interestingly, people with less relationships are
# more likely to be attrited. This may be due to them being less dependent on credit

## CATEGORICAL ANALYSIS: BAR CHART ##

plot_categorical_bars <- function(data, target_col, ncol = 3) {
  
  cat_cols <- names(data)[sapply(data, is.factor)]
  cat_cols <- cat_cols[cat_cols != target_col]
  
  plot_list <- lapply(cat_cols, function(col_name) {
    ggplot(data, aes(x = .data[[col_name]], fill = .data[[target_col]])) +
      geom_bar(position = "fill") + 
      scale_fill_manual(values = c("steelblue", "orange")) +
      theme_minimal() +
      theme(axis.text.x = element_text(angle = 45, hjust = 1),
            legend.position = "none",
            plot.title = element_text(size = 10, face = "bold")) +
      labs(title = col_name, x = "", y = "Proportion")
  })
  
  do.call(grid.arrange, c(plot_list, ncol = ncol))
}

plot_categorical_bars(dat, "Attrition_Flag", ncol = 3)
# The 'Education_Level' chart shows that the more educated individuals, 'Doctorate',
# and 'Post-Graduate' are more likely to leave, as well as those with the 'Platinum'
# card. It also shows that those with the income range of $60K-$80K are the least
# likely to leave

## CORRELATION MATRIX ## 

num_data <- dat %>% 
  dplyr::select(where(is.numeric))
cor_matrix <- cor(num_data)

corrplot(cor_matrix, method = "color", type = "upper", 
         order = "hclust", 
         tl.col = "black", tl.cex = 0.6,
         addCoef.col = "black", number.cex = 0.5,
         title = "Feature Correlation Matrix", mar=c(0,0,2,0))
# Some of the variables are very, sometimes perfectly, correlated with eachother.
# this is because some variables are very closely linked to another, for example:
# 'Avg_Open_To_Buy' is equal to the Credit_Limit - current balance of the account holder.
# so for the individuals who don't use their card much, their credit limit will 
# essentially be equal to their Avg_Open_To_Buy. This causes multicollinearity, which 
# breaks some models, so we will remove 'Avg_Open_To_Buy'.
#
# Snother potential problem is transaction behaviour:
# There is a very strong correlation (0.81) between `Total_Transaction_Amt` and `Total_Trans_Ct`,
# this makes sense, as people who use their card more often (Ct) usually spend more
# money in total (Amt). We will keep both variables in, however in models like logistic 
# regression, having two highly correlated variables makes the model unstable so
# we will have to take this into account. We can combine them to form one predictor
# in the models where multicollinearity is an issue.
#
# Another potential problem is that `Customer_Age` and `Months_on_book` are also 
# highly correlated (0.79). This makes sense, as older customers generally have held
# their account for longer. Similarly these two variables provide overlapping information.
# We will remove the customer age variable, as the months on book variable
# will be much more telling on how likely the customer is to leave.

## DATA MANIPULATION ## 

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

# re-checking the correlation matrix

num_data <- dat_no_corr %>% 
  dplyr::select(where(is.numeric))
cor_matrix <- cor(num_data)

corrplot(cor_matrix, method = "color", type = "upper", 
         order = "hclust", 
         tl.col = "black", tl.cex = 0.6,
         addCoef.col = "black", number.cex = 0.5,
         title = "Feature Correlation Matrix", mar=c(0,0,2,0))
# there is now no out-of-the-ordinary correlation

str(dat_no_corr)

#### LDA ####

## Split the Data (Using caret) ##
set.seed(2025) # Ensure reproducibility
trainIndex <- createDataPartition(dat_no_corr$Attrition_Flag, p = 0.7, list = FALSE)
train_data <- dat_no_corr[trainIndex, ]
test_data  <- dat_no_corr[-trainIndex, ]

## Fit the LDA Model ##
# We use the dot (.) to include all remaining variables in dat_no_corr as predictors
lda_model <- lda(Attrition_Flag ~ ., data = train_data)

# Output the model details
# "Prior probabilities of groups" shows the class balance in the training set.
# "Coefficients of linear discriminants" shows how each variable contributes to the separation.
print(lda_model)

## Make Predictions on Test Data ##
# The predict function for LDA returns a list containing:
# - class: The predicted class (Yes/No)
# - posterior: The probability of belonging to each class
lda_predictions <- predict(lda_model, newdata = test_data)

## Evaluation ##

# Confusion Matrix
# This provides Accuracy, Sensitivity (Recall), and Specificity
cm_lda <- confusionMatrix(lda_predictions$class, test_data$Attrition_Flag, positive = "Yes")
print(cm_lda)

# ROC Curve and AUC
# We use the posterior probability of "Yes" (Attrited) for the ROC curve
roc_lda <- roc(test_data$Attrition_Flag, lda_predictions$posterior[, "Yes"], levels = c("No", "Yes"), direction = "<")

# Plotting the ROC Curve
plot(roc_lda, col = "steelblue", lwd = 2, main = "ROC Curve: LDA")
auc_lda <- auc(roc_lda)
text(0.5, 0.5, paste("AUC =", round(auc_lda, 3)), col = "steelblue", font = 2)

# Analysis:
# The model correctly classifies 87.3% of its customers, however the no information rate is 83.9%,
# meaning that if we purely guessed 'no' for every customer, we would only lose 3.4% of accuracy
# compared with the LDA model. The sensitivity is 33%, meaning the model only predicts one in 
# three attrited customers. The model is excellent at identifying customers that are staying,
# however this is not important for our goal of predicting who is leaving.
# 
# Conclusion:
# LDA assumes the predictors are normally distributed and share a common covariance matrix.
# our EDA shows some skewed distributions, and the low sensitivity suggests that a linear
# boundary is inefficient to capture the customers that are a higher risk of leaving.

#### Classification Tree (CART) ####

## Data Splitting (Using the full 'dat' dataset, as multicollinearity doesn't matter here) ##
# We use the same seed to ensure we pick the same customers as the LDA split for fair comparison
set.seed(2025) 
trainIndex <- createDataPartition(dat$Attrition_Flag, p = 0.7, list = FALSE)
train_data_full <- dat[trainIndex, ]
test_data_full  <- dat[-trainIndex, ]

## Fit the Initial Tree ##
# The tree will now scan ALL variables in 'dat' to find the best splits
tree_model <- tree(Attrition_Flag ~ ., data = train_data_full)

# Summary of the tree 
summary(tree_model) # this means it was a good call to use the dataset with the correlated 
                    # variables, as they are both used in the tree construction

## Visualise the Unpruned Tree ##
plot(tree_model)
text(tree_model, pretty = 0) 
title("Unpruned Classification Tree")

## Cross-Validation for Pruning ##
# Using misclassification error as the guide
set.seed(2025)
cv_tree <- cv.tree(tree_model, FUN = prune.misclass)

# Visualise CV Results
plot(cv_tree$size, cv_tree$dev, type = "b", 
     xlab = "Tree Size", ylab = "CV Misclassification Errors",
     main = "CV Error vs Tree Size")

## Fit the Pruned Tree ##
# Select the size that minimizes deviance (error)
best_size <- cv_tree$size[which.min(cv_tree$dev)]
best_size
# 11 has the smallest error, however 9 has a negligibly larger error, so we go with
# the simpler model.

# Prune the tree
pruned_model <- prune.misclass(tree_model, best = 9)

# Visualise Pruned Tree
plot(pruned_model)
text(pruned_model, pretty = 0)
title("Pruned Classification Tree")

## Final Evaluation on Test Data ##
tree_pred_class <- predict(pruned_model, test_data_full, type = "class")
tree_pred_prob <- predict(pruned_model, test_data_full, type = "vector")

# Confusion Matrix
cm_tree <- confusionMatrix(tree_pred_class, test_data_full$Attrition_Flag, positive = "Yes")
print(cm_tree)

# ROC Curve
roc_tree <- roc(test_data_full$Attrition_Flag, tree_pred_prob[, "Yes"], levels = c("No", "Yes"), direction = "<")
plot(roc_tree, col = "darkgreen", lwd = 2, main = "ROC Curve: Pruned Tree")
auc_tree <- auc(roc_tree)
text(0.5, 0.5, paste("AUC =", round(auc_tree, 3)), col = "darkgreen", font = 2)

# Analysis:
# The CART model captures 71.7% (sensitivity) of churners, compared to only 33% in LDA.
# By allowing for non-linear boundaries the tree successfully found the high-risk variables
# that the LDA couldn't.
# The AUC of 0.94 is very strong, and shows it understands the attrited and remaining
# customers well.
#
# Conclusion:
# The very first split of the tree is `Total_Trans_Ct`. This shows that it is the
# single strongest predictor of attrited customers - customers using their card fewer
# than 58 times in the last year are immedietly flagged as higher risk of leaving.
# The fact that both `Total_Trans_Ct` and `Total_Trans_Amt` are included in the tree
# shows that using the larger dataset with the correlated variables was a good idea,
# as both are useful in model predictions.

#### Boosting (Gradient Boosted Machines) ####

## Data Prep for GBM ##
# GBM requires the target to be binary (0/1) rather than a factor (No/Yes)
# We create a copy of the training data for GBM
train_data_gbm <- train_data_full
train_data_gbm$Attrition_Flag <- ifelse(train_data_gbm$Attrition_Flag == "Yes", 1, 0)

test_data_gbm <- test_data_full
test_data_gbm$Attrition_Flag <- ifelse(test_data_gbm$Attrition_Flag == "Yes", 1, 0)

set.seed(2025)

## Fit the Boosting Model ##
# distribution = "bernoulli" is used for binary classification
# n.trees = 5000 is a standard starting point
# interaction.depth = 4 allows for complex variable interactions
# shrinkage = 0.01 (learning rate) controls overfitting
boost_model <- gbm(Attrition_Flag ~ ., 
                   data = train_data_gbm, 
                   distribution = "bernoulli", 
                   n.trees = 5000, 
                   interaction.depth = 4, 
                   shrinkage = 0.01,
                   verbose = FALSE)

## Summary & Variable Importance ##
# This produces the "Relative Influence" plot
summary(boost_model)
# This shows that `Total_Trans_Ct` (29), `Total_Trans_Amt` (21.7), and `Total_Revolving_Bal` (15.6)
# account for approximately 66% of the models predictive power. This shows that 
# customers leaving is a behavioral event rather than a demographic one (age, gender, marital status).
# This means a retention strategy from the bank can't target a specific demographic,
# but rather be activity based.

## Make Predictions ##
# n.trees should match the trained model
# type = "response" gives probabilities
boost_probs <- predict(boost_model, newdata = test_data_gbm, n.trees = 5000, type = "response")

## Convert Probabilities to Class Labels ##
# We use a threshold of 0.5
boost_pred_class <- ifelse(boost_probs > 0.5, "Yes", "No")

## Evaluation ##
# Convert back to factors for confusionMatrix
cm_boost <- confusionMatrix(as.factor(boost_pred_class), test_data_full$Attrition_Flag, positive = "Yes")
print(cm_boost)

# ROC Curve
roc_boost <- roc(test_data_full$Attrition_Flag, boost_probs, levels = c("No", "Yes"), direction = "<")

plot(roc_boost, col = "purple", lwd = 2, main = "ROC Curve: Boosting (GBM)")
auc_boost <- auc(roc_boost)
text(0.5, 0.5, paste("AUC =", round(auc_boost, 3)), col = "purple", font = 2)

# Analysis:
# The sensitivity has again increased. Starting at 33% (LDA), then 71.7% (CART), and 
# now 88.1% when using boosting. This shows that the Boosting model successfully identifies
# 16% more of the customers which are at risk of leaving than the single tree did,
# successfully predicting 430 attrited customers (true positive), while only missing 
# 58 (false negative).
#
# Conclusion:
# The AUC value of 0.993 is very high. This means the model has nearly perfectly
# separated the signals for 'attrited' vs. 'existing' customers.