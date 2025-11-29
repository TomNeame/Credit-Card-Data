#### Load Required Packages ####
library(janitor)     # Tabulation (tabyl)
library(ggplot2)     # Data visualisation
library(patchwork)
library(gridExtra)    # Arranging multiple plots
library(corrplot)     # Correlation analysis (Chap 2, Slide 37)
library(dplyr)

dat <- read.csv("BankChurners.csv")
dat <- dat[, -22:-23] # removing junk columns as instructed on source website

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
# Avg_Utilization_Ratio    : Average card utilisation ratio (Proportion of credit limit used each month)

summary(dat) # useful for the numerical columns

colSums(is.na(dat)) # no NA values to clean

# classifying the character variables as factors
dat$Attrition_Flag <- ifelse(dat$Attrition_Flag == "Attrited Customer", "Yes", "No")
dat$Attrition_Flag <- as.factor(dat$Attrition_Flag)
# applying to the rest of the character columns
dat[sapply(dat, is.character)] <- lapply(dat[sapply(dat, is.character)], as.factor)

#### EDA ####

## Outcome variable overview ##
tabyl(dat, Attrition_Flag)

ggplot(dat, aes(x = Attrition_Flag)) +     # visual plot of the number and split of medals
  geom_bar(fill = "steelblue") +
  theme_minimal() +
  labs(x = "Attrited (0=no, 1=yes)", y = "count",
       title = "Class balance: Attrition")

## Quick visual outlier scan by plotting boxplots ##
p_b1 <- ggplot(dat, aes(y = Customer_Age)) +
  geom_boxplot(fill = "lightblue") +
  theme_minimal() +
  labs(title = "Age")
p_b2 <- ggplot(dat, aes(y = Total_Trans_Ct)) +
  geom_boxplot(fill = "lightblue") +
  theme_minimal() +
  labs(title = "Total Transaction count")
p_b3 <- ggplot(dat, aes(y = Months_on_book)) +
  geom_boxplot(fill = "lightblue") +
  theme_minimal() +
  labs(title = "Months on book")
p_b4 <- ggplot(dat, aes(y = Avg_Utilization_Ratio)) +
  geom_boxplot(fill = "lightblue") +
  theme_minimal() +
  labs(title = "Average Utilisation Ratio")

(p_b1 + p_b2) / (p_b3 + p_b4)

## Numeric vs outcome ##
# boxplot x violin chart             # we plot a boxplot on top of a violin chart to give easily understandable visuals
p_box_age <- ggplot(dat, aes(x = Attrition_Flag, y = Customer_Age, fill = Attrition_Flag)) +
  geom_violin(width = 0.5, trim = FALSE, alpha = 0.7) +
  theme_minimal() +
  labs(title = NULL, x = "Attrited", y = "Age")

p_box_train <- ggplot(dat, aes(x = Attrition_Flag, y = Months_on_book, fill = Attrition_Flag)) +
  geom_violin(width = 0.5, trim = FALSE, alpha = 0.7) +
  theme_minimal() +
  labs(title = NULL, x = "Attrited", y = "Months on Book")

# Compare effects
(p_box_age / p_box_train) +
  plot_annotation(title = "Distributions of Training Hours and Age by Medal")
p_box_age
