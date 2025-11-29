

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
 # Avg_Utilization_Ratio    : Average card utilisation ratio

summary(dat) # useful for the numerical columns

colSums(is.na(dat)) # no NA values to clean



GLM T
LDA N
KLM T
CART N
RF T
Boosting - GBM N
Support vector T
Deep learning N

