#### Load Required Packages ####
library(janitor)     # Tabulation (tabyl)
library(ggplot2)     # Data visualisation
library(patchwork)   # Multiple plots
library(gridExtra)   # Arranging multiple plots
library(corrplot)    # Correlation analysis (Chap 2, Slide 37)
library(dplyr)       # Functions
library(tidyr)       # Neat plots

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
  select(Attrition_Flag, where(is.numeric)) %>%
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

num_data <- dat %>% select(where(is.numeric))
cor_matrix <- cor(num_data)

corrplot(cor_matrix, method = "color", type = "upper", 
         order = "hclust", 
         tl.col = "black", tl.cex = 0.6,
         addCoef.col = "black", number.cex = 0.5,
         title = "Feature Correlation Matrix", mar=c(0,0,2,0))
# some of the variables are very, sometimes perfectly, correlated with eachother.
# this is because some variables are very closely linked to another, for example:
# 'Avg_Open_To_Buy' is equal to the Credit_Limit - current balance of the account holder.
# so for the individuals who don't use their card much, their credit limit will 
# essentially be equal to their Avg_Open_To_Buy. This causes multicollinearity, which 
# breaks some models, so we will remove 'Avg_Open_To_Buy'
#
# another potential problem is transaction behaviour:
# there is a very strong correlation (0.81) between `Total_Transaction_Amt` and `Total_Trans_Ct`,
# this makes sense, as people who use their card more often (Ct) usually spend more
# money in total (Amt). We will keep both variables in, however in models like logistic 
# regression, having two highly correlated variables makes the model unstable so
# we will have to take this into account
#
# another potential problem is that `Customer_Age` and `Months_on_book` are also 
# highly correlated (0.79). This makes sense, as older customers generally have held
# their account for longer. Similarly these two variables provide overlapping information.
# We will have to check the importance of customer age, as the months on book variable
# may be much more telling on how likely the customer is to leave
