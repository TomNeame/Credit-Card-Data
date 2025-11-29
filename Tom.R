

dat <- read.csv("BankChurners.csv")
dat <- dat[, -22:-23] # removing junk columns as instructed on source website

#
library(highcharter)
library(dplyr)
library(viridis)
library(readr)
library(purrr)

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


cols_to_plot <- c("Gender", "Education_Level", "Marital_Status", 
                  "Income_Category", "Card_Category")

# 3. Create a list of charts using a loop (map)
chart_list <- map(cols_to_plot, function(col_name) {
  
  # Prepare data for this specific column
  plot_data <- dat %>%
    group_by(.data[[col_name]], Attrition_Flag) %>%
    count() %>%
    ungroup()
  
  # Create the chart
  hchart(plot_data, 
         type = "column",
         hcaes(x = !!sym(col_name), y = n, group = Attrition_Flag)) %>%
    
    # Dynamic Titles
    hc_title(text = gsub("_", " ", col_name), align = "center",
             style = list(fontSize = "14px")) %>% 
    
    # Custom Colors: Attrited (Orange), Existing (Blue)
    hc_colors(c("#FF8C00", "#1f77b4")) %>% 
    
    # Simplified Axis formatting for small grid
    hc_yAxis(title = list(text = "")) %>%
    hc_xAxis(title = list(text = "")) %>%
    
    # Shared Theme
    hc_add_theme(hc_theme_smpl()) %>%
    hc_legend(enabled = FALSE) # Hide legend to save space (it's redundant)
})

# 4. Display them all together in a grid
# ncol = 2 means two charts per row
hw_grid(chart_list, ncol = 2, rowheight = 400)

