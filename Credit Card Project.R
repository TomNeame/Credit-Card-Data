

dat <- read.csv("BankChurners.csv")
dat <- dat[, -22:-23] # removing junk columns as instructed on source website

#### Quality Checking ####

# visualising variables
head(dat) # 'Attrition_Flag' is the outcome showing whether the customer is existing or has left (Attrited)

colSums(is.na(dat)) # no NA values to clean

str(dat) # matches what is expected, may be worth turning 'Attrition_Flag' Binary
summary(dat) # useful for the numerical columns
