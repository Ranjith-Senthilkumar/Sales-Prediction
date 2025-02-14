from google.colab import files
uploaded = files.upload()

%%R
library(tidyverse)
library(caret)
library(lubridate)

%%R
df <- read.csv("sales_data_sample.csv")

%%R
df$ORDERDATE <- as.Date(df$ORDERDATE, format="%m/%d/%Y")  # Convert ORDERDATE to Date format
df$YEAR <- year(df$ORDERDATE)  # Extract Year
df$MONTH <- month(df$ORDERDATE)  # Extract Month

# Remove non-useful columns
df <- df %>% select(-c(ORDERNUMBER, PRODUCTCODE, ORDERDATE, STATUS, COUNTRY, TERRITORY, DEALSIZE))

%%R
set.seed(123)  # For reproducibility
trainIndex <- createDataPartition(df$SALES, p=0.8, list=FALSE)
train_data <- df[trainIndex, ]
test_data <- df[-trainIndex, ]

%%R
model <- lm(SALES ~ QUANTITYORDERED + PRICEEACH + MONTH_ID + YEAR_ID + MSRP, data=train_data)
summary(model)  # Check model performance

%%R
predictions <- predict(model, test_data)

%%R
rmse <- sqrt(mean((predictions - test_data$SALES)^2))  # RMSE
mae <- mean(abs(predictions - test_data$SALES))  # MAE

cat("Root Mean Squared Error (RMSE):", rmse, "\n")
cat("Mean Absolute Error (MAE):", mae, "\n")

%%R
# Step 8: Prepare new data for future predictions (Year 2006)
# Assuming you're predicting for every month in 2006

# Create a data frame with the months of 2006
future_data <- data.frame(
  QUANTITYORDERED = rep(100, 12),  # Example: Assume quantity ordered is 100 for each month
  PRICEEACH = rep(50, 12),         # Example: Assume price each is 50 for each month
  MONTH_ID = 1:12,                    # 12 months (January to December)
  YEAR_ID = rep(2006, 12),            # Year 2006
  MSRP = rep(120, 12)              # Example: Assume MSRP is 120 for each month
)

# Step 9: Make predictions for the entire year 2006
predictions_2006 <- predict(model, future_data)

# Step 10: Output the predicted sales for each month of 2006
cat("Predicted Sales for each month in 2006:\n")
print(predictions_2006)

# Step 11: Optionally, sum the predictions to get total sales for 2006
total_sales_2006 <- sum(predictions_2006)
cat("\nTotal Predicted Sales for 2006:", total_sales_2006, "\n")

