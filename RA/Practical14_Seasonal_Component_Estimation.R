# Practical 14: Estimation of Seasonal Component
# Aim: To estimate the seasonal index for each month using two different
# methods, and compare them.
# Dataset: AirPassengers.

data("AirPassengers")

## 14(a): Simple Average Method
df = data.frame(Year=floor(time(AirPassengers)), Month=cycle(AirPassengers),
                 Passenger=as.numeric(AirPassengers))
monthly_avg = aggregate(Passenger ~ Month, data=df, FUN=mean)
overall_avg = mean(df$Passenger)
monthly_avg$Seasonal_Index = (monthly_avg$Passenger / overall_avg) * 100
monthly_avg


## 14(b): Ratio-to-Moving-Average Method
# NOTE: the original practical file's snippet for this section was
# abbreviated (it referenced ts_vec / ratio_df without showing their
# construction). Setup lines have been added below so the code runs
# end-to-end; the core method lines are exactly as in the original file.
library(zoo)
ts_vec = AirPassengers
ma12 = rollmean(ts_vec, k=12, align="center", fill=NA)   # zoo::rollmean
ratio = (ts_vec / ma12) * 100
ratio_df = data.frame(Month=cycle(ts_vec), Ratio=as.numeric(ratio))
ratio_df = na.omit(ratio_df)

seasonal_rma = aggregate(Ratio ~ Month, data=ratio_df, FUN=mean)
seasonal_rma$Seasonal_Index = seasonal_rma$Ratio * (1200/sum(seasonal_rma$Ratio))
seasonal_rma

# Comparison of the two methods' seasonal indices (should sum to ~1200)
comparison = data.frame(
  Month = monthly_avg$Month,
  SimpleAvg = round(monthly_avg$Seasonal_Index, 2),
  RatioToMA = round(seasonal_rma$Seasonal_Index, 2)
)
comparison
