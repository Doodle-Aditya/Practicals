# Practical 13: Estimation of Linear Trend using Least Square Method
# Aim: To fit a straight-line trend to annual totals using ordinary least
# squares (i.e. simple linear regression of the series on time), and to
# forecast the next year.
# Dataset: AirPassengers, aggregated to annual totals.

data("AirPassengers")

annual_passenger = aggregate(AirPassengers, nfrequency = 1, FUN=sum)
df = data.frame(Year=time(annual_passenger), Passengers=as.numeric(annual_passenger))

trend_model = lm(Passengers ~ Year, data=df)
summary(trend_model)

plot(df$Year, df$Passengers, type="o", pch=19, xlab="Year", ylab="Annual Passengers")
abline(trend_model, col="red", lwd=2)

predict(trend_model, newdata=data.frame(Year=1961))
