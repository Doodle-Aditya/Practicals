# Practical 12: Estimation of Trend using Moving Average
# Aim: To smooth out seasonal and irregular fluctuations and estimate the
# underlying trend using centred moving averages of different window lengths.
# Dataset: AirPassengers.

data("AirPassengers")

ma12 = filter(AirPassengers, filter = rep(1/12, 12), sides = 2)
plot(AirPassengers, main="AirPassengers with 12-Month Moving Average")
lines(ma12, col="red", lwd=2)

ma3 = filter(AirPassengers, filter = rep(1/3, 3), sides = 2)
plot(AirPassengers, main="AirPassengers with 3-Month Moving Average")
lines(ma3, col="blue", lwd=2)
