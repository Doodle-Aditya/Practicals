# Practical 16: Auto-Regressive Model (AR1) & (AR2)
# Aim: To fit first- and second-order autoregressive models to the
# (differenced, trend-removed) series, compare them by AIC, and forecast
# forward.
# Dataset: AirPassengers, first-differenced to remove the trend.

data("AirPassengers")

diff_passengers = diff(AirPassengers)
plot(diff_passengers, main="First-differenced AirPassengers series")

air1_model = arima(diff_passengers, order=c(1,0,0))   # AR(1)
air2_model = arima(diff_passengers, order=c(2,0,0))   # AR(2)
air1_model; air2_model
AIC(air1_model, air2_model)

forecast_air1 = predict(air1_model, n.ahead=12)
forecast_air2 = predict(air2_model, n.ahead=12)
