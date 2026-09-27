# Practical 15: Exponential Smoothing
# Aim: To fit a Simple Exponential Smoothing (SES) model, which forecasts the
# future as a weighted average of past observations with exponentially
# decaying weights, and to produce a short-term forecast.
# Dataset: AirPassengers. Package: forecast.

library(forecast)

data("AirPassengers")

ses_model = ses(AirPassengers)
ses_model$model

fitted_values = fitted(ses_model)
plot(AirPassengers); lines(fitted_values, lty=2, col="red")

forecast_12 = forecast(ses_model, h=6)
plot(forecast_12)
forecast_12
