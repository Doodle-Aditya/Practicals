# Practical 8: Autocorrelation & Model-Diagnostic
# Aim: To test the MLR model's residuals for first-order autocorrelation
# using the Durbin-Watson test.
# Dataset: same MLR model as Practical 5.

library(lmtest)

insurance = read.csv("insurance.csv")
data = insurance
model = lm(charges ~ age + bmi + children, data)

dwtest(model)
plot(residuals(model), main="Residuals in observation order")
abline(h=0, col="red")
