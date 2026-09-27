# Practical 7: Multicollinearity & Heteroscedasticity in MLR
# Aim: To check the MLR model of Practical 5 for multicollinearity (VIF) and
# heteroscedasticity (Breusch-Pagan test), and to correct heteroscedasticity
# using a log transform and weighted least squares (WLS).
# Packages: car (for vif), lmtest (for bptest).

library(car)
library(lmtest)

insurance = read.csv("insurance.csv")
data = insurance
model = lm(charges ~ age + bmi + children, data)

vif(model)
bptest(model)

model_log = lm(log(charges) ~ age + bmi + children, data)
summary(model_log)
bptest(model_log)

data$predicted = fitted(model)
data$weights = 1/data$predicted^2
model_wls = lm(charges ~ age + bmi + children, data, weights = data$weights)
summary(model_wls)
bptest(model_wls)
