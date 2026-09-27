# Practical 6: Residual Analysis & Detection of Outliers in MLR
# Aim: To compute standardised residuals for an MLR model, flag outliers,
# and examine the standard 4-panel diagnostic plot.
# Dataset: same MLR model as Practical 5.

insurance = read.csv("insurance.csv")
data = insurance
model = lm(charges ~ age + bmi + children, data)

data$predicted = fitted(model)
data$resid     = residuals(model)
data$stdr      = rstandard(model)    # standardised residuals
head(data[,c("charges","predicted","resid","stdr")])

outliers = which(abs(data$stdr) > 2)
length(outliers)

par(mfrow=c(2,2))
plot(model)   # Residuals vs Fitted, Q-Q, Scale-Location, Residuals vs Leverage
