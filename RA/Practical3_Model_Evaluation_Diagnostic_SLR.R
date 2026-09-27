# Practical 3: Model Evaluation & Diagnostic Analysis of SLR
# Aim: To evaluate the quality of a fitted SLR model beyond just R-squared,
# using residual diagnostics, and to formally identify outliers.
# Dataset: Insurance data, bmi ~ age (same model as Practical 2, Section 2.4).
#
# This practical reuses the same model and residual-vs-fitted analysis already
# carried out in Practical 2 (Section 2.4). Re-run here for completeness.

insurance = read.csv("insurance.csv")

model = lm(bmi ~ age, data = insurance)
coef(model); summary(model)

plot(insurance$age, insurance$bmi, col="orange", xlab="age",
     ylab="bmi", pch=19)
abline(model, col="red", lwd=2)

age_fitted = predict(model)
bmi_resid  = residuals(model)
plot(age_fitted, bmi_resid, main="Residuals vs Fitted: bmi ~ age",
     xlab="Fitted values (age model)", ylab="Residuals")
abline(h=0, col="red", lwd=2)

# Identify possible outliers
which(abs(bmi_resid) > 20)
sort(bmi_resid)

# Diagnostic principles for reading a residual-vs-fitted plot:
#   Random scatter = good relationship (assumptions satisfied)
#   Curved scatter  = not a linear relationship
#   Funnel shape    = unequal variance / heteroscedasticity
#   Clustered shape = group formation / omitted variable(s)
