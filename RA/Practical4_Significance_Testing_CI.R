# Practical 4: Significance Testing & Confidence Interval
# Aim: To formally test the significance of a regression coefficient
# (H0: beta1 = 0) and to construct a confidence interval for it.
# Dataset: Maternal Health Risk data, SystolicBP ~ Age (same model as
# Practical 2, Section 2.1).

Dataset = read.csv("maternal_health_risk.csv")

model = lm(SystolicBP ~ Age, data = Dataset)
summary(model)          # t-value, p-value for H0: beta1 = 0
confint(model, level=0.95)
