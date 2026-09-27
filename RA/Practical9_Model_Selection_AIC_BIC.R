# Practical 9: Model Selection using AIC & BIC
# Aim: To compare several nested MLR models using AIC and BIC, and to see
# whether the two criteria agree on the "best" model.
# Dataset: Insurance data, four nested models built from age, bmi, children.

insurance = read.csv("insurance.csv")
data = insurance

model  = lm(charges ~ age + bmi + children, data)   # Full
model1 = lm(charges ~ age + bmi, data)               # Reduced 1
model2 = lm(charges ~ age + children, data)          # Reduced 2
model3 = lm(charges ~ age, data)                     # Simple

AIC(model, model1, model2, model3)
BIC(model, model1, model2, model3)
