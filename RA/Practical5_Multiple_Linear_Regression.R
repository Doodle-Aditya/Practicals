# Practical 5: Multiple Linear Regression
# Aim: To fit a multiple linear regression model with more than one regressor,
# and interpret the partial regression coefficients.
# Dataset: Insurance data, charges ~ age + bmi + children.

insurance = read.csv("insurance.csv")
data = insurance
head(data); tail(data)

model = lm(charges ~ age + bmi + children, data)
summary(model)

new = data.frame(age=40, bmi=30, children=2)
predict(model, newdata=new)
