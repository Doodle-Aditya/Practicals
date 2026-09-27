# Practical 10: Binary Logistic Regression
# Aim: To fit a binary logistic regression model predicting survival from the
# Titanic disaster using Age and Fare, and to evaluate its classification
# performance.
# Dataset: Titanic (Kaggle), 891 passengers; 177 have missing Age and are
# dropped, leaving 714 rows. Package: pscl (for McFadden's pseudo R-squared).

Titanic = read.csv("titanic.csv")
data = Titanic[,c("Survived","Age","Fare")]
data = na.omit(data)

model = glm(Survived ~ Age + Fare, data=data, family=binomial)
summary(model)
exp(coef(model))                 # odds ratios

data$predprob = predict(model, type="response")
data$predclass = ifelse(data$predprob >= 0.5, 1, 0)
table(Actual=data$Survived, Predicted=data$predclass)
mean(data$Survived == data$predclass)   # accuracy

library(pscl)
pR2(model)                       # McFadden pseudo R-squared
