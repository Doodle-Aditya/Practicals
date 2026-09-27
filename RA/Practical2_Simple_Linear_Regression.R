# Practical 2: Simple Linear Regression & Model Estimation
# Aim: To fit a simple linear regression model y = a + bx to real data, using
# three different Kaggle data sets, and to interpret the fitted coefficients
# and predictions.

## 2.1 Dataset 1: Maternal Health Risk Data (SystolicBP ~ Age)
Dataset = read.csv("maternal_health_risk.csv")
head(Dataset); tail(Dataset); str(Dataset); class(Dataset)
table(Dataset$RiskLevel)

plot(Dataset$Age, Dataset$SystolicBP, col="blue", xlab="Age",
     ylab="SystolicBP", pch=19)
model = lm(SystolicBP ~ Age, data = Dataset)
coef(model); summary(model)
abline(model, col="red", lwd=2)
predict(model, newdata = data.frame(Age=c(30,45,60)))
confint(model, level=0.95)


## 2.2 Dataset 2: Salary Dataset (Salary ~ YearsExperience)
Salary_dataset = read.csv("Salary_dataset.csv")
head(Salary_dataset); tail(Salary_dataset); str(Salary_dataset)

plot(Salary_dataset$YearsExperience, Salary_dataset$Salary,
     col="blue", xlab="YearsExperience", ylab="Salary", pch=19)
model = lm(Salary ~ YearsExperience, data = Salary_dataset)
coef(model); summary(model)
abline(model, col="red", lwd=2)
predict(model, newdata = data.frame(YearsExperience = c(2,4,6)))


## 2.3 Dataset 3: Medical Insurance Cost Data (charges ~ bmi)
insurance = read.csv("insurance.csv")
head(insurance); tail(insurance); str(insurance)

plot(insurance$bmi, insurance$charges, col="blue", xlab="bmi",
     ylab="charges", pch=19)
model = lm(charges ~ bmi, data = insurance)
coef(model); summary(model)
abline(model, col="red", lwd=2)
predict(model, newdata = data.frame(bmi=c(20,30,40)))


## 2.4 Dataset 3 revisited: bmi ~ age, with Residual Diagnostics
plot(insurance$age, insurance$bmi, col="orange", xlab="age",
     ylab="bmi", pch=19)
model = lm(bmi ~ age, data = insurance)
coef(model); summary(model)
abline(model, col="red", lwd=2)
predict(model, newdata = data.frame(age=c(20,30,40)))

# Residuals vs fitted
age_fitted = predict(model)
bmi_resid  = residuals(model)
plot(age_fitted, bmi_resid)
abline(h=0, col="red", lwd=2)

# Identify possible outliers
# NOTE: original file used which(abs(bmi)>20000), which is a copy-paste error
# from the charges model. Corrected below to use the bmi-model residuals with
# a threshold appropriate for this model's scale.
which(abs(bmi_resid) > 20)
sort(bmi_resid)
