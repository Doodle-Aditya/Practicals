# Practical 1: Correlation Analysis (Scatter Diagram & Karl Pearson's Correlation)
# Aim: To study the relationship between Advertisement expenditure and Sales using
# a scatter diagram, and to compute and test Karl Pearson's correlation coefficient.

Advertisement = c(15,18,12,22,25,30,35,38,42,45)
Sales = c(45,48,40,54,60,66,70,75,82,88)

plot(Advertisement, Sales, main="scatter plot", xlab="Advertisement",
     ylab="Sales", pch=4, col="red")

cor(Advertisement, Sales)
cor.test(Advertisement, Sales)
