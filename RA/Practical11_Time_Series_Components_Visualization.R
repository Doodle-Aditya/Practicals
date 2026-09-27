# Practical 11: Time-Series Components & Visualization
# Aim: To visualise a classic time series and decompose it into trend,
# seasonal, and random (irregular) components, both additively and
# multiplicatively.
# Dataset: AirPassengers - monthly international airline passenger totals,
# Jan 1949 to Dec 1960 (built into R).

data("AirPassengers")
plot(AirPassengers, main="AirPassengers: Monthly totals 1949-1960")

ma = filter(AirPassengers, filter=rep(1/12,12), sides=2)
plot(AirPassengers); lines(ma, col="red", lwd=2)

additive <- decompose(AirPassengers, type="additive")
plot(additive)

multiplicative <- decompose(AirPassengers, type="multiplicative")
plot(multiplicative)
