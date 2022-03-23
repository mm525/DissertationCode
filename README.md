# Public Dissertation Code

This repository contains code that I have used for sections of my dissertation.

Current sections:
- ARIMA modelling of bond returns: OOP that enables user to create objects (based on securities) and plot a graph of their returns, carry out Augmented Dickey Fuller tests, plot correlograms to identify AR and MA processes, select an ARMA model, evaluate the best ARMA model subject to specified maximum AR and MA values and model volatility using the GARCH model.
  - Create new class for equities that inherits most of the methods from the bond class, but re-defines a few things to account for the fact that we must calculate ln(Pt+1/Pt) to estimate returns (vs prices) for equities.
