class bondStats:
    
    # Create object for a security
    def __init__(self, symbol, start = None, end = None, ar = None, ma = None, integ = None, ar_max = None, ma_max = None):
        self.symbol = symbol
        self.start = start
        self.end = end
        self.ar = ar
        self.ma = ma
        self.integ = integ
        self.ar_max = ar_max
        self.ma_max = ma_max
        self.get_data()
    
    # Select the security's return over the time period
    def get_data(self):
        df = pd.read_excel('[file_name].xlsx')
        df = df.dropna(axis = 0)
        self.date = np.array(df['Date']).reshape(-1,1)
        self.data = np.array(df[self.symbol]).reshape(-1,1)
        
        # Calculate first difference of the time series
        self.data_diff = df[self.symbol].diff().dropna()
        
        # Create a new DataFrame that contains the dummy variables - needed for the ARIMA model
        self.dummies = df[['QE1', 'QE2', 'QE3', 'QE4']]
    
    # Plot graph of the security's returns over time 
    def graph(self, start = None, end = None):
        self.start = start
        self.end = end
        fig1, ax1 = plt.subplots(figsize = (10, 5))
        ax1.plot(self.date[self.start:self.end], self.data[self.start:self.end])
        plt.title(f'Market Yield on Security {self.symbol}')
        plt.xlabel('Time')
        plt.ylabel('Yield (%)')
    
    # Calculate the Augmented Dickey Fuller test statistic and p-value
    def adf(self):
        self.adf_result = adfuller(self.data)
        print(f'ADF Statistic: {self.adf_result[0]}, p-value: {self.adf_result[1]}')
    
    # Calculate the first-differenced Augmented Dickey Fuller test statistic and p-value
    def diff_adf(self):
        self.diff_adf_result = adfuller(self.data_diff)
        print(f'ADF Statistic: {self.diff_adf_result[0]}, p-value: {self.diff_adf_result[1]}')
        
    # Correlogram (autocorrelation and partial autocorrelation)
    def acp(self):
        fig2, ax2 = plt.subplots(nrows = 1, ncols = 2, sharex = True, figsize = (10, 5))
        plot_acf(self.data, ax = ax2[0])        
        plot_pacf(self.data, ax = ax2[1])
    
    # First-differenced correlogram (autocorrelation and partial autocorrelation)
    def diff_acp(self):
        fig3, ax3 = plt.subplots(nrows = 1, ncols = 2, sharex = True, figsize = (10, 5))
        plot_acf(self.data_diff, ax = ax3[0])        
        plot_pacf(self.data_diff, ax = ax3[1])

    # Model the time series as an ARMA process (regression includes dummy variables)
    def mod_ARMA(self, ar, integ = 0, ma):
        self.ar = ar
        self.ma = ma
        self.integ = integ
        self.mod = statsmodels.tsa.arima.model.ARIMA(self.data, exog = self.dummies, order = (self.ar, self.integ, self.ma))
        self.mod_fit = self.mod.fit(cov_type = 'robust_approx')
        self.pred = self.mod_fit.predict()
        print(self.mod_fit.summary())
        
        # Observe homo/hetero-skedasticity of random errors      
        fig4, ax4 = plt.subplots(figsize = (15, 10))
        self.residuals = self.data - np.array(self.pred).reshape(-1,1)
        ax4.scatter(self.date[1:], self.residuals[1:])
        
        # Examine degree of autocorrelation (i.e. proxy for degree of model misspecification) 
        print(f'Durbin-Watson statistic: {str(durbin_watson(self.residuals))}')

    # Evaluate the best ARMA model by outputting AIC and BIC, subject to specified maximum AR and MA lags
    def eval_ARMA(self, ar_max, integ1 = 0, ma_max):
        self.ar_max = ar_max
        self.ma_max = ma_max
        self.integ1 = integ1 
        for i in range(self.ar_max + 1):
            for j in range(self.ma_max + 1):
                self.eval_mod = statsmodels.tsa.arima.model.ARIMA(self.data, exog = self.dummies, order = (i, self.integ1, j))
                self.eval_fit = self.eval_mod.fit()
                print(f'ARIMA ({i}, {self.integ1}, {j}): AIC = {self.eval_fit.aic}, BIC = {self.eval_fit.bic}')
