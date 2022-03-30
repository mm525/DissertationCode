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
        self.data_diff = df[self.symbol].diff().dropna()
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

    # Calculate the ADF statistic and p-value for the differenced time series
    def diff_adf(self):
        self.adf_result = adfuller(self.data_diff)
        print(f'ADF Statistic: {self.adf_result[0]}, p-value: {self.adf_result[1]}')
        
    # Correlogram (autocorrelation and partial autocorrelation)
    def acp(self):
        fig2, ax2 = plt.subplots(nrows = 1, ncols = 2, sharex = True, figsize = (10, 5))
        plot_acf(self.data, ax = ax2[0])        
        plot_pacf(self.data, ax = ax2[1])
        
    # Correlogram for differenced time series    
    def diff_acp(self):
        fig3, ax3 = plt.subplots(nrows = 1, ncols = 2, sharex = True, figsize = (10, 5))
        plot_acf(self.data_diff, ax = ax3[0])        
        plot_pacf(self.data_diff, ax = ax3[1])

    # Model the time series as an AR(I)MA process (regression includes dummy variables)
    def mod_ARMA(self, ar, ma, integ = 0):
        self.ar = ar
        self.ma = ma
        self.integ = integ
        self.mod = statsmodels.tsa.arima.model.ARIMA(self.data, exog = self.dummies,
                                                     order = (self.ar, self.integ, self.ma))
        self.mod_fit = self.mod.fit(cov_type = 'robust')
        self.pred = self.mod_fit.predict()
        print(self.mod_fit.summary())
        fig4, ax4 = plt.subplots(figsize = (15, 10))
        self.residuals = self.data - np.array(self.pred).reshape(-1,1)
        ax4.scatter(self.date[1:], self.residuals[1:])
        print(f'Durbin-Watson statistic: {str(durbin_watson(self.residuals))}')
        print(str(statsmodels.stats.diagnostic.het_arch(self.residuals, nlags = 1)))
        
    # Correlogram for volatility    
    def acp_GARCH(self):
        self.vol = self.data_diff**2
        fig4, ax4 = plt.subplots(nrows = 1, ncols = 2, sharex = True, figsize = (10, 5))
        plot_acf(self.vol, ax = ax4[0])        
        plot_pacf(self.vol, ax = ax4[1])
        
    # Model return volatility using GARCH
    def mod_GARCH(self, p, q):
        self.p = p
        self.q = q
        self.am = arch_model(self.data_diff, vol = "Garch", p = self.p, o = 0, q = self.q, dist = "Normal")
        self.res = self.am.fit(update_freq = 5)
        print(self.res.summary())
        
    # Evaluate the best AR(I)MA model by outputting AIC and BIC, subject to max. AR and MA values
    def eval_ARMA(self, ar_max, ma_max, integ1 = 0):
        self.ar_max = ar_max
        self.ma_max = ma_max
        self.integ1 = integ1 
        for i in range(self.ar_max + 1):
            for j in range(self.ma_max + 1):
                self.eval_mod = statsmodels.tsa.arima.model.ARIMA(self.data, 
                                                                  exog = self.dummies,
                                                                  order = (i, self.integ1, j))
                self.eval_fit = self.eval_mod.fit()
                print(f'ARIMA ({i}, {self.integ1}, {j}): AIC = {self.eval_fit.aic}, BIC = {self.eval_fit.bic}')
                
class equityStats(bondStats): 
    """
    
    Create new class for Equities so that it inherits all bondStats methods but re-defines the variable self.data in terms of (ln) returns rather
    than just prices, and re-defines certain methods to account for not needing to difference the time series (i.e. replacement of self.data_diff 
    with self.data)
    
    """
    
    def get_data(self):
        df = pd.read_excel('[file_name].xlsx')
        df = df.dropna(axis = 0)
        self.date = np.array(df['Date'][1:])
        self.data = np.array(np.log(df[self.symbol]/df[self.symbol].shift(1)).dropna()).reshape(-1,1)
        self.dummies = df[['QE1', 'QE2', 'QE3', 'QE4']][1:]

    def acp_GARCH(self):
        self.vol = self.data**2
        fig4, ax4 = plt.subplots(nrows = 1, ncols = 2, sharex = True, figsize = (10, 5))
        plot_acf(self.vol, ax = ax4[0])        
        plot_pacf(self.vol, ax = ax4[1])
    
    def mod_GARCH(self, p, q):
        self.p = p
        self.q = q
        self.am = arch_model(self.data, vol = "Garch", p = self.p, o = 0, q = self.q, dist = "Normal")
        self.res = self.am.fit(update_freq = 5)
        print(self.res.summary())
