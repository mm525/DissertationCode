class bondStats:
    
    # Create object for a security
    def __init__(self, symbol, start = None, end = None,
                 ar = None, ma = None, ar_max = None, ma_max = None):
        self.symbol = symbol
        self.start = start
        self.end = end
        self.ar = ar
        self.ma = ma
        self.ar_max = ar_max
        self.ma_max = ma_max
        self.get_data()
    
    # Select the security's return over the time period
    def get_data(self):
        df = pd.read_excel('/Users/marcusmayfield/Documents/Diss_Bond_Data_excl30v2.xlsx')
        df = df.dropna(axis = 0)
        self.date = np.array(df['Date']).reshape(-1,1)
        self.data = np.array(df[self.symbol]).reshape(-1,1)
    
    # Plot graph of the security's returns over time 
    def graph(self, start = None, end = None):
        self.start = start
        self.end = end
        fig1, ax1 = plt.subplots(figsize = (10, 5))
        ax1.plot(self.date[self.start:self.end], self.data[self.start:self.end])
        plt.title(f'Market Yield on U.S. Treasury Securities at {self.symbol}')
        plt.xlabel('Time')
        plt.ylabel('Yield (%)')
    
    # Calculate the Augmented Dickey Fuller test statistic and p-value
    def adf(self):
        self.adf_result = adfuller(self.data)
        print(f'ADF Statistic: {self.adf_result[0]}, p-value: {self.adf_result[1]}')

    # Correlogram (autocorrelation and partial autocorrelation)
    def acp(self):
        fig2, axes = plt.subplots(nrows = 1, ncols = 2, sharex = True, figsize = (10, 5))
        plot_acf(self.data, ax = axes[0])        
        plot_pacf(self.data, ax = axes[1])

    # Model the time series as an ARMA process (regression includes dummy variables)
    def mod_ARMA(self, ar, ma):
        self.ar = ar
        self.ma = ma
        self.mod = statsmodels.tsa.arima.model.ARIMA(self.data, 
                                                     np.array(df['dum1']).reshape(-1,1),
                                                     order = (self.ar, 0, self.ma))
        self.mod_fit = self.mod.fit()
        self.pred = self.mod_fit.predict()
        print(self.mod_fit.summary())
        fig3, ax3 = plt.subplots(figsize = (15, 10))
        ax3.plot(self.date[520:580], self.data[520:580], label = 'Raw')
        ax3.plot(self.date[520:580], self.pred[520:580], label = 'Fit')
        plt.legend(loc = 0)

    # Evaluate the best ARMA model by outputting AIC and BIC, subject to max. AR and MA values
    def eval_ARMA(self, ar_max, ma_max):
        self.ar_max = ar_max
        self.ma_max = ma_max
        for i in range(self.ar_max + 1):
            for j in range(self.ma_max + 1):
                self.eval_mod = statsmodels.tsa.arima.model.ARIMA(self.data, 
                                                                  np.array(df['dum1']).reshape(-1,1),
                                                                  order = (i, 0, j))
                self.eval_fit = self.eval_mod.fit()
                print(f'ARIMA ({i}, 0, {j}): AIC = {self.eval_fit.aic}, BIC = {self.eval_fit.bic}')
