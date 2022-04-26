
# Future devs
class Stock:
  def __init__(self, stockName, stockData, nextAnalysisTime):
    self.stockName = stockName
    self.stockData = stockData
    self.nextAnalysisTime = nextAnalysisTime
    volumeProfiles = caculate_volume_profiles(stockData)
    atr = []
    ma5 = []
    ma10 = []
    rsi = stock_data['rsi_14']
    macd_histogram = stock_data['macdh']
    change = stock_data['change']
    close = stock_data["close"]
    high = stock_data["high"]
    low = stock_data["low"]
  
  def caculate_volume_profiles(stock_data):
    # https://medium.com/swlh/how-to-analyze-volume-profiles-with-python-3166bb10ff24
    # print(stock_data["close"].min(), " ", stock_data["close"].max(), " ", len(stock_data["close"]))
    kde_factor = 0.05
    num_samples = len(stock_data)
    kde = stats.gaussian_kde(stock_data["close"],weights=stock_data["volume"],bw_method=kde_factor)
    xr = np.linspace(stock_data["close"].min(),stock_data["close"].max(),num_samples)
    kdy = kde(xr)

    # Find peaks
    min_prom = kdy.max() * 0.1
    peaks, peak_props = signal.find_peaks(kdy, prominence=min_prom)

    # Add peaks to dictionary
    volumePeaks = {}
    # pocIndex = -1
    for i in range(0, len(peaks)):
      volumePeaks[xr[peaks[i]]] = kdy[peaks[i]]/kdy.min()*100
    #   if kdy[peaks[pocIndex]] < kdy[peaks[i]]:
    #     pocIndex = i
    # if pocIndex >= 0:
    #   volumePeaks["poc"] = xr[peaks[pocIndex]]
    # print (volumePeaks)

    ##### Draw the figure
    # ticks_per_sample = (xr.max() - xr.min()) / num_samples
    # pk_marker_args=dict(size=10)
    # pkx = xr[peaks]
    # pky = kdy[peaks]

    # fig = get_dist_plot(stock_data["close"], stock_data["volume"], xr, kdy)
    # fig.add_trace(go.Scatter(name='Peaks', x=pkx, y=pky, mode='markers', marker=pk_marker_args))
    # fig.show()
    return volumePeaks

  def get_dist_plot(c, v, kx, ky):
    fig = go.Figure()
    fig.add_trace(go.Histogram(name='Vol Profile', x=c, y=v, nbinsx=150, 
                              histfunc='sum', histnorm='probability density',
                              marker_color='#B0C4DE'))
    fig.add_trace(go.Scatter(name='KDE', x=kx, y=ky, mode='lines', marker_color='#D2691E'))
    return fig
