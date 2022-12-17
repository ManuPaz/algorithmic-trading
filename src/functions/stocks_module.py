import pandas as pd
import numpy as np


class OtherEquities():
    def __init__(self, index, economic_calendar):
        self.index = index
        self.economic_calendar = economic_calendar


class StocksSummary():
    def __init__(self,  general_summary=None, fundamental_summary=None,
                 fundamental_summary_finhub=None, earnings_calendar=None, stock_profile=None):
        """

         general_summary: dataframe with actual  info for different stocks

         fundamental_summary: dataframe with actual financial info for different stocks

         fundamental_summary_finhub: dataframe with actual financial info from finhub for different stocks

         earnings_calendar: earnings calendar until a specific date

         stock_profile: dafaframe with company info for different stocks
        """

        self.stock_objects = {}
        self.general_summary = general_summary
        self.fundamental_summary = fundamental_summary
        self.fundamental_summary_finhub = fundamental_summary_finhub
        self.earnings_calendar = earnings_calendar
        self.stock_profile = stock_profile
        self.fundamental_summary_complete = pd.merge(self.fundamental_summary_finhub, self.fundamental_summary,
                                                     right_index=True, left_index=True)
        self.dataframe_prices_total=None
        self.dataframe_prices_by_column={}

    def update(self, general_summary=None, fundamental_summary=None,
               fundamental_summary_finhub=None, earnings_calendar=None, stock_profile=None):
        self.earnings_calendar = earnings_calendar
        dfs={"stock_profile":{"new":stock_profile,"old":self.stock_profile},
             "general_summary":{"new":general_summary,"old":self.general_summary},
             "fundamental_summary":{"new":fundamental_summary,"old":self.fundamental_summary}}
        for key,df in dfs.items():
            if df["new"] is not None:
                df=pd.concat([df["old"], df["new"]], axis=0)
                df = df[~ df.index.duplicated(keep='last')]
                self.__setattr__(key,df)
        if fundamental_summary_finhub is not None:
            self.fundamental_summary_finhub = pd.concat([self.fundamental_summary_finhub, fundamental_summary_finhub],
                                                        axis=0)
            self.fundamental_summary_finhub = self.fundamental_summary_finhub[
                ~ self.fundamental_summary_finhub.index.duplicated(keep='last')]
        if self.fundamental_summary_finhub is not None and self.fundamental_summary is not None:
            self.fundamental_summary_complete = pd.merge(self.fundamental_summary_finhub, self.fundamental_summary,
                                                         right_index=True, left_index=True, how="outer")

    def get_all_prices(self,names):

        if self.dataframe_prices_total is not None  and set(names).issubset(set(self.dataframe_prices_total.columns.levels[0])):
            print("Prices dataframe already calculated")

            return self.dataframe_prices_total.loc[:,(names,slice(None))]
        dataframe_total = None
        for name in names:
            stock=self.stock_objects[name]

            if stock.prices is not None:
                prices = stock.prices.copy()
                prices.columns = pd.MultiIndex.from_product([[name], prices.columns])
                if dataframe_total is None:
                    dataframe_total=prices
                else:
                    dataframe_total=pd.merge(dataframe_total,prices,right_index=True,left_index=True,how="outer")
        self.dataframe_prices_total=dataframe_total
        return dataframe_total.loc[:,(names,slice(None))]

    def get_all_prices_one_column(self,names,column):
        if column in self.dataframe_prices_by_colum.keys()\
                and set(names).issubset(set(self.dataframe_prices_by_colum[column].columns)):
            print("{} dataframe already calculated".format(column))
            return self.dataframe_prices_by_colum[column].loc[:,names]

        prices=self.get_all_prices(names)
        prices=prices.loc[:,(slice(None),column)].droplevel(1,axis=1)
        self.dataframe_prices_by_colum[column]=prices
        return prices.loc[:,names]

    def get_market_caps(self,symbols, date, stocks_summary):
        if isinstance(symbols[0], tuple):
            symbol_names = [symbol[1] for symbol in symbols]
        else:
            symbol_names = symbols
        prices = self.get_all_prices_one_column(symbol_names, "adj_close")

        p = prices.loc[:date].iloc[-1]
        p1 = prices.iloc[-1]
        symbols = list(
            (set(self.fundamental_summary_complete.droplevel(1).index).intersection(set(symbols))).difference(
                set(p.loc[p.isna()].index).union(set(p.loc[p1.isna()].index))))
        if isinstance(symbols[0], tuple):
            m_caps_actual = self.fundamental_summary.loc[symbols, "marketCapitalization"]
        else:
            m_caps_actual = self.fundamental_summary_complete.droplevel(1).loc[
                symbols, "marketCapitalization"].drop_duplicates()

        market_caps = (m_caps_actual * (p / p1)).loc[symbols]
        return market_caps, symbols

    def clean_data(self,df_returns,max_return=0.4):
        maximo=df_returns.max(axis=1)
        index=maximo.loc[maximo>=max_return].index
        return df_returns.loc[index].applymap(lambda x: np.nan if x<max_return else x)

    def init_filtering(self,data,indice=None):
        data = data.dropna(how="all")
        data.loc[:, ["sector", "industry"]] = self.stock_profile.loc[:,["sector", "industry"]]
        data = data.fillna(value=np.nan)
        if indice is not  None:
            data=data.loc[(slice(None),indice),:]
        return data
    def filter_margin(self,data,margin=25):
        filter="netProfitMarginTTM>{}".format(margin)
        data_filtered = data.query(filter)
        return data_filtered
    def filter_ev_cashflow(self,data,ratio=25):
        data_filtered = data.loc[data["currentEv/freeCashFlowTTM"]<ratio]
        return data_filtered
    def filter_debt(self,data,ratio=10):
        data["netDebt"]=data["totalDebt"]-data["totalCash"]
        data["netDebt_to_fcf"]=data["netDebt"]/data["freeCashflow"]
        filter="netDebt_to_fcf<{}".format(ratio)
        data=data.query(filter)
        return  data

    def filter_growth(self,data,years=3):
        filter = "revenueGrowth{}Y>10 and epsGrowth{}Y>15 and netMarginGrowth5Y>1".format(years,years)
        data_filtered = data.query(filter)
        return data_filtered

    def filter_bigfall(self,data):
        data_filtered = data.loc[(data["52WeekPriceReturnDaily"]<(-40)) ]
        return data_filtered

    def filter_target_prices(self,data):
        data_filtered = data.loc[(data["targetMedianPrice"]/data["currentPrice"]>=2)
                                &(data["targetLowPrice"]/data["currentPrice"]>=1.5)
                                &(data["numberOfAnalystOpinions"]>=4)
                                &(data["revenueGrowth5Y"]>0)&(data["epsGrowth5Y"]>0)]
        return data_filtered

    def filter_columns(self,data):
        columns=["52WeekPriceReturnDaily","beta","currentDividendYieldTTM",\
                "currentEv/freeCashFlowAnnual","ebitdaCagr5Y","epsGrowth5Y",\
                "epsGrowthQuarterlyYoy","marketCapitalization","netMarginGrowth5Y",\
                "netProfitMarginTTM","operatingMarginTTM","revenueGrowth5Y",\
                "revenueGrowthQuarterlyYoy","industry","sector"]
        return data.loc[:,columns]



class Stock():
    def __init__(self, name, prices, **kwargs):
        """
         name: symbol name

         prices:  prices dataframes

         **sentiment: dataframe with sentiment scores

         **twitter: dataframe with  twitter sentiment scores

         **reddit: dataframe with  reditt sentiment scores

         **annual_eps: dataframe with past annual eps

         **quarterly_eps: dataframe with past quarterly eps

         **trending_eps: dataframe with future eps estimations both quarterly and annual

         **puts: put options with different expiration dates

         **calls: call options with different expiration dates

         **quarterly_financials: quarterly financial statements

         **annual_financials:  annual financial statements

         **recomendation: annalist recomendations

         **indicators: indicators dataframe
        """
        self.name = name
        self.prices = prices if prices.shape[0] > 0 else None
        for key in kwargs.keys():
            if kwargs[key] is not None:
                value = kwargs[key] if kwargs[key].shape[0] > 0 else None
                self.__setattr__(key, value)
