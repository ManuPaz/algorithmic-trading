import pandas as pd
import numpy as np


class OtherEquities():
    def __init__(self, index, economic_calendar):
        self.index = index
        self.economic_calendar = economic_calendar


class StocksSummary():
    def __init__(self, stocks_list=None, general_summary=None, fundamental_summary=None,
                 fundamental_summary_finhub=None, earnings_calendar=None, stock_profile=None):
        """
         stocks_list: stocks which have series info

         general_summary: dataframe with actual  info for different stocks

         fundamental_summary: dataframe with actual financial info for different stocks

         fundamental_summary_finhub: dataframe with actual financial info from finhub for different stocks

         earnings_calendar: earnings calendar until a specific date

         stock_profile: dafaframe with company info for different stocks
        """
        self.stocks_list = stocks_list
        self.stock_objects = {}
        self.general_summary = general_summary
        self.fundamental_summary = fundamental_summary
        self.fundamental_summary_finhub = fundamental_summary_finhub
        self.earnings_calendar = earnings_calendar
        self.stock_profile = stock_profile
        self.fundamental_summary_complete = pd.merge(self.fundamental_summary_finhub, self.fundamental_summary,
                                                     right_index=True, left_index=True)

    def update(self, stocks_list=None, general_summary=None, fundamental_summary=None,
               fundamental_summary_finhub=None, earnings_calendar=None, stock_profile=None):
        self.earnings_calendar = earnings_calendar
        self.stocks_list = list(np.unique(stocks_list + stocks_list))
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
        data_filtered = data.loc[(data["52WeekPriceReturnDaily"]<(-40)) & (data["13WeekPriceReturnDaily"]<(-13))]
        return data_filtered




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