import requests_cache
import json
import pandas as pd
import numpy as np
import csv
import yahoo_fin.stock_info as si
import pandas_datareader.data as web
from src.functions import utils
import logging.config
logging.config.fileConfig('resources/logging.conf')
logger = logging.getLogger('api_error')



class Api():
    def __init__(self, api_key):
        self.api_key = api_key
        self.session = requests_cache.CachedSession()

class AlphaVantage(Api):
    def __init__(self,api_key):
        super().__init__(api_key)
    def get_daily_stock_prices(self,symbol,format="full"):
        url = 'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={}&outputsize={}&apikey={}'.format(symbol,format,self.api_key)
        r = self.session.get(url)
        data = r.json()
        data=pd.DataFrame.from_dict(data["Time Series (Daily)"],orient='index')

        return data
    def get_eps(self,symbol):
        url='https://www.alphavantage.co/query?function=EARNINGS&symbol={}&apikey={}'.format(symbol,self.api_key)
        r =self.session.get(url)
        data = r.json()
        a_earnings=pd.DataFrame()
        q_earnings=pd.DataFrame()
        if "quarterlyEarnings" in data.keys():
            q_earnings=pd.DataFrame(data["quarterlyEarnings"])
        else:
            logger.error("Quartery earnings not abled for stock {}".format(symbol))
        if "annualEarnings" in data.keys():
            a_earnings =pd.DataFrame(data["annualEarnings"])
        else:
            logger.error("Annual earnings not abled for stock {}".format(symbol))
        return  a_earnings, q_earnings
    def get_earnings_calendar(self,horizon="3month"):
        url = 'https://www.alphavantage.co/query?function=EARNINGS_CALENDAR&horizon={}&apikey={}'.format(horizon,self.api_key)
        r = self.session.get(url).content.decode('utf-8')
        cr = csv.reader(r.splitlines(), delimiter=',',)
        data=pd.DataFrame(cr)
        data.columns=data.loc[0]
        data=data.iloc[1:]
        data=data.set_index("reportDate")
        data.index=pd.to_datetime(data.index)
        data=data.sort_index()
        return data

    def get_US_listed_stocks(self):
        url = 'https://www.alphavantage.co/query?function=LISTING_STATUS&apikey={}'.format(self.api_key)
        r = self.session.get(url).content.decode('utf-8')
        cr = csv.reader(r.splitlines(), delimiter=',')
        return pd.DataFrame(cr)

    def get_new_and_sentiments_list(self,stock,topics=None):
        if topics is not  None:
            url = 'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={}&topics={}&apikey={}'.format(stock,topics,self.api_key)
        else:
            url = 'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={}&apikey={}'.format(
                stock, self.api_key)

        r =  self.session.get(url)
        data = r.json()
        if "feed" in data.keys():
            return data["feed"]
        logger.error("News and sentiments not abled for stock {}".format(stock))
        return pd.DataFrame()
    def get_sentiments(self,stock,topics=None):
        data=self.get_new_and_sentiments_list(stock,topics)
        data_aux = [ticker for element in data for ticker in element["ticker_sentiment"]]
        data_aux=pd.DataFrame(data_aux)
        if data_aux.shape[0]>0:
            data_aux=data_aux.loc[data_aux.ticker==stock]
            data_aux["title"]=[element["title"] for element in data]
            data_aux["url"] = [element["url"] for element in data]
            data_aux["time_published"] = [element["time_published"] for element in data]
            data_aux["time_published"]=pd.to_datetime(data_aux["time_published"])
            data_aux=data_aux.set_index("time_published")
            data_aux=data_aux.sort_index()
            return data_aux
        return pd.DataFrame()
    def get_macroeconomic_data(self,interval="monthly"):
        info=["CPI","INFLATION","INFLATION_EXPECTATION",
              "CONSUMER_SENTIMENT","RETAIL_SALES","DURABLES",
              "UNEMPLOYMENT","NONFARM_PAYROLL","REAL_GDP",
              "REAL_GDP_PER_CAPITA","TREASURY_YIELD",
              "FEDERAL_FUNDS_RATE"]
        data_total=None
        for series_name in info :
            url='https://www.alphavantage.co/query?function={}&interval={}&apikey={}'.format(series_name,interval,self.api_key)
            data=  self.session.get(url).content.decode('utf-8')
            data=json.loads(data)
            if "data" in data.keys():
                data=pd.DataFrame(data["data"]).set_index("date").rename(columns={"value":series_name})
                if data_total is None:
                    data_total=data
                else:
                    data_total=pd.merge(data_total,data,how="outer",left_index=True,right_index=True)

        return data_total
class Finhub(Api):
    def __init__(self,api_key):
        super().__init__(api_key)

    def get_financial_data(self,stock,freq="quarterly"):
        if freq=="None" or freq=="annual":
            url="https://finnhub.io/api/v1/stock/financials-reported?symbol={}&token={}".format(stock,self.api_key)
        else:
            url = "https://finnhub.io/api/v1/stock/financials-reported?symbol={}&token={}&freq={}".format(stock, self.api_key,freq)

        data=self.session.get(url).content.decode("utf-8")
        try:
            data=json.loads(data)
        except Exception as e:
            logger.error("{} financial data not abled for stock {}".format(freq,stock))
            return pd.DataFrame()
        if "data" not in data.keys():
            logger.error("{} financial data not abled for stock {}".format(freq, stock))
            return pd.DataFrame()

        df_return= pd.DataFrame(data["data"])
        for st in ["bs","cf","ic"]:

            df = [e["report"][st] for e in data["data"]]
            df = [{e["concept"]: e["value"] for e in i} for i in df]
            df=pd.DataFrame(df)
            df_return=pd.merge(df_return,df,right_index=True,left_index=True)
        df_return=df_return.rename(columns={e:e.replace("us-gaap_","gaap_") for e in df_return.columns})
        df_return=df_return.rename(columns={e:e.replace("_x","") for e in df_return.columns})
        columns_y=[e for e in df_return.columns if "_y" in e]
        df_return=df_return.drop(columns_y,axis=1)
        cols=list(df_return.columns)
        cols.sort()
        df_return=df_return.loc[:,cols]
        if  df_return.shape[0]>0:
            df_return=df_return.drop("report",axis=1)
            return df_return
        logger.error("{} financial data not abled for stock {}".format(freq,stock))
        return df_return

    def get_sentiment(self,stock):
        url = "https://finnhub.io/api/v1/stock/social-sentiment?symbol={}&token={}&from=2021-01-01&to=2022-12-01".format(stock, self.api_key)
        data = self.session.get(url).content.decode("utf-8")
        data=json.loads(data)
        social_media_dfs={"twitter":None,"reddit":None}
        for social_media_type in social_media_dfs:
            if social_media_type not in data.keys():
                social_media_dfs[social_media_type]=pd.DataFrame()

            else:
                df=pd.DataFrame(data[social_media_type])

            if df.shape[0]>0:
                df=df.set_index("atTime")
                df.index = pd.to_datetime(df.index)
                df = df.sort_index()
                social_media_dfs[social_media_type] = df
            else:
                logger.error("{} info not abled for stock {}".format(social_media_type,stock))

        return social_media_dfs["twitter"],social_media_dfs["reddit"]

    def get_earnings(self,stock):
        url = "https://finnhub.io/api/v1/stock/earnings?symbol={}&token={}".format(stock, self.api_key)
        data = self.session.get(url).content.decode("utf-8")
        data = json.loads(data)
        data=pd.DataFrame(data)
        return data

    def get_basic_financials_current(self, stocks,indice):

        df = None
        for stock in stocks:
            data = self. get_basic_financials_current_(stock)
            if data.shape[0] > 0:
                if df is None:
                    df = pd.DataFrame(columns=data.index)
                df.loc[stock] = data
        df["indice"]=indice
        df = df.set_index("indice", append=True)
        return df
    def get_basic_financials_current_(self,stock):
        url = "https://finnhub.io/api/v1/stock/metric?symbol={}&token={}&metric=all".format(stock, self.api_key)
        data = self.session.get(url).content.decode("utf-8")
        data = json.loads(data)
        if "metric" not in data.keys():
            logger.error("Actual finhub financial summary not abled for stock {}".format(stock))
            return pd.Series()
        data = pd.Series(data["metric"])
        return data

    def get_basic_financials(self, stock,freq="annual"):
        url = "https://finnhub.io/api/v1/stock/metric?symbol={}&token={}&metric=all".format(stock, self.api_key)
        data = self.session.get(url).content.decode("utf-8")
        data = json.loads(data)
        if "series" not in data.keys():
            logger.error("{} financial data not abled for stock {}".format(freq, stock))
            return pd.DataFrame()

        if freq=="annual" and "annual" in data["series"].keys():
            raw_dict=data["series"]["annual"]
        elif  freq!="annual" and "quarterly" in data["series"].keys():
            raw_dict=data["series"]["quarterly"]
        else:
            logger.error("{} financial data not abled for stock {}".format(freq,stock))
            return  pd.DataFrame()
        for c in raw_dict.keys():
            raw_dict[c]={e["period"]:e["v"] for e in raw_dict[c]}
        raw_df=pd.DataFrame(raw_dict)
        raw_df.index=pd.to_datetime(raw_df.index)
        raw_df=raw_df.sort_index()
        raw_df1=raw_df.resample("m").mean().dropna(how="all")
        return  raw_df1
    def get_recomendation_trends(self,stock):
        url = "https://finnhub.io/api/v1/stock/recommendation?symbol={}&token={}".format(stock, self.api_key)
        data = self.session.get(url).content.decode("utf-8")
        data=json.loads(data)
        if len(data)==0:
            logger.error("Recommendation trends not abled for stock {}".format(stock))
            return pd.DataFrame()
        elif isinstance(data,dict) and "error" in data.keys():
            logger.error("Recommendation trends not abled for stock {}".format(stock))
            return pd.DataFrame()
        data=pd.DataFrame(data).set_index("period")
        data.index=pd.to_datetime(data.index)
        data=data.sort_index()
        return  data
    def get_technical_indicators(self,stock,indicators,resolution="1d",from_='2021-01-01',to='2022-12-01'):
        from_=int(utils.timestamp_from_string(from_))
        to=int(utils.timestamp_from_string(to))
        data_total=None
        dict_resolutions={"1d":"D","30m":30}
        resolution_api=dict_resolutions[resolution]
        for indicator in indicators:
            url = "https://finnhub.io/api/v1/indicator?symbol={}&resolution={}&indicator={}&token={}&timeperiod=1&from={}&to={}".\
                format(stock,resolution_api,indicator,self.api_key,from_,to)
            data = self.session.get(url).content.decode("utf-8")
            data = json.loads(data)
            if "t" in data.keys():
                data=pd.DataFrame({"time":data["t"],indicator:data[indicator]})
                data=data.set_index("time")
                data.index=pd.to_datetime(data.index,unit="s")
                if data_total is None:
                    data_total=data
                else:
                    data_total=pd.merge(data,data_total,right_index=True,left_index=True)
        if data_total is not None:
            data_total = data_total.resample(resolution.replace("m","Min")).mean().dropna(how="all")
        else:
            logger.error("Technical indicators not abled for stock {}".format(stock))
            return pd.DataFrame()
        return data_total


class Polygon(Api):
    def __init__(self,api_key):
        super().__init__(api_key)

    #TO DO
    def  get_financial_data(self,stock,freq="quarterly",limit=10):
        url="https://api.polygon.io/vX/reference/financials?ticker={}&apiKey={}&timeframe={}&limit={}".format(stock,self.api_key,freq,limit)
        data = self.session.get(url).json()
        return None

class YFinance(Api):
    def __init__(self, api_key=""):
        self.index_sources={
            "sp500":si.tickers_sp500,
            "nasdaq":si.tickers_nasdaq,
            "dow30":si.tickers_dow,
            "ftse100":si.tickers_ftse100,
            "nifty50":si.tickers_nifty50
        }
        super().__init__(api_key)
    def get_tickers(self,index):

        return self.index_sources[index]()
    def get_stock_prices(self,stock,interval="1d",range="100d",close="adjusted",events_splits=False,from_=None,to=None):
        if from_ is None or to is None:
            url="https://query1.finance.yahoo.com/v8/finance/chart/{}?interval={}&range={}&close={}".format(stock,interval,range,close)
        else:
            from_=int(utils.timestamp_from_string(from_))
            to=int(utils.timestamp_from_string(to))
            url = "https://query1.finance.yahoo.com/v8/finance/chart/{}?interval={}&close={}&period1={}&period2={}".format(stock,interval,
                                                                                                              close,from_,to)
        if events_splits:
            url+="&events=div%7Csplit"
        data=self.session.get(url,headers={'User-agent': 'Mozilla/5.0'})
        data=data.content.decode("utf-8")
        data = json.loads(data)["chart"]
        if "result" not in data.keys() or data["result"] is None  or len(data["result"])==0:
            logger.error("Stock prices not abled for stock {}".format(stock))
            return pd.DataFrame()
        data=data["result"][0]
        if "timestamp" not in data.keys():
            logger.error("Stock prices not abled for stock {}".format(stock))
            return pd.DataFrame()
        time=pd.to_datetime(np.array(data["timestamp"]),unit="s")
        quotes = pd.DataFrame.from_dict(data["indicators"]["quote"][0], orient="columns")
        quotes["time"] = time
        quotes=quotes.set_index("time")
        if interval=="1d":
            adj_close=data["indicators"]["adjclose"][0]["adjclose"]
            quotes["adj_close"]=adj_close
        quotes=quotes.resample(interval.replace("m","Min")).mean().dropna(how="all")
        return quotes

    def get_options(self,symbol):
        url="https://query1.finance.yahoo.com/v7/finance/options/{}".format(symbol)
        data = self.session.get(url, headers={'User-agent': 'Mozilla/5.0'})
        data = data.content.decode("utf-8")
        data = json.loads(data)
        if len(data["optionChain"]["result"])>0 and  len(data["optionChain"]["result"][0]["options"])>0:
            exp_dates=data["optionChain"]["result"][0]["expirationDates"]
            options={"calls":None,"puts":None}
            for exp_date in exp_dates:
                url = "https://query1.finance.yahoo.com/v7/finance/options/{}?date={}".format(symbol,exp_date)
                data = self.session.get(url, headers={'User-agent': 'Mozilla/5.0'})
                data = data.content.decode("utf-8")
                data = json.loads(data)
                data=data["optionChain"]["result"][0]["options"][0]
                for option_type in options.keys():
                    df=pd.DataFrame(data[option_type])
                    if "expiration" in df.columns:
                        df["expiration"]=pd.to_datetime(df["expiration"],unit="s")
                        df=df.set_index("expiration")
                        if  options[option_type] is None:
                            options[option_type]=df
                        else:
                            options[option_type] = pd.concat([options[option_type], df], axis=0)


            return options["calls"],options["puts"]
        logger.error("Options info not abled for stock {}".format(symbol))
        return pd.DataFrame(),pd.DataFrame()

    def get_actual_financial_data(self, stocks,indice):
        df=None
        for stock in stocks:
            data=self.get_actual_financial_data_(stock)
            if data.shape[0]>0:
                if df is None:
                    df=pd.DataFrame(columns=data.index)

                df.loc[stock]=data
        df["indice"] = indice
        df=df.set_index("indice",append=True)
        return df
    def get_stock_profile(self,stocks,indice):
        df = None
        for stock in stocks:
            data = self.get_stock_profile_(stock)
            if data.shape[0] > 0:
                if df is None:
                    df = pd.DataFrame(columns=data.index)

                df.loc[stock] = data
        df["indice"] = indice
        df = df.set_index("indice", append=True)
        return df

    def get_stock_profile_(self, stock):
        url_sector = "https://query1.finance.yahoo.com/v10/finance/quoteSummary/{}?modules=assetProfile".format(stock)
        data = self.session.get(url_sector, headers={'User-agent': 'Mozilla/5.0'})
        data = data.content.decode("utf-8")
        data = json.loads(data)
        if data["quoteSummary"]["result"] is  None or len( data["quoteSummary"]["result"])==0:
            logger.error("Stock profile not abled for stock {}".format(stock))
            return pd.Series()
        data=pd.Series(data["quoteSummary"]["result"][0]["assetProfile"])
        return data

    def get_actual_financial_data_(self, stock, ):
        url="https://query1.finance.yahoo.com/v11/finance/quoteSummary/{}?modules=financialData".format(stock)

        data = self.session.get(url, headers={'User-agent': 'Mozilla/5.0'})
        data = data.content.decode("utf-8")
        data = json.loads(data)
        if not isinstance(data["quoteSummary"]["result"],list):
            logger.error("Actual financial summary not abled for stock {}".format(stock))
            return pd.Series()
        data=data["quoteSummary"]["result"][0]["financialData"]
        for key,value in data.items():
            if isinstance(value,dict) and "raw" in value.keys():
                data[key]=value["raw"]
            elif isinstance(value, dict):
                data[key]=np.nan
        data= pd.Series(data)
        data=data.transform(lambda x: float(x.replace("%",""))*0.01 if isinstance(x,str) and "%" in x else x)
        data=data.transform(lambda x: float(x.replace("M",""))*1000000 if isinstance(x,str) and "M" in x else x)
        return data

    def get_actual_summary(self,stocks,indice):
        url="https://query1.finance.yahoo.com/v7/finance/quote?symbols={}".format(",".join(stocks))
        data = self.session.get(url, headers={'User-agent': 'Mozilla/5.0'})
        data = data.content.decode("utf-8")
        data = json.loads(data)
        data=pd.DataFrame(data["quoteResponse"]["result"])
        data=data.set_index("symbol")
        data["indice"]=indice
        data = data.set_index("indice", append=True)
        return data

    def get_earnings(self,stock):
        url="https://query1.finance.yahoo.com/v10/finance/quoteSummary/{}?modules={}".format(stock,",".join(["earnings","earningsHistory","earningsTrend"]))
        data = self.session.get(url, headers={'User-agent': 'Mozilla/5.0'})
        data = data.content.decode("utf-8")
        data = json.loads(data)
        if not isinstance(data["quoteSummary"]["result"],list):
            logger.error("Earnings not abled for stock {}".format(stock))
            return pd.DataFrame(),pd.DataFrame(),pd.DataFrame()
        data=data["quoteSummary"]["result"][0]
        earnings_q=pd.DataFrame(data["earningsHistory"]["history"]).applymap(lambda x: x["raw"] if isinstance(x,dict) and "raw" in x.keys() else x)
        earnings_a=pd.DataFrame(data["earnings"]["financialsChart"]["yearly"]).applymap(lambda x: x["raw"] if isinstance(x,dict)  and "raw" in x.keys()  else x)
        earnings_a=earnings_a.applymap(lambda x: np.nan if isinstance(x,dict)    else x)
        earnings_q=earnings_q.applymap(lambda x: np.nan  if isinstance(x,dict)    else x)
        for e in data["earningsTrend"]["trend"]:

            for k in ["earningsEstimate", "revenueEstimate", "epsTrend", "epsRevisions"]:
                reform = {(k,outerKey, innerKey): values for outerKey, innerDict in e[k].items() for innerKey, values in
                          innerDict.items()}
                e.pop(k)
                e.update(reform)
        earnings_trend=pd.DataFrame(data["earningsTrend"]["trend"])
        earnings_trend=earnings_trend.set_index(["period","endDate"]).drop(["maxAge","growth"],axis=1)
        earnings_trend.columns = pd.MultiIndex.from_tuples(earnings_trend.columns)
        earnings_trend=earnings_trend.loc[:,(slice(None),slice(None),"raw")].droplevel(2,axis=1)
        earnings_q["quarter"]=pd.to_datetime(earnings_q["quarter"],unit="s")
        return  earnings_a,earnings_q,earnings_trend


class PandasDataReader():
    def get_index(self,index=[]):
        index = web.DataReader(index, 'fred')
        return index.dropna(how="all")
