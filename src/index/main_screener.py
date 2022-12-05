
import os
os.chdir("../../")
from src.functions import stocks_module
import pickle
if __name__=="__main__":
    with open("resources/stocks_summary.obj","rb") as stocks_summary_file:
        stocks_summary=pickle.load(stocks_summary_file)

    data_filtered=stocks_summary.init_filtering(stocks_summary.fundamental_summary_complete, indice="sp500")
    data_filtered=stocks_summary.filter_growth( data_filtered,)
    data_filtered=stocks_summary. filter_bigfall(data_filtered)
    data_filtered=stocks_summary.filter_ev_cashflow(data_filtered)
    data_filtered=stocks_summary.filter_debt(data_filtered)
    data_filtered=stocks_summary.filter_margin(data_filtered)
    data_filtered