
import os
os.chdir("../../")
from src.functions import stocks_module
import pickle
if __name__=="__main__":
    with open("resources/stocks_summary.obj","rb") as stocks_summary_file:
        stocks_summary=pickle.load(stocks_summary_file)

    data_filtered=stocks_summary.init_filtering(stocks_summary.fundamental_summary_complete,indice=None)
    data_filtered=stocks_summary. filter_target_prices(data_filtered)
    #data_filtered=stocks_summary.filter_columns(data_filtered)
    data_filtered=data_filtered[["targetLowPrice","targetHighPrice","currentPrice",
                                 "numberOfAnalystOpinions","targetMeanPrice",
                                 "targetMedianPrice"]]
    data_filtered["ratio"]=data_filtered["targetMedianPrice"]/ data_filtered["currentPrice"]
    data_filtered