
import os
os.chdir("../../")
from src.functions import stocks_module
import pickle
if __name__=="__main__":
    with open("resources/stocks_summary.obj","rb") as stocks_summary_file:
        stocks_summary=pickle.load(stocks_summary_file)

    data_filtered=stocks_summary.init_filtering(stocks_summary.fundamental_summary_complete,indice="russell2000")
    data_filtered=stocks_summary.filter_columns(data_filtered)
    data_filtered