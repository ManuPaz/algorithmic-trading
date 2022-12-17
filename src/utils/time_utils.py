import datetime
import datetime

import pandas as pd
from dateutil import  relativedelta
import numpy as np
def validate_datetime(date_text):
    try:
        datetime.datetime.strptime(date_text, '%Y-%m-%d %H:%M:%S')
        return date_text
    except ValueError:
        pass
    try:
        date_text=datetime.datetime.strptime(date_text, '%Y-%m-%d').strftime('%Y-%m-%d %H:%M:%S')
        return  date_text
    except ValueError:
        pass

def timestamp_from_string(string):
    string=validate_datetime(string)
    element = datetime.datetime.strptime(string, '%Y-%m-%d %H:%M:%S')
    timestamp = datetime.datetime.timestamp(element)
    return timestamp


def generate_random_intervals(start_date,end_date,train_duration=90,test_duration=20,frequency="d"):


    current_date = start_date
    if frequency == "d":
        current_date =     current_date + relativedelta.relativedelta(
            days=int(np.random.randint(0, int(train_duration / 3), 1)[0]))
    elif frequency == "h":
        current_date =    current_date + relativedelta.relativedelta(
            hours=int(np.random.randint(0, int(train_duration / 3), 1)[0]))

    intervals=pd.DataFrame(columns=["start_train","end_train","start_test","end_test"])
    while current_date+datetime.timedelta(days=train_duration)+ datetime.timedelta(days=test_duration) <= end_date:
        train_start_date = current_date
        train_end_date = train_start_date + datetime.timedelta(days=train_duration)

        # Calcula el segundo intervalo de fechas sumando la duración del período de prueba a la fecha final del primer intervalo
        test_start_date = train_end_date+ datetime.timedelta(days=1)
        test_end_date = test_start_date + datetime.timedelta(days=test_duration)
        intervals.loc[intervals.shape[0],:]=[train_start_date,train_end_date,test_start_date,test_end_date]
        # Actualiza la fecha actual como la fecha final del segundo intervalo
        if frequency=="d":
            current_date = test_end_date+relativedelta.relativedelta(days=int(np.random.randint(1,int(train_duration/3),1)[0]))
        elif frequency=="h":
            current_date = test_end_date + relativedelta.relativedelta(hours=int(np.random.randint(1, int(train_duration / 3), 1)[0]))


    return  intervals