import datetime


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