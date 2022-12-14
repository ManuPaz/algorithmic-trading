import logging.config
logging.config.fileConfig('resources/logging.conf')
logger = logging.getLogger('api_error')

def delete_nas(df,columns):
    if columns=="all":
        columns=list(df.columns)
    logger.error("NAS in df {}".format(df.loc[:,columns].isna().sum()))
    df=df.dropna(subset=columns)
    return df
