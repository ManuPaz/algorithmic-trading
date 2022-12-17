import logging.config
logging.config.fileConfig('resources/logging.conf')
logger = logging.getLogger('api_error')
logger2 = logging.getLogger('general')
def drop_rows_with_nas(df, columns):
    if columns=="all":
        columns=list(df.columns)
    logger.error("NAS in df {}".format(df.loc[:,columns].isna().sum()))
    df=df.dropna(subset=columns)
    return df
def drop_zeros(df):
    variance=df.var()
    cols=variance.loc[variance==0].index
    logger2.info("Cols dropped because of low variance {}".format(cols))
    return df.drop(cols,axis=1)
def drop_columns_with_many_nas(df,lim_nas=0.2,lim_consecutive_nas=0.08):
    df = df.dropna(thresh=int(len(df) * (1-lim_nas)), axis=1,)
    first_returns=df.iloc[0:int(lim_consecutive_nas*len(df))]
    nas_in_first_returns = first_returns.isna().sum()/first_returns.shape[0]
    nas_in_first_returns = nas_in_first_returns.loc[nas_in_first_returns == 1]
    last_returns = df.iloc[-int(lim_consecutive_nas * len(df)):]
    nas_in_last_returns = last_returns.isna().sum() / last_returns.shape[0]
    nas_in_last_returns = nas_in_last_returns.loc[nas_in_last_returns == 1]
    cols_drop=list(set(nas_in_first_returns.index).union(set(nas_in_last_returns.index)))
    return df.drop(cols_drop,axis=1)