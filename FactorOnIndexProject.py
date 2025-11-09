# get all total price data
from Barra_cne5.barra_cne5_factor import GetData

total_df = GetData.get_price_()



#preprocess(check col name, dropduplicate, sort, shift, )
def preprocess_total_data(df):
    df['open_fq'] = df['open'] * df['factor']
    df['close_fq'] = df['close'] * df['factor']
    df['open_fwd'] = df['open'].shift(-1)
    df['close_fwd'] = df['close'].shift(-1)
    df['high_limit_fwd'] = df['high_limit'].shift(-1)
    df['low_limit_fwd'] = df['low_limit'].shift(-1)
    df['factor_fwd'] = df['factor'].shift(-1)
    df['open_fwd_fq'] = df['open_fwd'] * df['factor_fwd']
    df['close_fwd_fq'] = df['close_fwd'] * df['factor_fwd']
    df['open_tm1B'] = df['open'].shift()
    df['close_tm1B'] = df['close'].shift()
    df['factor_tm1B'] = df['factor'].shift()

    return df

g_stock = total_df.groupby('stock_code', group_keys=False)
total_data = g_stock.apply(preprocess_total_data)

# get trade date
# preprocess(check col name, dropduplicate, sort, shift, )
index_total = GetData._get_index()
index_target = index_total[index_total['stock_code']== '000300.XSHG'].copy()
index_target['datetime'] = pd.to_datetime(index_target['trade_date'])
index_target['year'] = index_target['datetime'].dt.year
index_target['month'] = index_target['datetime'].dt.month
index_target['week'] = index_target['datetime'].dt.isocalendar().week
index_target['trade_date'] = index_target['datetime'].dt.date
trade_date = pd.DataFrame(indexv1[['trade_date','year','week','month']])


def get_trade_date(df, start_year, period=None):
    df = df[df['year'] >= start_year]
    period_date = None
    if period in ['week', 'month']:
        period_date = df.groupby(['year','week'])['trade_date'].last()
    elif type(period) == int:
        period_date = df[::period]['trade_date']

    return pd.DataFrame(period_date)


# get asked freq trade date (func)
trade_date = get_trade_date(trade_date, START_YEAR, period='week')

# get factor  include barra

preprocess factor file

# factor file



# merge trade date

main trade_date(freq)
merge factor df and barra df get factor explode at specific freq

merge total_price df to get ret and forcast ret


# single factor test(all func writen)
exclude st. paused , limit up and down

starting date


# index enhance

get index relation file

merge trade date
merge index data to get index ret
merge total data and get each date ret and stock ret corellation (specific freq)




