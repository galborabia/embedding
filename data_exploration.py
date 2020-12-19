import pandas as pd
import numpy as np
import datetime as dt
from csv import writer
import os.path
from os import path

def split_data_by_year(path_to_data,path_to_output):
    chunksize = 10 ** 6
    temp_data=None
    year=2013
    for chunk in pd.read_csv(path_to_data, chunksize=chunksize):
        tail_row=chunk.tail(1)
        tail_date=tail_row['date']
        tail_year=tail_date.values[0]
        date_split=tail_year.split('-')
        chunk_year=int(date_split[0])
        if chunk_year==year:
            if temp_data is None:
                temp_data=pd.DataFrame(chunk)
            else:
                temp_data=temp_data.append(chunk)
        else:
            chunk['date'] = pd.to_datetime(chunk['date'])
            include = chunk[chunk['date'].dt.year == year]
            exclude = chunk[chunk['date'].dt.year == year+1]
            if temp_data is None:
                temp_data=pd.DataFrame(include)
            else:
                temp_data=temp_data.append(include)
            temp_data.to_csv(path_to_output+'\\'+str(year)+".csv",index=False)
            year+=1
            if len(exclude)>0:
                temp_data=pd.DataFrame(exclude)
    temp_data.to_csv(path_to_output + '\\' + str(year) + ".csv",index=False)

def create_oil_csv(path_to_data,output_path):
    average_oil_price1=[]
    average_oil_price2 = []
    new_data=[]
    oil_df=pd.read_csv(path_to_data)
    month=1
    first_day=None
    second_day=None
    for index,row in oil_df.iterrows():
        if np.isnan(row['dcoilwtico']):
            continue
        date=row['date']
        date_split = date.split('-')
        month_row=int(date_split[1])
        day_row =int(date_split[2])
        if month==month_row:
            if day_row <=15:
                average_oil_price1.append(row['dcoilwtico'])
                if first_day is None:
                    first_day=row['date']
            else:
                average_oil_price2.append(row['dcoilwtico'])
                if second_day is None:
                    second_day=row['date']
        else:
            new_data.append([first_day,np.mean(average_oil_price1), np.std(average_oil_price1)])
            new_data.append([second_day,np.mean(average_oil_price2), np.std(average_oil_price2)])
            average_oil_price1=[]
            average_oil_price2=[]
            first_day=None
            second_day=None
            month+=1
            month=month%13
            if month==0:
                month+=1
            if day_row<=15:
                average_oil_price1.append(row['dcoilwtico'])
                first_day=row['date']
            else:
                average_oil_price2.append(row['dcoilwtico'])
                second_day=row['date']
    oil_df=pd.DataFrame(new_data)
    oil_df.to_csv(path_to_output)

def transform_date_to_day(data,column='date'):
    week_days=[]
    months=[]
    years=[]
    for index,row in data.iterrows():
        date = row['date']
        day=date.weekday()
        month=date.month
        year=date.year
        week_days.append(day)
        months.append(month)
        years.append(year)
    data['day']=week_days
    data['month'] = months
    data['year'] = years
    return data


def write_chunk_to_csv(new_data,output_path):
    if(path.exists(output_path)):
        df=pd.DataFrame(new_data)
        df.to_csv(output_path,mode='a',header=False,index=False)
    else:
        df=pd.DataFrame(new_data)
        df.to_csv(output_path,index=False)


def create_holiday_csv(holiday_path,data_path,output_path,year):
    holiday_df=pd.read_csv(holiday_path)
    holiday_dict={}
    for index,row in holiday_df.iterrows():
        date = dt.datetime.strptime(row['date'],'%Y-%m-%d')
        if date.year==year:
            holiday_dict[date]=row
    chunksize = 10 ** 5
    for chunk in pd.read_csv(data_path, chunksize=chunksize):
        new_data = []
        for index,row in chunk.iterrows():
            if len (row['date'].split())>1:
                split=row['date'].split()
                row['date']=split[0]
            date = dt.datetime.strptime(row['date'], '%Y-%m-%d')
            is_before_holiday_week=False
            is_after_holiday_week=False
            is_holiday_day=False
            holiday_type='none'
            holiday_locale='none'
            holiday_locale_name='none'
            transferred=False
            for holiday in holiday_dict.keys():
                if holiday==date:
                    is_holiday_day=True
                    holiday_det=holiday_dict.get(holiday)
                    holiday_type=holiday_det['type']
                    holiday_locale=holiday_det['locale']
                    holiday_locale_name=holiday_det['locale_name']
                    transferred=holiday_det['transferred']
                if abs(holiday-date).days < 7 :
                    if holiday  < date:
                        is_before_holiday_week=True
                    else:
                        is_after_holiday_week=True
            day=date.weekday()
            month=date.month
            payment_week=False
            if date.day < 8 or (date.day > 15 and date.day < 22):
                payment_week=True
            new_data.append([row['id'],row['date'],row['store_nbr'],row['item_nbr'],row['unit_sales'],row['onpromotion'],is_before_holiday_week,is_after_holiday_week,is_holiday_day,holiday_type,holiday_locale,holiday_locale_name,payment_week])
        write_chunk_to_csv(new_data,output_path)


def join_stores_transaction(transaction_path,stores_path,output_path):
    stores=pd.read_csv(stores_path)
    transaction=pd.read_csv(transaction_path)
    merge_df=pd.merge(stores,transaction,on="store_nbr")
    merge_df.to_csv(output_path,index=False)


def calculate_item_ave_per_weekday(items_path,output_path,year):
    chunk_size = 10 ** 6
    session_chunks = None
    current_session = dt.date(year,1,1)
    stop_session=dt.date(year,2,1)
    next_session = None
    for chunk in pd.read_csv(items_path, chunksize=chunk_size):
        chunk['date'] = pd.to_datetime(chunk['date'])
        chunk['date'] = chunk['date'].apply(lambda x: x.date())
        mask = (chunk['date'] >= current_session) & (chunk['date'] < stop_session)
        session_time = chunk.loc[mask]
        flag = True
        tail_date = chunk.tail(1)['date'].values[0]
        print(tail_date)
        if session_chunks is None:
            session_chunks = pd.DataFrame(session_time)
            flag = False
        if flag:
            frames = [session_chunks, session_time]
            session_chunks = pd.concat(frames)
        if tail_date >= stop_session:
            next_session = chunk[chunk['date'] >= stop_session]
            if len(next_session) == 0:
                next_session = None
            session_chunks['day'] = session_chunks['date'].apply(lambda x: x.weekday())
            session_chunks['month'] = session_chunks['date'].apply(lambda x: x.month)
            session_chunks['year'] = session_chunks['date'].apply(lambda x: x.year)
            gb_store_item = session_chunks.groupby(by=['store_nbr','item_nbr','day','month','year']).agg({'unit_sales': ['sum','mean','std','median']})
            gb_item = session_chunks.groupby(by=['item_nbr','day','month','year']).agg({'unit_sales': ['sum','mean','std','median']})
            write_chunk_to_csv(gb_store_item, output_path=output_path + "//gb_stores_item_date"+str(year)+".csv")
            write_chunk_to_csv(gb_item, output_path=output_path + "//gb_item_date"+str(year)+".csv")
            session_chunks = None
            chunks=None
            if not next_session is None:
                session_chunks = pd.DataFrame(next_session)
                next_session = None
            current_session = stop_session
            month = stop_session.month + 1
            if month>12:
                month=1
            stop_session = dt.date(year, month, 1)
    if session_chunks is None:
        return
    else:
        chunks = pd.DataFrame(session_chunks)
        chunks = transform_date_to_day(chunks)
        gb_store_item = chunks.groupby(by=['store_nbr', 'item_nbr', 'day','month','year']).agg({'unit_sales': ['sum','mean','std','median']})
        gb_item = chunks.groupby(by=['item_nbr', 'day','month','year']).agg({'unit_sales': ['sum','mean','std','median']})
        write_chunk_to_csv(gb_store_item, output_path=output_path + "//gb_stores_item_date"+str(year)+".csv")
        write_chunk_to_csv(gb_item, output_path=output_path + "//gb_item_date"+str(year)+".csv")

def calculate_item_ave_per_two_weeks(items_path,output_path,year):
    chunk_size=10**6
    session_chunks = None
    start_date=dt.date(year,1,1)
    end_date=dt.date(year,1,15)
    next_session=None
    for chunk in pd.read_csv(items_path,chunksize=chunk_size):
        chunk['date'] = pd.to_datetime(chunk['date'])
        chunk['date'] = chunk['date'].apply(lambda x: x.date())
        mask = (chunk['date'] >= start_date) & (chunk['date'] <= end_date)
        session_time = chunk.loc[mask]
        flag=True
        tail_date=chunk.tail(1)['date'].values[0]
        print(tail_date)
        if session_chunks is None:
            session_chunks = pd.DataFrame(session_time)
            flag=False
        if flag:
            frames=[session_chunks,session_time]
            session_chunks=pd.concat(frames)
        if tail_date>end_date:
            next_session=chunk[chunk['date']>end_date]
            if len(next_session)==0:
                next_session=None
            gb_store_item = session_chunks.groupby(by=['store_nbr', 'item_nbr']).agg({'unit_sales':['sum','mean','std','median']})
            gb_item = session_chunks.groupby(by='item_nbr').agg({'unit_sales':['sum','mean','std','median']})
            gb_store_item['month'] = start_date.month
            gb_item['month'] = start_date.month
            gb_store_item['date_day'] = start_date.day
            gb_item['date_day'] = start_date.day
            write_chunk_to_csv(gb_store_item, output_path=output_path + "//gb_stores_item"+str(year)+".csv")
            write_chunk_to_csv(gb_item, output_path=output_path + "//gb_item"+str(year)+".csv")
            session_chunks=None
            if not next_session is None:
                session_chunks= pd.DataFrame(next_session)
                next_session =None
            if start_date.day==1:
                start_date=dt.date(year,end_date.month,16)
                new_month=start_date.month+1
                if new_month>12:
                    new_month=1
                    year_t=year +1
                    end_date=dt.date(year_t,new_month,1)
                else:
                    end_date = dt.date(year, new_month, 1)
                end_date = end_date - dt.timedelta(days=1)
            else:
                new_month=start_date.month+1
                if new_month>12:
                    new_month=12
                    start_date = dt.date(year,new_month,16)
                    end_date = dt.date(year+1, 1, 1)
                else:
                    start_date = dt.date(year, new_month, 1)
                    end_date = dt.date(year,new_month,15)
    if session_chunks is None:
        return
    else:
        chunks = pd.DataFrame(session_chunks)
        gb_store_item = chunks.groupby(by=['store_nbr', 'item_nbr']).agg({'unit_sales': ['sum','mean','std','median']})
        gb_item = chunks.groupby(by='item_nbr').agg({'unit_sales': ['sum','mean','std','median']})
        gb_store_item['month']=start_date.month
        gb_item['month']=start_date.month
        gb_store_item['date_day'] = start_date.day
        gb_item['date_day'] = start_date.day
        write_chunk_to_csv(gb_store_item, output_path=output_path + "//gb_stores_item"+str(year)+".csv")
        write_chunk_to_csv(gb_item, output_path=output_path + "//gb_item"+str(year)+".csv")

# session=r"C:\Users\gal\Desktop\ISE\semesterG\deep learning workshop\assignment2\Embedding\2016_combined.csv"
# output=r"C:\Users\gal\Desktop\ISE\semesterG\deep learning workshop\assignment2\Embedding"
# calculate_item_ave_per_two_weeks(session,output,2016)

# session=r"C:\Users\gal\Desktop\ISE\semesterG\deep learning workshop\assignment2\Embedding\2015.csv"
# output=r"C:\Users\gal\Desktop\ISE\semesterG\deep learning workshop\assignment2\Embedding"
# calculate_item_ave_per_two_weeks(session,output,2015)

def check_values(path,date_s,date_f,item,store):
    chunk_size = 10 ** 6
    date_start = date_s
    date_finish=date_f
    session_chunks=None
    for chunk in pd.read_csv(path, chunksize=chunk_size):
        chunk['date'] = pd.to_datetime(chunk['date'])
        chunk['date'] = chunk['date'].apply(lambda x: x.date())
        tail_date = chunk.tail(1)['date'].values[0]
        head_date = chunk.head(1)['date'].values[0]
        if tail_date<date_start:
            continue
        elif head_date>date_finish:
            break
        elif tail_date>=date_start or head_date<=date_finish:
            mask = (chunk['date'] >= date_start) & (chunk['date'] <= date_finish)
            session_time=chunk.loc[mask]
            if session_chunks is None:
                session_chunks=pd.DataFrame(session_time)
            else:
                frames=[session_chunks,session_time]
                session_chunks=pd.concat(frames)
            if tail_date>date_finish:
                break

    item_list=session_chunks[session_chunks['item_nbr'] ==item]
    store_list=item_list[item_list['store_nbr']==store]
    data=store_list.agg({'unit_sales':['sum','mean','std','median']})
    print(data.head(4))

def check_dates_values(path,date_s,date_f,item,store,day):
    chunk_size = 10 ** 6
    date_start = date_s
    date_finish=date_f
    session_chunks=None
    for chunk in pd.read_csv(path, chunksize=chunk_size):
        chunk['date'] = pd.to_datetime(chunk['date'])
        chunk['date'] = chunk['date'].apply(lambda x: x.date())
        tail_date = chunk.tail(1)['date'].values[0]
        head_date = chunk.head(1)['date'].values[0]
        if tail_date<date_start:
            continue
        elif head_date>date_finish:
            break
        elif tail_date>=date_start or head_date<=date_finish:
            mask = (chunk['date'] >= date_start) & (chunk['date'] <= date_finish)
            session_time=chunk.loc[mask]
            if session_chunks is None:
                session_chunks=pd.DataFrame(session_time)
            else:
                frames=[session_chunks,session_time]
                session_chunks=pd.concat(frames)
            if tail_date>date_finish:
                break
    item_list=session_chunks[session_chunks['item_nbr'] ==item]
    store_list=item_list[item_list['store_nbr']==store]
    store_list=transform_date_to_day(store_list)
    day_list=store_list[store_list['day']==day]
    # item_list.to_csv(r"C:\Users\gal\Desktop\ISE\semesterG\deep learning workshop\assignment2\Embedding\\itemlist.csv")
    data=day_list.agg({'unit_sales':['sum','mean','std','median']})
    print(data.head(4))


def create_transactions_csv(path_to_data,output_path):
    transaction_df=pd.read_csv(path_to_data)
    transaction_df=transform_date_to_day(transaction_df)
    group_by_store=transaction_df.groupby(['store_nbr','day','month','year']).agg({'transactions':['mean','min','max','std']})
    group_by_store.columns=['transaction_mean','transaction_min','transaction_max','transaction_std']
    group_by_store.to_csv(output_path)


def create_salary_csv(path_to_data,output_path):
    df=pd.read_csv(path_to_data)
    payments=[]
    for index,row in df.iterrows():
        date=row['date']
        split_date=date.split('-')
        day=int(split_date[2])
        if day<8 or ( day>15 and day<22):
            payments.append('T')
        else:
            payment.append('F')
    df['weekly_payment']=payments
    df.to_csv(output_path)

def create_store_plot(data):
    group_by_store = data.groupby('store_nbr').agg({'unit_sales': ['sum']})

def create_item_plot(data):
    group_by_item = chunks.groupby('item_nbr').agg({'unit_sales': ['sum']})

def present_data(date_start,date_finish,path):
    chunksize = 10 ** 5
    chunks=None
    for chunk in pd.read_csv(data, chunksize=chunksize):
        tail_date=chunk.tail(1)['date']
        head_date=chunk.head(1)['date']
        date_time_tail = dt.datetime.strptime(tail_date, '%Y-%m-%d')
        date_time_head = dt.datetime.strptime(head_date, '%Y-%m-%d')
        if date_start<date_time_tail:
            continue
        elif date_finish<date_time_tail:
            chunks=pd.DataFrame(chunk)
            break;
        else:
            if chunks is None:
                chunks=pd.DataFrame(chunk)
            else:
                chunks.append(chunk)
    create_item_plot(chunks)
    create_store_plot(chunks)



def calculate_last_session(session_path,last_session_path,output):
    session=pd.read_csv(session_path)
    session.fillna(0)
    session_before=pd.read_csv(last_session_path)
    session_before.fillna(0)
    items=session.loc[:,['item_nbr','month','month_day']].copy()
    merged=pd.merge(session_before,items,on=['item_nbr','month','month_day'])
    last_session_std=[]
    last_session_median = []
    last_session_mean = []
    last_session_sum = []
    counter=0
    for index,row in session.iterrows():
        item=row['item_nbr']
        if row['month'] == 1 and row['month_day'] == 1:
            value = session_before.loc[(session_before['item_nbr'] == item) & (session_before['month_day']  == 16) & (session_before['month'] == 12)]
        elif row['month_day'] == 1:
            month=row['month']-1
            value = session.loc[(session['item_nbr'] == item) & (session['month_day'] == 16) & (session['month'] == month)]
        else :
            month = row['month']
            value = session.loc[(session['item_nbr'] == item) & (session['month_day'] == 1) & (session['month'] == month)]
        if len(value) >0:
            last_session_std.append(value['std'].values[0])
            last_session_median.append(value['median'].values[0])
            last_session_mean.append(value['mean'].values[0])
            last_session_sum.append(value['sum'].values[0])
        else:
            print(item)
            counter+=1
            last_session_std.append(-1)
            last_session_median.append(-1)
            last_session_mean.append(-1)
            last_session_sum.append(-1)
    session['last_session_std']=last_session_std
    session['last_session_median']=last_session_median
    session['last_session_mean']=last_session_mean
    session['last_session_sum']=last_session_sum
    session.to_csv(session_path)
    merged.to_csv(output)
    print("missing values season before: "+str(counter))



def merge_df_by_key(df_1,df_2,output):
    last_session_std = []
    last_session_median = []
    last_session_mean = []
    last_session_sum = []
    counter=0
    df_2.set_index(keys=['item_nbr','month_day','month'])
    for index, row in df_1.iterrows():
        item=row['item_nbr']
        month_day=row['month_day']
        month=row['month']
        value = df_2.loc[(df_2['item_nbr'] == item) & (df_2['month_day'] == month_day) & (df_2['month'] == month)]
        if len(value) >0:
            last_session_std.append(value['year_before_std'].values[0])
            last_session_median.append(value['year_before_median'].values[0])
            last_session_mean.append(value['year_before_mean'].values[0])
            last_session_sum.append(value['year_before_sum'].values[0])
        else:
            print(item)
            counter+=1
            last_session_std.append(-1)
            last_session_median.append(-1)
            last_session_mean.append(-1)
            last_session_sum.append(-1)
    df_1['year_before_std'] = last_session_std
    df_1['year_before_median'] = last_session_median
    df_1['year_before_mean'] = last_session_mean
    df_1['year_before_sum'] = last_session_sum
    df_1.to_csv(output)


# path1=r"C:\Users\gal\Desktop\ISE\semesterG\deep learning workshop\assignment2\Embedding\gb_item20161.csv"
# df_1=pd.read_csv(path1)
# path2=r"C:\Users\gal\Desktop\ISE\semesterG\deep learning workshop\assignment2\Embedding\year_beforedata2016.csv"
# df_2=pd.read_csv(path2)
# output=r"C:\Users\gal\Desktop\ISE\semesterG\deep learning workshop\assignment2\Embedding\merge_sessions_2016.csv"
# merge_df_by_key(df_1,df_2,output)

def fill_missing_values_sessions(data,last_season,year,output):
    last_season.set_index(keys=['item_nbr','month','month_day'])
    for index,row in data.iterrows():
        if row['last_session_sum']==-1:
            item=row['item_nbr']
            if row['month']==1:
                value = last_season.loc[(last_season['item_nbr'] == item) & (last_season['month_day'] == 1) & (last_season['month'] == 12)]
                if len(value) > 0:
                    data.loc[(data['item_nbr'] == item) & (data['month_day'] == row['month_day']) & (data['month'] == row['month']),'last_session_sum' ] = value['sum'].values[0]
                    data.loc[(data['item_nbr'] == item) & (data['month_day'] == row['month_day']) & (data['month'] == row['month']),'last_session_median' ] = value['median'].values[0]
                    data.loc[(data['item_nbr'] == item) & (data['month_day'] == row['month_day']) & (data['month'] == row['month']),'last_session_mean' ] = value['mean'].values[0]
                    data.loc[(data['item_nbr'] == item) & (data['month_day'] == row['month_day']) & (data['month'] == row['month']),'last_session_std' ] = value['std'].values[0]
                else:
                    data.loc[(data['item_nbr'] == item) & (data['month_day'] == row['month_day']) & (data['month'] == row['month']),'last_session_sum' ] = 0
                    data.loc[(data['item_nbr'] == item) & (data['month_day'] == row['month_day']) & (data['month'] == row['month']),'last_session_median' ] = 0
                    data.loc[(data['item_nbr'] == item) & (data['month_day'] == row['month_day']) & (data['month'] == row['month']),'last_session_mean' ] = 0
                    data.loc[(data['item_nbr'] == item) & (data['month_day'] == row['month_day']) & (data['month'] == row['month']),'last_session_std' ] = 0
            else:
                day=row['month_day']
                month=row['month']-1
                if day==1:
                    day=16
                else:
                    day=16
                    value = data.loc[(data['item_nbr'] == item) & (data['month_day'] == day) & (data['month'] == month)]
                if len(value) > 0:
                    data.loc[(data['item_nbr'] == item) & (data['month_day'] == row['month_day']) & (data['month'] == row['month']), 'last_session_sum'] = value['last_session_sum'].values[0]
                    data.loc[(data['item_nbr'] == item) & (data['month_day'] == row['month_day']) & (data['month'] == row['month']), 'last_session_median'] = value['last_session_median'].values[0]
                    data.loc[(data['item_nbr'] == item) & (data['month_day'] == row['month_day']) & (data['month'] == row['month']), 'last_session_mean'] = value['last_session_mean'].values[0]
                    data.loc[(data['item_nbr'] == item) & (data['month_day'] == row['month_day']) & (data['month'] == row['month']), 'last_session_std'] = value['last_session_std'].values[0]
                else:
                    data.loc[(data['item_nbr'] == item) & (data['month_day'] == row['month_day']) & (
                                    data['month'] == row['month']), 'last_session_sum'] = 0
                    data.loc[(data['item_nbr'] == item) & (data['month_day'] == row['month_day']) & (
                                    data['month'] == row['month']), 'last_session_median'] = 0
                    data.loc[(data['item_nbr'] == item) & (data['month_day'] == row['month_day']) & (
                                    data['month'] == row['month']), 'last_session_mean'] = 0
                    data.loc[(data['item_nbr'] == item) & (data['month_day'] == row['month_day']) & (
                                    data['month'] == row['month']), 'last_session_std'] = 0
    data.to_csv(output)

def fill_missing_values_year(data,last_season,output):
    last_season.set_index(keys=['item_nbr','month','month_day'])
    for index,row in data.iterrows():
        if row['year_before_sum']==-1:
            item=row['item_nbr']
            month=row['month']
            day=row['month_day']
            if month==1 and day==1 :
                data.loc[(data['item_nbr'] == item) & (data['month_day'] == row['month_day']) & (
                        data['month'] == row['month']), 'year_before_sum'] = 0
                data.loc[(data['item_nbr'] == item) & (data['month_day'] == row['month_day']) & (
                        data['month'] == row['month']), 'year_before_median'] = 0
                data.loc[(data['item_nbr'] == item) & (data['month_day'] == row['month_day']) & (
                        data['month'] == row['month']), 'year_before_mean'] = 0
                data.loc[(data['item_nbr'] == item) & (data['month_day'] == row['month_day']) & (
                        data['month'] == row['month']), 'year_before_std'] = 0
                continue
            if day==16:
                day=1
            else:
                day=16
                month=month-1
                if month==0:
                    day=1
                    month=1
            value = last_season.loc[(last_season['item_nbr'] == item) & (last_season['month_day'] == day) & (last_season['month'] == month)]
            if len(value) > 0:
                data.loc[(data['item_nbr'] == item) & (data['month_day'] == row['month_day']) & (
                            data['month'] == row['month']), 'year_before_sum'] = value['sum'].values[0]
                data.loc[(data['item_nbr'] == item) & (data['month_day'] == row['month_day']) & (
                            data['month'] == row['month']), 'year_before_median'] = value['median'].values[0]
                data.loc[(data['item_nbr'] == item) & (data['month_day'] == row['month_day']) & (
                            data['month'] == row['month']), 'year_before_mean'] = value['mean'].values[0]
                data.loc[(data['item_nbr'] == item) & (data['month_day'] == row['month_day']) & (
                            data['month'] == row['month']), 'year_before_std'] = value['std'].values[0]
            else:
                data.loc[(data['item_nbr'] == item) & (data['month_day'] == row['month_day']) & (
                            data['month'] == row['month']), 'year_before_sum'] = 0
                data.loc[(data['item_nbr'] == item) & (data['month_day'] == row['month_day']) & (
                            data['month'] == row['month']), 'year_before_median'] = 0
                data.loc[(data['item_nbr'] == item) & (data['month_day'] == row['month_day']) & (
                            data['month'] == row['month']), 'year_before_mean'] = 0
                data.loc[(data['item_nbr'] == item) & (data['month_day'] == row['month_day']) & (
                            data['month'] == row['month']), 'year_before_std'] = 0
    data.to_csv(output)

# path1=r"C:\Users\gal\Desktop\ISE\semesterG\deep learning workshop\assignment2\Embedding\gb_item2015.csv"
# df_2=pd.read_csv(path1)
# path2=r"C:\Users\gal\Desktop\ISE\semesterG\deep learning workshop\assignment2\Embedding\merge_sessions_fill_2016.csv"
# df_1=pd.read_csv(path2)
# output=r"C:\Users\gal\Desktop\ISE\semesterG\deep learning workshop\assignment2\Embedding\complete_merge_2016.csv"
# fill_missing_values_year(df_1,df_2,output)

def group_transaction_by_month(data,output):
    data['date'] = pd.to_datetime(data['date'])
    data['date'] = data['date'].apply(lambda x: x.date())
    data['day'] = data['date'].apply(lambda x: x.day)
    data['month'] = data['date'].apply(lambda x: x.month)
    data['year'] = data['date'].apply(lambda x: x.year)
    transactions_by_month=data.groupby(by=['month','year','store_nbr']).agg({'transactions': ['min','max','mean','std','median']})
    transactions_by_month.to_csv(output)


def set_stores_transactios_data(transactions,output):
    relevent=transactions.loc[transactions['year']>=2016]
    transactions.set_index(keys=['store_nbr','month'])
    for index,row in relevent.iterrows():
        month=row['month']
        year = row['year']
        # transactions year before
        last_year_transactions = transactions.loc[(transactions['store_nbr'] == row['store_nbr']) & (transactions['month'] == month) & (transactions['year'] == year-1)]
        if month==1:
            month=12
            year-=1
        else:
            month=month-1
        # transactions month before
        last_month_transactions = transactions.loc[(transactions['store_nbr'] == row['store_nbr']) & (transactions['month'] == month) & (transactions['year'] == year)]
        if len(last_year_transactions)>0:
            transactions.loc[(transactions['store_nbr'] == row['store_nbr']) & (transactions['month'] == row['month']) & (transactions['year'] == row['year']), 'last_year_transactions_min'] = last_year_transactions['transactions_min'].values[0]
            transactions.loc[(transactions['store_nbr'] == row['store_nbr']) & (transactions['month'] == row['month']) & (transactions['year'] == row['year']), 'last_year_transactions_max'] = last_year_transactions['transactions_max'].values[0]
            transactions.loc[(transactions['store_nbr'] == row['store_nbr']) & (transactions['month'] == row['month']) & (transactions['year'] == row['year']), 'last_year_transactions_mean'] = last_year_transactions['transactions_mean'].values[0]
            transactions.loc[(transactions['store_nbr'] == row['store_nbr']) & (transactions['month'] == row['month']) & (transactions['year'] == row['year']), 'last_year_transactions_std'] = last_year_transactions['transactions_std'].values[0]
            transactions.loc[(transactions['store_nbr'] == row['store_nbr']) & (transactions['month'] == row['month']) & (transactions['year'] == row['year']), 'last_year_transactions_median'] = last_year_transactions['transactions_median'].values[0]
        if len(last_month_transactions)>0:
            transactions.loc[(transactions['store_nbr'] == row['store_nbr']) & (transactions['month'] == row['month']) & (transactions['year'] == row['year']), 'last_month_transactions_min'] = last_month_transactions['transactions_min'].values[0]
            transactions.loc[(transactions['store_nbr'] == row['store_nbr']) & (transactions['month'] == row['month']) & (transactions['year'] == row['year']), 'last_month_transactions_max'] = last_month_transactions['transactions_max'].values[0]
            transactions.loc[(transactions['store_nbr'] == row['store_nbr']) & (transactions['month'] == row['month']) & (transactions['year'] == row['year']), 'last_month_transactions_mean'] = last_month_transactions['transactions_mean'].values[0]
            transactions.loc[(transactions['store_nbr'] == row['store_nbr']) & (transactions['month'] == row['month']) & (transactions['year'] == row['year']), 'last_month_transactions_std'] = last_month_transactions['transactions_std'].values[0]
            transactions.loc[(transactions['store_nbr'] == row['store_nbr']) & (transactions['month'] == row['month']) & (transactions['year'] == row['year']), 'last_month_transactions_median'] = last_month_transactions['transactions_median'].values[0]
    new_data=transactions.loc[transactions['year']>=2016]
    new_data.to_csv(output)





def merge_data_stores(data_path,stores,output):
    stores.set_index(keys=['store_nbr','month','year'])
    for chunk in pd.read_csv(data_path,chunksize=10 ** 6):
        chunk['date'] = pd.to_datetime(chunk['date'])
        chunk['date'] = chunk['date'].apply(lambda x: x.date())
        chunk['day'] = chunk['date'].apply(lambda x: 1 if x.day<16 else 16)
        chunk['month'] = chunk['date'].apply(lambda x: x.month)
        chunk['year'] = chunk['date'].apply(lambda x: x.year)
        chunk['weekday'] = chunk['date'].apply(lambda x: x.weekday())
        mergerd = chunk.merge(stores,how="left",on=['store_nbr','month','year'])
        write_chunk_to_csv(mergerd,output)


def merge_data_items(data_path,items,output):
    items.set_index(keys=['item_nbr','month','day'])
    for chunk in pd.read_csv(data_path,chunksize=10 ** 6):
        mergerd = chunk.merge(items,how="left",on=['item_nbr','month','day'])
        write_chunk_to_csv(mergerd,output)




def merge_data(data_path,items,output_categorical,output_numeric):
    items.set_index(keys='item_nbr')
    for chunk in pd.read_csv(data_path,chunksize=10 ** 6):
        chunk['date'] = pd.to_datetime(chunk['date'])
        chunk['date'] = chunk['date'].apply(lambda x: x.date())
        chunk['day'] = chunk['date'].apply(lambda x: x.day)
        mergerd = chunk.merge(items,how="left",on='item_nbr')
        categorical_data =  mergerd.loc[:,['id','date','store_nbr', 'item_nbr', 'onpromotion', 'before_holiday_week', 'is_after_holiday_week',
                          'is_holiday_day', 'holiday_type', 'holiday_locale', 'holiday_locale_name', 'payment_week','day',
                          'month', 'year', 'weekday', 'city', 'state', 'type', 'cluster','item_family','item_class','perishable','unit_sales']]

        numeric_data = mergerd.loc[:,['id', 'date', 'last_year_transactions_min', 'last_year_transactions_max', 'last_year_transactions_mean', 'last_year_transactions_std',
                          'last_year_transactions_median','last_month_transactions_min', 'last_month_transactions_max', 'last_month_transactions_mean', 'last_month_transactions_std',
                          'last_month_transactions_median','last_session_std', 'last_session_median', 'last_session_mean', 'last_session_sum', 'year_before_std', 'year_before_median', 'year_before_mean', 'year_before_sum']]
        write_chunk_to_csv(numeric_data,output_numeric)
        write_chunk_to_csv(categorical_data,output_categorical)
        categorical_data=None
        numeric_data=None

# data_path=r"C:\Users\gal\Desktop\ISE\semesterG\deep learning workshop\assignment2\Embedding\2017_complete_not_normalize.csv"
# stores=pd.read_csv(r"C:\Users\gal\Desktop\ISE\semesterG\deep learning workshop\assignment2\Embedding\items.csv")
# output_c=r"C:\Users\gal\Desktop\ISE\semesterG\deep learning workshop\assignment2\Embedding\categorical_data_2017.csv"
# output_n=r"C:\Users\gal\Desktop\ISE\semesterG\deep learning workshop\assignment2\Embedding\numeric_data_2017.csv"
# merge_data(data_path,stores,output_c,output_n)

# date = dt.date(2017,5,1)
# pathf=r"C:\Users\gal\Desktop\ISE\semesterG\deep learning workshop\assignment2\Embedding\categorical_data_2017.csv"
# last_three_months=r"C:\Users\gal\Desktop\ISE\semesterG\deep learning workshop\assignment2\Embedding\\last_three_months.csv"
# for chunk in pd.read_csv(pathf, chunksize=1000000):
#
#     tail_date=dt.datetime.strptime(chunk.tail(1)['date'].values[0], '%Y-%m-%d').date()
#     head_date = dt.datetime.strptime(chunk.head(1)['date'].values[0], '%Y-%m-%d').date()
#     if tail_date < date:
#         continue
#     elif head_date>=date:
#         write_chunk_to_csv(chunk,last_three_months)
#     else:
#         chunk['date'] = pd.to_datetime(chunk['date'])
#         chunk['date'] = chunk['date'].apply(lambda x: x.date())
#         data=chunk.loc[chunk['date']>date]
#         write_chunk_to_csv(data,last_three_months)
