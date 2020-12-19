from tensorflow.keras.layers import Input, Embedding, add,Flatten, concatenate,Dropout, Dense,BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import *
from sklearn.metrics import mean_squared_error,mean_absolute_error
from tensorflow.keras.optimizers import *
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import KFold,StratifiedKFold,train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

categorical_path=r"C:\Users\gal\Desktop\ISE\semesterG\deep learning workshop\assignment2\Embedding\\last_three_months.csv"
data = pd.read_csv(categorical_path)

item_nbr = {p:i for (i,p) in enumerate(data['item_nbr'].unique())}
item_class = {p:i for (i,p) in enumerate(data['item_class'].unique())}
item_family = {p:i for (i,p) in enumerate(data['item_family'].unique())}
perishable = {p:i for (i,p) in enumerate(data['perishable'].unique())}
onpromotion = {p:i for (i,p) in enumerate(data['onpromotion'].unique())}

store_nbr = {p:i for (i,p) in enumerate(data['store_nbr'].unique())}
store_type = {p:i for (i,p) in enumerate(data['type'].unique())}
store_cluster = {p:i for (i,p) in enumerate(data['cluster'].unique())}
store_city = {p:i for (i,p) in enumerate(data['city'].unique())}
store_state = {p:i for (i,p) in enumerate(data['state'].unique())}

before_holiday_week = {p:i for (i,p) in enumerate(data['before_holiday_week'].unique())}
is_after_holiday_week = {p:i for (i,p) in enumerate(data['is_after_holiday_week'].unique())}
is_holiday_day = {p:i for (i,p) in enumerate(data['is_holiday_day'].unique())}
holiday_type = {p:i for (i,p) in enumerate(data['holiday_type'].unique())}
holiday_locale = {p:i for (i,p) in enumerate(data['holiday_locale'].unique())}
holiday_locale_name = {p:i for (i,p) in enumerate(data['holiday_locale_name'].unique())}

payment_week = {p:i for (i,p) in enumerate(data['payment_week'].unique())}

month = {p:i for (i,p) in enumerate(data['month'].unique())}
year = {p:i for (i,p) in enumerate(data['year'].unique())}
weekday = {p:i for (i,p) in enumerate(data['weekday'].unique())}
day = {p:i for (i,p) in enumerate(data['day'].unique())}

processed_data = data.loc[:,['item_nbr','onpromotion','item_family','item_class','perishable',
                             'store_nbr','cluster','type','city','state',
                             'before_holiday_week','is_after_holiday_week','is_holiday_day','holiday_type',
                             'holiday_locale','holiday_locale_name','payment_week',
                             'month','year','weekday','day']].copy()

# items categorical featuers
processed_data['item_nbr'] = [item_nbr[x] for x in data['item_nbr']]
processed_data['perishable'] = [perishable[x] for x in data['perishable']]
processed_data['item_family'] = [item_family[x] for x in data['item_family']]
processed_data['item_class'] = [item_class[x] for x in data['item_class']]
processed_data['onpromotion'] = [onpromotion[x] for x in data['onpromotion']]

# store categorical featuers
processed_data['store_nbr'] = [store_nbr[x] for x in data['store_nbr']]
processed_data['cluster'] = [store_cluster[x] for x in data['cluster']]
processed_data['type'] = [store_type[x] for x in data['type']]
processed_data['city'] = [store_city[x] for x in data['city']]
processed_data['state'] = [store_state[x] for x in data['state']]

processed_data['before_holiday_week'] = [before_holiday_week[x] for x in data['before_holiday_week']]
processed_data['is_after_holiday_week'] = [is_after_holiday_week[x] for x in data['is_after_holiday_week']]
processed_data['is_holiday_day'] = [is_holiday_day[x] for x in data['is_holiday_day']]
processed_data['holiday_type'] = [holiday_type[x] for x in data['holiday_type']]
processed_data['holiday_locale'] = [holiday_locale[x] for x in data['holiday_locale']]
processed_data['holiday_locale_name'] = [holiday_locale_name[x] for x in data['holiday_locale_name']]

processed_data['payment_week'] = [payment_week[x] for x in data['payment_week']]

processed_data['month'] = [month[x] for x in data['month']]
processed_data['year'] = [year[x] for x in data['year']]
processed_data['weekday'] = [weekday[x] for x in data['weekday']]
processed_data['day'] = [day[x] for x in data['day']]


print(processed_data.info())

target = data['unit_sales'].values


item_inp = Input(shape=(1,),dtype='int64')
item_family_inp = Input(shape=(1,),dtype='int16')
item_class_inp = Input(shape=(1,),dtype='int16')
perishable_inp = Input(shape=(1,),dtype='bool')
onpromotion_inp = Input(shape=(1,),dtype='bool')

store_inp = Input(shape=(1,),dtype='int16')
store_cluster_inp = Input(shape=(1,),dtype='int16')
store_type_inp = Input(shape=(1,),dtype='int16')
store_city_inp = Input(shape=(1,),dtype='int64')
store_state_inp = Input(shape=(1,),dtype='int64')

before_holiday_week_inp = Input(shape=(1,),dtype='bool')
is_after_holiday_week_inp = Input(shape=(1,),dtype='bool')
is_holiday_day_inp = Input(shape=(1,),dtype='bool')

holiday_type_inp = Input(shape=(1,),dtype='int64')
holiday_locale_inp = Input(shape=(1,),dtype='int64')
holiday_locale_name_inp = Input(shape=(1,),dtype='int64')

payment_week_inp = Input(shape=(1,),dtype='bool')

month_inp = Input(shape=(1,),dtype='int64')
year_inp = Input(shape=(1,),dtype='int64')
weekday_inp = Input(shape=(1,),dtype='int64')
day_inp = Input(shape=(1,),dtype='int64')


item_size=int(np.log2(len(item_nbr)))
store_size=int(np.log2(len(store_nbr)))

item_emb = Embedding(len(item_nbr),item_size,input_length=1, embeddings_regularizer=l2(1e-5))(item_inp)
item_family_emb = Embedding(len(item_family),5,input_length=1, embeddings_regularizer=l2(1e-5))(item_family_inp)
item_class_emb = Embedding(len(item_class),5,input_length=1, embeddings_regularizer=l2(1e-5))(item_class_inp)
item_onpromotion_emb = Embedding(len(onpromotion),1,input_length=1, embeddings_regularizer=l2(1e-5))(onpromotion_inp)
item_perishable_emb = Embedding(len(perishable),1,input_length=1, embeddings_regularizer=l2(1e-5))(perishable_inp)

store_emb = Embedding(len(store_nbr),store_size,input_length=1, embeddings_regularizer=l2(1e-5))(store_inp)
store_cluster_emb = Embedding(len(store_cluster),5,input_length=1, embeddings_regularizer=l2(1e-5))(store_cluster_inp)
store_type_emb = Embedding(len(store_type),5,input_length=1, embeddings_regularizer=l2(1e-5))(store_type_inp)
store_city_emb = Embedding(len(store_city),5,input_length=1, embeddings_regularizer=l2(1e-5))(store_city_inp)
store_state_emb = Embedding(len(store_state),5,input_length=1, embeddings_regularizer=l2(1e-5))(store_state_inp)

before_holiday_week_emb = Embedding(len(before_holiday_week),5,input_length=1, embeddings_regularizer=l2(1e-5))(before_holiday_week_inp)
is_after_holiday_week_emb = Embedding(len(is_after_holiday_week),5,input_length=1, embeddings_regularizer=l2(1e-5))(is_after_holiday_week_inp)
is_holiday_day_emb = Embedding(len(is_holiday_day),5,input_length=1, embeddings_regularizer=l2(1e-5))(is_holiday_day_inp)
holiday_type_emb = Embedding(len(holiday_type),5,input_length=1, embeddings_regularizer=l2(1e-5))(holiday_type_inp)
holiday_locale_emb = Embedding(len(holiday_locale),5,input_length=1, embeddings_regularizer=l2(1e-5))(holiday_locale_inp)
holiday_locale_name_emb = Embedding(len(holiday_locale_name),5,input_length=1, embeddings_regularizer=l2(1e-5))(holiday_locale_name_inp)

payment_week_emb = Embedding(len(payment_week),5,input_length=1, embeddings_regularizer=l2(1e-5))(payment_week_inp)

month_emb = Embedding(len(month),5,input_length=1, embeddings_regularizer=l2(1e-5))(month_inp)
year_emb = Embedding(len(year),5,input_length=1, embeddings_regularizer=l2(1e-5))(year_inp)
weekday_emb = Embedding(len(weekday),5,input_length=1, embeddings_regularizer=l2(1e-5))(weekday_inp)
day_emb = Embedding(len(day),5,input_length=1, embeddings_regularizer=l2(1e-5))(day_inp)

x = concatenate([item_emb,item_family_emb,item_class_emb,item_onpromotion_emb,item_perishable_emb,store_emb,store_cluster_emb,
                 store_type_emb,store_city_emb,store_state_emb,before_holiday_week_emb,is_after_holiday_week_emb,
                 is_holiday_day_emb,holiday_type_emb,holiday_locale_emb,holiday_locale_name_emb,payment_week_emb,
                 month_emb,year_emb,weekday_emb,day_emb])
x = Flatten()(x)
x = BatchNormalization()(x)
x = Dense(32,activation='relu')(x)
x = Dense(16,activation='relu')(x)
x = Dropout(0.4)(x)
x = BatchNormalization()(x)
x = Dense(16,activation='relu')(x)
x = Dense(8,activation='relu')(x)
x = Dropout(0.4)(x)
x = Dense(1,activation='relu')(x)

nn_model = Model([item_inp,item_family_inp,item_class_inp,onpromotion_inp,perishable_inp,store_inp,store_cluster_inp,
                  store_type_inp,store_city_inp,store_state_inp,before_holiday_week_inp,is_after_holiday_week_inp,
                  is_holiday_day_inp,holiday_type_inp,holiday_locale_inp,holiday_locale_name_inp,payment_week_inp,
                  month_inp,year_inp,weekday_inp,day_inp],x)
nn_model.compile(loss = 'mse',optimizer='adam')
nn_model.summary()

def set_callbacks(description='run1',patience=15):
    cp = ModelCheckpoint('best_model_weights_{}.h5'.format(description),save_best_only=True)
    es = EarlyStopping(patience=patience,monitor='val_loss')
    rlop = ReduceLROnPlateau(patience=5)
    cb = [cp,es,rlop]
    return cb


def NWRMSLE(y, pred, w):
    return mean_squared_error(y, pred, sample_weight=w)**0.5

# for column in numeric_data:
#     scaler = MinMaxScaler()
#     numeric_data[column]=scaler.fit_transform(numeric_data[column])
# merged=[data,numeric_data]
# merged_data=pd.concat(merged,axis=1)
# X_train, X_test, y_train, y_test = train_test_split(merged,target,test_size=0.25,random_state=42,shuffle=True)
# features = merged_data.columns
# nn_model.fit([X_train[f] for f in features],y_train,epochs=10,batch_size=32,
#             validation_data=([X_test[f] for f in features],y_test))


weights=processed_data['perishable'].apply(lambda x: 1.25 if x==1 else 1)
class_weights_dic=weights.to_dict()
X_train, X_test, y_train, y_test = train_test_split(processed_data,target,test_size=0.25,random_state=42,shuffle=True)
W_test = X_test['perishable'].map({0:1.0, 1:1.25})
features = processed_data.columns

nn_model.fit([X_train[f] for f in features],y_train,epochs=10,batch_size=32,
             validation_data=([X_test[f] for f in features],y_test),
             class_weight=class_weights_dic, callbacks=set_callbacks())

preds = nn_model.predict([X_test[f] for f in features])

print('NWRMSLE RF',NWRMSLE((y_test),(preds),W_test.values))
rmse_rfr_val = np.sqrt(mean_squared_error(y_test, preds))
print('Val RMSE - %.3f' % rmse_rfr_val)

