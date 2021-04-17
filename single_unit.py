# Library edded
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import tensorflow as tf

# reading data 
df = pd.read_csv("Jan_2019_ontime.csv")
df.head()

#  DEP_TIME_BLK column using for take day time for that reason I am going to split dataset "-" and selecting first value  
df["Dep1"] = df["DEP_TIME_BLK"].apply(lambda x:x.split("-")[0])

# after selecting - For exam. "06:00-07:00"  - fisrt value of 06:00 making than I am going to select first 2 value 06
df['DEP'] = df["Dep1"].apply(lambda z: z[:2])

# in that section I am goint to use drop all coumns and just keeping DEP_DEL15 and date  
df1 = df.drop(['DAY_OF_WEEK','OP_UNIQUE_CARRIER','OP_CARRIER_AIRLINE_ID','OP_CARRIER','TAIL_NUM','OP_CARRIER_FL_NUM','ORIGIN_AIRPORT_ID','ORIGIN_AIRPORT_SEQ_ID','ORIGIN','DEST_AIRPORT_ID','DEST_AIRPORT_SEQ_ID','DEST','DEP_TIME','DEP_TIME_BLK','ARR_TIME','ARR_DEL15','CANCELLED','DIVERTED', 'DISTANCE', 'Unnamed: 21', 'Dep1'],axis=1)
df1 = df1.astype({'DEP': 'int64'}) # here im change type of data format to int64 format for next section

hours = [7,15,23] # here is day time of days im going to sum delyas between 00:00-07:00,07:00-15:00,15:00-23:00

val = [] # sum of value for each part of days

days = []

for day in range(1,31):
    test = df1[df1["DAY_OF_MONTH"]==day] 
    for hour in hours:
        test = test.astype({'DEP': 'int64'})
        test1 = test[test["DEP"]<=hour]
        val.append(test1['DEP_DEL15'].sum())
        day_all = "2019-01-"+str(day)+" " + str(hour) + ":" + "00"
        days.append(day_all)

        
dic_df = {'Delay': val,"Data":days} # creating dict with new params

df2 = pd.DataFrame(data=dic_df) # make DataFrame

df2["Data"] = pd.to_datetime(df2["Data"] ,infer_datetime_format=True)

df2 = df2.set_index(["Data"])

scaler = MinMaxScaler() # our dataset containing value more than 1 so for making model as well params will be sclae between 0-1
dataset = scaler.fit_transform(df2[['Delay']])



n = len(dataset)

train = dataset[0:int(n*0.9)] #90% of data for training  
test = dataset[int(n*0.9):] # 10% data for testing section

# we will use it for making new dataset after traing 
data_time = df2.index

# same like pervious
train_data = data_time[0:int(n*0.9)] #90% of data for training 
test_data = data_time[int(n*0.9):] # 10% data for testing section

# here is my data set making function 
def to_sequences(dataset, seq_size=1):
    x = []
    y = []

    for i in range(len(dataset)-seq_size-1):
        window = dataset[i:(i+seq_size), 0]
        x.append(window)
        y.append(dataset[i+seq_size, 0])
        
    return np.array(x),np.array(y)


train_x , train_y = to_sequences(train,seq_size=5) # it mean that after 5 parans im goint to make prediction

test_x , test_y = to_sequences(test,seq_size=5)

trainX = np.reshape(train_x, (train_x.shape[0], 1, train_x.shape[1])) # make dataset for ready my input value
testX = np.reshape(test_x, (test_x.shape[0], 1, test_x.shape[1]))


model = tf.keras.models.Sequential()
model.add(tf.keras.layers.LSTM(64,input_shape=(None,5)))
model.add(tf.keras.layers.Dense(32))
model.add(tf.keras.layers.Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')
model.summary()

model.fit(trainX,train_y,validation_data=(testX,test_y),epochs=100)

# testing section
testPredict = model.predict(testX) 
testPredict = scaler.inverse_transform(testPredict) # we are invers our datset as pervious one
testY = scaler.inverse_transform([test_y]) # our pred data aso same 
# training section 

trainPredict = model.predict(trainX)
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([train_y])

#  train_data , test_data


train_data1 = train_data[6:] # we did make pred for 5 sep of dataset so for that reason im gpint to drop first 5 params 
test_data1 = test_data[:3]

train_df = pd.DataFrame(trainY.T) 
test_df = pd.DataFrame(testY.T)


test_df_pred = pd.DataFrame(testPredict) # making dataset with predicted values
train_data_pred = pd.DataFrame(trainPredict) 

train_df = train_df.set_index(train_data1)

test_df = test_df.set_index(test_data1)

test_df_pred = test_df_pred.set_index(test_data1)
train_data_pred = train_data_pred.set_index(train_data1)

plt.plot(train_df,label="traing")
plt.plot(train_data_pred, label="training pred")
plt.plot(test_df, label="test")
plt.plot(test_df_pred, label="test pred")
plt.legend()
##################  Here I am going to do pred for 2020 delays dataset and here i will do alsmost same like 2019 
df_test = pd.read_csv("Jan_2020_ontime.csv")

df_test["Dep1"] = df_test["DEP_TIME_BLK"].apply(lambda x:x.split("-")[0])
df_test['DEP'] = df_test["Dep1"].apply(lambda z: z[:2])


df1_test = df_test.drop(['DAY_OF_WEEK','OP_UNIQUE_CARRIER','OP_CARRIER_AIRLINE_ID','OP_CARRIER','TAIL_NUM','OP_CARRIER_FL_NUM','ORIGIN_AIRPORT_ID','ORIGIN_AIRPORT_SEQ_ID','ORIGIN','DEST_AIRPORT_ID','DEST_AIRPORT_SEQ_ID','DEST','DEP_TIME','DEP_TIME_BLK','ARR_TIME','ARR_DEL15','CANCELLED','DIVERTED', 'DISTANCE', 'Unnamed: 21', 'Dep1'],axis=1)
df1_test = df1_test.astype({'DEP': 'int64'})

hours_test = [7,15,23]

val_test = []

days_test = []

for day_test in range(1,31):
    test_test = df1_test[df1_test["DAY_OF_MONTH"]==day_test]
    for hour_test in hours_test:
        test_test = test_test.astype({'DEP': 'int64'})
        test1_test = test_test[test_test["DEP"]<=hour_test]
        val_test.append(test1_test['DEP_DEL15'].sum())
        day_all_test = "2020-01-"+str(day)+" " + str(hour) + ":" + "00"
        days_test.append(day_all_test)

        
dic_df_test = {'Delay': val_test,"Data":days_test} 

df2_test = pd.DataFrame(data=dic_df_test)

df2_test["Data"] = pd.to_datetime(df2_test["Data"] ,infer_datetime_format=True)

df2_test = df2_test.set_index(["Data"])

scaler_test = MinMaxScaler()
dataset_test = scaler_test.fit_transform(df2_test[['Delay']])





data_time_test = df2_test.index




def to_sequences(dataset, seq_size=1):
    x = []
    y = []

    for i in range(len(dataset)-seq_size-1):
        #print(i)
        window = dataset[i:(i+seq_size), 0]
        x.append(window)
        y.append(dataset[i+seq_size, 0])
        
    return np.array(x),np.array(y)


train_new_x , train_new_y = to_sequences(dataset_test,seq_size=5)



trainX_new = np.reshape(train_new_x, (train_new_x.shape[0], 1, train_new_x.shape[1]))

##### new dataset prediction 



testPredict_new = model.predict(trainX_new)
testPredict_new = scaler_test.inverse_transform(testPredict_new)
train_new_y = scaler.inverse_transform([train_new_y])
# training section 



#train_new_x1 = train_new_x[6:]


train_df_new = pd.DataFrame(train_new_y.T)



test_df_pred_new = pd.DataFrame(testPredict_new)

data_time_test_1 = data_time_test[6:]

#test_df_pred_new = test_df_pred_new.set_index(data_time_test_1)

#train_df_new = train_df_new.set_index(data_time_test_1)





plt.plot(train_df_new,label="2020 train")
plt.plot(test_df_pred_new,label="2020 predictions")
plt.legend()