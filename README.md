# time_series_delays

[This model](https://github.com/tural327/time_series_delays/blob/master/single_unit.py) was bulit only using delays during period
* After reading Jan_2019_ontime file I changed couple of things
* DEP_TIME_BLK column using for take day time for that reason I am going to split dataset "-" and selecting first value  For exm. 06:00-07:00 i took only 06:00
* Then all parametrs was droped expext delyas 
* I divide days 3 parts im going to sum delyas between 00:00-07:00,07:00-15:00,15:00-23:00
* By using for loop delyas was sum() for each part of days
* Parametrs was scaled between 0-1 
* Using def to_sequences for making dataset train in LSTM 
* While building netwqork I used single LSTM with 64 unit than full connected layers added (next part I will use more lstms for more params)
* After traing by using model predict function I got predicted values than inverse_transformed for getting not sclaed values
* For testing my model I am going to use Jan_2020_ontime file
* I did same things for Jan_2020_ontime also like Jan_2019_ontime
Here is my result
![](https://github.com/tural327/time_series_delays/blob/master/result.png)
