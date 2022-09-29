# Time Series Analysis and ForecastWith Deep Learning models.

A time series is a sequence of data points that occur in successive order over some period of time. It allows one to see what factors influence certain variables from period to period. It is used to predict future values based on previously observed values.

Time Series Forecasting is an important technique in machine learning, applicable to several domains including medicine, weather forecasting, biology, supply chain management and stock prices forecasting, etc.

As different time series problems are studied in many different fields, many new architectures have been developed in recent years. This has also been simplified by the growing availability of open-source frameworks, which make the development of new custom network components easier and faster.

Why Deep Learning? Deep learning neural networks are able to automatically learn arbitrary complex mappings from inputs to outputs and support multiple inputs and outputs.

Artificial neural networks (ANNs), usually simply called neural networks (NNs), are computing systems inspired by the Biological Neural Networks (BNNs) that constitute animal brains.

![Neuron3](https://user-images.githubusercontent.com/88746246/192987579-4b40b7e4-9f17-4911-a13e-e52e40b2a428.png)

An ANN is based on a collection of connected units or nodes called Artificial Neurons, which loosely model the neurons in a biological brain. Each connection can transmit a signal to other neurons.

The connections are called edges. Neurons and edges typically have a weight that adjusts as learning proceeds. The weight increases or decreases the strength of the signal at a connection.

Typically, neurons are aggregated into layers. Different layers may perform different transformations on their inputs. Signals travel from the first layer (the input layer), to the last layer (the output layer), possibly after traversing the layers multiple times.

## Our Datasets
The dataset used in this project was gotten from a competition on kaggle (The M5 Competition), which ran from 2 March to 30 June 2020 and the script notebook is on my Github.

I used these datasets to predict item sales at stores in various locations for a 28-day time period.

Our datasets consist of sales data, generously made available by Walmart, starting at the item level and aggregating to that of departments, product categories and stores in three geographical areas of the US: California, Texas, and Wisconsin with 10 stores in total: 4 in Califonia and 3 each in Texas and Winsonsin.

Our first dataset (Train sales) consists of the number of units sold at day i, starting from 2011–01–29 to 2016–06–19 in the 10 stores.

<img width="886" alt="Screenshot 2022-09-22 at 09 24 14" src="https://user-images.githubusercontent.com/88746246/192988057-0fbf355e-2d9d-4078-aef6-271f8336ace9.png">

The goods sold are divided into 3 categories which are: Hobbies, Household and Foods. These are then further divided into items which are Hobbies_1, Hobbies_2, Household_1, Household_2, Foods_1, Foods_2, Foods_3.

Our second dataset (Calendar) contains information about the dates the products are sold.

<img width="799" alt="Screenshot 2022-09-22 at 09 42 15" src="https://user-images.githubusercontent.com/88746246/192988228-17cd0d60-b87c-4087-98a8-f8fad22ea7b6.png">

Snap_CA, snap_TX, and snap_WI are binary variables (0 or 1) indicating whether the stores of CA, TX or WI allow SNAP purchases on the examined date.
1 indicates that SNAP purchases are allowed.

The United States federal government provides a nutrition assistance benefit called the Supplement Nutrition Assistance Program (SNAP).
SNAP provides low income families and individuals with an Electronic Benefits Transfer debit card to purchase food products.

Our third dataset (sell prices) contains information about the price of the products sold per store and date. It is the price of the product for the given week/store. The price is provided per week (average across seven days).
If not available, this means that the product was not sold during the examined week.

## Difficulties Encountered
Firstly, the datasets are really large and so merging them made my work station crash a lot. I then moved to Goggle Colab. It worked for a while and then it started crashing again.

I finally moved to Amazon Sagemaker and it was a lifesaver.

## Exploratory Data Analysis
Here is the datasets time series plot:

<img width="407" alt="Screenshot 2022-09-22 at 09 14 06" src="https://user-images.githubusercontent.com/88746246/192988571-5e54569a-41d8-4a4c-adc7-4cf87b28aa60.png">

The obvious drop is due to the Christmas effect. There is a drop in sales on Christmas day.

<img width="415" alt="Screenshot 2022-09-22 at 09 35 47" src="https://user-images.githubusercontent.com/88746246/192988838-0b8e8f66-db29-4e12-ac1a-19c0c6ab0e2d.png">


This shows the different time series distribution for the different 3 states. Clearly there is an upward trend over the years with California having more sales. This is due to the fact that California has 4 stores and the 2 other states have 3 each.

<img width="427" alt="Screenshot 2022-09-22 at 09 35 58" src="https://user-images.githubusercontent.com/88746246/192989039-5c6a7da5-a3e2-4f59-9762-a3a9e57491f4.png">

This shows the different time series for the different categories. Food takes is sold most, followed by household and then hobbies.

<img width="767" alt="Screenshot 2022-09-22 at 09 36 10" src="https://user-images.githubusercontent.com/88746246/192989155-056b9162-7331-4417-b326-60645406d9cf.png">

This is further broken down to show the different stores in the different states.

## For Seasonalities,

<img width="307" alt="Screenshot 2022-09-22 at 09 36 34" src="https://user-images.githubusercontent.com/88746246/192989321-a462fe0c-bb86-432e-a854-aad254dc1748.png">


there is usually a drop in sales at midweek, it picks up by Friday, peaks on Sunday (except for Wisconsin) and then drops. Obviously, stores are open on Sundays, unlike here in Germany where stores are mostly shut.

<img width="436" alt="Screenshot 2022-09-22 at 09 36 44" src="https://user-images.githubusercontent.com/88746246/192989596-fa449c52-f9dd-407e-b279-fcd66d4b8ad8.png">

Also, for the monthly seasonality, we see a drop in sales by May, but then it picks up which could be due to the Summer season. It peaks in August then drops. Picks up by November due to the holiday seasons and then drops at Christmas, just as the year is coming to an end.

## Data Modelling Approach
After merging the datasets and dropping some columns to avoid repetitions and to also save space, I grouped this new large dataset that I now have so that each column is a representation of each category in each store. With this, I can then use one column's data to fit my model and then predict the next 28 days.

<img width="933" alt="Screenshot 2022-09-22 at 10 21 51" src="https://user-images.githubusercontent.com/88746246/192989841-984155a4-2c00-4057-bcb4-3481cd121a90.png">

Models
LSTM: Long Short Term Memory networks “LSTMs” — are a special kind of RNN (Recurrent Neural Networks), capable of learning long-term dependencies. LSTMs are mainly designed to avoid the long-term dependency problem (remembering information for long periods of time) which RNNs have.

A LSTM unit is composed of: a cell state C(t) , that brings information along the entire sequence and represents the memory of the network; a forget gate, that decides what is relevant to keep from previous time steps; an input gate, that decides what information is relevant to add from the current time step; an output gate, that decides the value of the output at current time step.

Neural Prophet: NeuralProphet is a python library for modelling time-series data based on neural networks. It’s built on top of PyTorch and is heavily inspired by Facebook Prophet and AR-Net libraries.

Using PyTorch’s Gradient Descent optimization engine making the modeling process much faster than Prophet Using AR-Net for modeling time-series autocorrelation (aka serial correlation) Custom losses and metrics.

Facebook Prophet: Facebook Prophet follows the sklearn model API. It create an instance of the Prophet class and then call its fit and predict methods.

The input to Prophet is always a dataframe with two columns: ds and y. The ds (datestamp) column should be of a format expected by Pandas, ideally YYYY-MM-DD for a date or YYYY-MM-DD HH:MM:SS for a timestamp. The y column must be numeric, and represents the measurement we wish to forecast.



I compared my results to one of a former colleague who used the ARIMA and SARIMA models for the same forecast.

SARIMA: Seasonal Autoregressive Integrated Moving Average, or Seasonal ARIMA, is an extension of ARIMA that explicitly supports univariate time series data with a seasonal component.

It has three new hyperparameters to specify the autoregression (AR), differencing (I) and moving average (MA) for the seasonal component of the series, as well as an additional parameter for the period of the seasonality.

## Models Comparison
After running the previously defined models and obtaining predictions, I plotted it against the actual values which was provided after the competition for evaluation.



<img width="884" alt="Screenshot 2022-09-22 at 11 13 28" src="https://user-images.githubusercontent.com/88746246/192990453-3559ff35-63ed-4941-a6d0-b9a6bd5bad92.png">





### Points to note: Facebook Prophet and Neural Prophet when compare to our actual results were close. However, looking at the LSTM model, the forecast is 1 step ahead of the actual predictions which does not look good.

### Next points of action

-To make the LSTM model better, things I could do:

-Increase the hidden layers in the LSTM node,

-Add more layers of the LSTM,

-Hyperparameters tuning.

-The scripts can be found on my GitHub.

Thanks for reading.














