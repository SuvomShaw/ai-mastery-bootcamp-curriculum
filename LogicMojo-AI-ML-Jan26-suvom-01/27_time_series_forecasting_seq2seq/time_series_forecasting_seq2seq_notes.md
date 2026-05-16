# Time Series Forecasting with Seq2Seq Models 

## 1. Core Idea

Time series forecasting means using past ordered values to predict future ordered values.

Simple idea:

```text
past values  ->  forecasting model  ->  future values
```

Example:

```text
last 30 days of sales  ->  model  ->  next 7 days of sales
```

The most important rule:

```text
In time series data, order matters.
```

If the rows are shuffled, the model may lose the relationship between past and future.

---

## 2. What Is Time Series Data?

A time series is a sequence of observations arranged in chronological order.

Mathematical form:

```text
y1, y2, y3, ..., yT
```

Where:

- `yt` is the value at step `t`
- `T` is the total number of observations

Example:

```text
Day:    1    2    3    4    5    6    7
Sales:  50   55   52   60   65   63   70
        --------------------------------->
              order moves forward
```

Forecasting question:

```text
Given what happened earlier, what is likely to happen next?
```

---

## 3. Why Order Matters

Look at this sequence:

```text
10, 12, 15, 18, 22
```

This shows an increasing pattern.

Now shuffle it:

```text
18, 10, 22, 12, 15
```

The increasing pattern becomes harder to see.

In normal tabular machine learning, rows are often independent.

In forecasting, earlier values provide context for later values.

Key point:

```text
Past values are not just rows.
Past values are context.
```

---

## 4. Real-Life Forecasting Examples

| Domain | Data | Forecast |
|---|---|---|
| Retail | daily product sales | next 7 days of sales |
| Electricity | hourly demand | next 24 hours of demand |
| Website | hourly visitors | next few hours of traffic |
| Hospital | daily bed usage | next week of bed demand |
| Cloud systems | CPU usage | next 15 minutes of load |
| Finance | daily volatility | next few days of risk |
| Logistics | parcel volume | next week of deliveries |
| Weather | temperature and rainfall | next few days |

Forecasting is useful because future estimates support decisions:

- inventory planning
- staffing
- energy generation
- delivery planning
- autoscaling servers
- risk management

---

## 5. Components of a Time Series

Many series can be understood using four parts:

```text
Observed series = level + trend + seasonality + noise
```

| Component | Meaning | Example |
|---|---|---|
| Level | normal average value | sales are usually around 500 units |
| Trend | long-term direction | sales are increasing month by month |
| Seasonality | repeated pattern | sales rise every weekend |
| Noise | random movement | one unusual spike or drop |

Diagram:

```text
Observed series
       |
       +--> Level
       |    normal average value
       |
       +--> Trend
       |    long-term upward or downward movement
       |
       +--> Seasonality
       |    repeated pattern
       |
       +--> Noise
            random variation
```

Visual idea:

```text
value
  ^
  |                         *       *
  |                    *       *        *
  |               *        *
  |          *        *
  |     *        *
  +------------------------------------------> order
       level + trend + seasonality + noise
```

---

## 6. Important Forecasting Vocabulary

### Target

The variable we want to predict.

Example:

```text
target = daily sales
```

### Frequency

How often values are recorded.

Examples:

```text
hourly, daily, weekly, monthly
```

### Lookback Window

The amount of past history given to the model.

Example:

```text
Use previous 30 days as input.
lookback = 30
```

### Forecast Horizon

How far ahead we predict.

Example:

```text
Predict next 7 days.
horizon = 7
```

### One-Step Forecasting

Predict only the next value.

```text
past 30 days -> next 1 day
```

### Multi-Step Forecasting

Predict many future values.

```text
past 30 days -> next 7 days
```

### Univariate Forecasting

Use one variable.

```text
past sales -> future sales
```

### Multivariate Forecasting

Use many variables.

```text
sales + price + discount + weekday -> future sales
```

### Exogenous Variables

Extra variables that help predict the target.

Examples:

- price
- discount
- holiday flag
- weather
- marketing spend
- weekday

Important:

```text
Only use future features if they are known when making the forecast.
```

Valid future-known feature:

```text
Tomorrow is Sunday.
```

Invalid future feature:

```text
Tomorrow's actual sales.
```

---

## 7. Forecasting Problem Setup

Before modeling, define:

1. Target variable
2. Data frequency
3. Lookback length
4. Forecast horizon
5. Available features
6. Evaluation metric

Example:

```text
Goal: predict next 7 days of store sales
Target: sales
Frequency: daily
Lookback: previous 30 days
Horizon: next 7 days
Features: sales, discount, holiday, weekday
```

Mathematical form:

```text
input:  [y(t-L+1), ..., y(t)]
output: [y(t+1),   ..., y(t+H)]
```

Where:

- `L` is lookback length
- `H` is forecast horizon

Diagram:

```text
Past                                                   Future
|-------------------- lookback L --------------------|--- horizon H ---|
day t-29                                             day t             t+1 ... t+7

Input to model: previous 30 days
Model output:   next 7 days
```

---

## 8. Sliding Window Transformation

A raw series is one long sequence.

Machine learning needs many input-output examples.

Sliding windows convert a sequence into supervised learning samples.

Example series:

```text
[10, 12, 15, 18, 20, 23, 25, 28]
```

Use:

```text
lookback = 3
horizon  = 2
```

Training samples:

```text
X1 = [10, 12, 15] -> y1 = [18, 20]
X2 = [12, 15, 18] -> y2 = [20, 23]
X3 = [15, 18, 20] -> y3 = [23, 25]
X4 = [18, 20, 23] -> y4 = [25, 28]
```

Diagram:

```text
Series:
10   12   15   18   20   23   25   28

[--------- X1 --------][--- y1 ---]
     [--------- X2 --------][--- y2 ---]
          [--------- X3 --------][--- y3 ---]
               [--------- X4 --------][--- y4 ---]
```

Key point:

```text
Sliding windows convert forecasting into supervised learning.
```

---

## 9. Tensor Shapes

For deep learning, inputs are usually arranged like this:

```text
X shape = (batch_size, lookback, num_features)
y shape = (batch_size, horizon, target_features)
```

Example:

```text
batch_size      = 32
lookback        = 30
num_features    = 4
horizon         = 7
target_features = 1
```

Then:

```text
X = (32, 30, 4)
y = (32, 7, 1)
```

Meaning:

```text
32 -> examples in one batch
30 -> each example has 30 past steps
4  -> each step has 4 features
7  -> model predicts 7 future steps
1  -> one target variable
```

---

## 10. Train, Validation, and Test Split

Time series data should be split by order.

Correct:

```text
Past                                                   Future
|---------------- train ----------------|----- val -----|----- test -----|
```

Wrong:

```text
random rows from the full series mixed into train and test
```

Why random split is wrong:

```text
It can allow future information to leak into training.
```

Example:

```text
Correct:
Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec
|---------- train ----------|--- val ---|-- test --|

Wrong:
train rows randomly selected from Jan to Dec
test rows randomly selected from Jan to Dec
```

The correct evaluation question is:

```text
Can the past predict the future?
```

---

## 11. Scaling Rule

Neural networks usually train better when features are scaled.

Common scaling:

```text
scaled_value = (value - mean) / standard_deviation
```

Important rule:

```text
Fit the scaler on training data only.
```

Correct:

```text
fit scaler on train
transform train
transform validation
transform test
```

Wrong:

```text
fit scaler on full dataset
```

Why wrong?

```text
The scaler has seen future values from validation and test periods.
```

This is data leakage.

---

## 12. Forecasting Baselines

Before using complex models, create simple baselines.

Why?

```text
A complex model is only useful if it beats simple reasonable methods.
```

Common baselines:

| Baseline | Rule | Example |
|---|---|---|
| Naive | repeat last value | tomorrow = today |
| Seasonal naive | repeat previous season | next Monday = last Monday |
| Moving average | repeat recent average | tomorrow = average of last 7 days |

Naive example:

```text
history    = [20, 22, 25]
prediction = [25]
```

Seasonal naive example:

```text
If weekly seasonality exists:
next Monday forecast = last Monday value
```

Moving average example:

```text
history = [20, 22, 25, 24, 26, 28, 27]
forecast = average(history)
```

Important:

```text
If a deep learning model cannot beat seasonal naive, it is not useful yet.
```

---

## 13. Forecasting Metrics

### MAE

Mean Absolute Error:

```text
MAE = average(|actual - predicted|)
```

Easy to understand because it uses the same unit as the target.

Example:

```text
MAE = 8 sales units
```

### MSE

Mean Squared Error:

```text
MSE = average((actual - predicted)^2)
```

Large errors are punished more strongly.

### RMSE

Root Mean Squared Error:

```text
RMSE = sqrt(MSE)
```

Same unit as the target.

### MAPE

Mean Absolute Percentage Error:

```text
MAPE = average(|actual - predicted| / |actual|) * 100
```

Useful when percentage error is needed.

Problem:

```text
MAPE fails when actual value is 0 or very close to 0.
```

Example:

```text
actual = 0
predicted = 5
MAPE is undefined because division by zero happens.
```

### Horizon-Wise Error

For multi-step forecasting, check error at each future step.

Example:

| Step | t+1 | t+2 | t+3 | t+4 | t+5 | t+6 | t+7 |
|---|---:|---:|---:|---:|---:|---:|---:|
| MAE | 3.1 | 3.8 | 4.5 | 5.2 | 6.0 | 7.1 | 8.9 |

Diagram:

```text
MAE
^
9 |                              *
8 |                         *
7 |                    *
6 |               *
5 |          *
4 |     *
3 | *
  +----+----+----+----+----+----+----> horizon step
    1    2    3    4    5    6    7
```

Key point:

```text
The model may be accurate for t+1 but weak for t+7.
```

---

## 14. Multi-Step Forecasting Strategies

### Recursive Strategy

Train one model to predict one step ahead, then feed predictions back into the model.

```text
predict t+1
use predicted t+1 to predict t+2
use predicted t+2 to predict t+3
```

Advantage:

```text
Simple.
```

Limitation:

```text
Errors can accumulate.
```

### Direct Strategy

Train separate models for each future step.

```text
model_1 predicts t+1
model_2 predicts t+2
model_3 predicts t+3
```

Advantage:

```text
Each model specializes in one horizon step.
```

Limitation:

```text
Many models are needed.
```

### Multi-Output Strategy

Train one model to output the entire horizon at once.

```text
past window -> [t+1, t+2, ..., t+H]
```

Advantage:

```text
Simple prediction call.
```

Limitation:

```text
May not model step-by-step dependency deeply.
```

### Seq2Seq Strategy

Use an encoder to read the past and a decoder to generate the future.

```text
past sequence -> encoder -> decoder -> future sequence
```

Advantage:

```text
Natural fit when input and output are both sequences.
```

Limitation:

```text
Training and inference behave differently, so implementation must be careful.
```

---

## 15. Choosing a Forecasting Technique

Do not always start with deep learning.

Good approach:

```text
start simple -> compare honestly -> add complexity only if needed
```

Model complexity ladder:

```text
More data, more features, more complexity
^
|
|   Deep learning
|   Seq2Seq, LSTM, GRU, TCN, Transformers
|
|   Machine learning with lag features
|   Random Forest, XGBoost, LightGBM, CatBoost
|
|   Classical seasonal models
|   SARIMA, SARIMAX, ETS
|
|   Classical non-seasonal models
|   ARIMA, exponential smoothing
|
|   Baselines
|   naive, seasonal naive, moving average
+-------------------------------------------------> model complexity
```

### ARIMA

ARIMA stands for:

```text
AR = AutoRegressive
I  = Integrated or differencing
MA = Moving Average error
```

Use ARIMA when:

- There is one main target series
- The dataset is small or medium
- Trend exists but seasonality is not strong
- A statistical baseline is needed
- Interpretability matters

Avoid ARIMA when:

- Strong seasonality exists
- Many features are needed
- Many related series must be modeled together
- Patterns are highly nonlinear

Diagram:

```text
Original series with trend

value
  ^
  |                         *
  |                    *
  |               *
  |          *
  |     *
  +--------------------------------> order

After differencing

change
  ^
  |       *   *      *    *
  |   *          *          *
  |------ stable level -------------
  |      *       *     *
  +--------------------------------> order
```

Example:

```text
36 months of total monthly revenue
no extra features
forecast next 3 months

Good choices:
naive, moving average, exponential smoothing, ARIMA
```

### SARIMA

SARIMA means Seasonal ARIMA.

It is ARIMA plus seasonal behavior.

Use SARIMA when:

- The series has clear seasonality
- The seasonal period is known
- There are one or a few series
- A statistical seasonal model is needed

Common seasonal periods:

| Data frequency | Season length |
|---|---:|
| daily data with weekly pattern | 7 |
| monthly data with yearly pattern | 12 |
| hourly data with daily pattern | 24 |
| hourly data with weekly pattern | 168 |

Diagram:

```text
Daily sales with weekly seasonality

value
  ^
  |        *           *           *
  |      *   *       *   *       *   *
  |    *       *   *       *   *       *
  |  *           *           *
  +--------------------------------------> order
     Mon Tue Wed Thu Fri Sat Sun Mon ...

Idea:
today may depend on yesterday and also on the same weekday last week.
```

Example:

```text
Restaurant daily sales
weekends are high
Mondays are low
forecast next 14 days

Good choices:
seasonal naive, SARIMA, SARIMAX
```

### SARIMAX

SARIMAX means SARIMA with external variables.

The `X` means exogenous variables.

Examples of external variables:

- holiday
- temperature
- promotion
- price
- marketing spend

Use SARIMAX when:

```text
seasonality exists and a few known external variables are useful.
```

Example:

```text
Electricity demand = past demand + daily pattern + temperature
```

### ETS and Exponential Smoothing

ETS stands for:

```text
Error, Trend, Seasonality
```

Use ETS when:

- The data has level, trend, or seasonality
- The dataset is limited
- A simple strong forecast is needed

Example:

```text
Monthly product sales with trend and yearly seasonality.
ETS can be a strong first model.
```

### Tree-Based Machine Learning

Tree models become forecasting models after creating lag features.

Example features:

```text
sales_lag_1
sales_lag_7
sales_lag_14
rolling_mean_7
rolling_mean_28
discount
holiday
weekday
month
```

Feature row example:

```text
day:   1   2   3   4   5   6   7   8
sales: 20  22  25  24  26  30  32  29

For day 8:
target          = sales_day_8
sales_lag_1     = sales_day_7
sales_lag_7     = sales_day_1
rolling_mean_7  = average(day_1 ... day_7)
```

Use tree-based ML when:

- There are many useful features
- Nonlinear relationships exist
- A strong tabular model is needed
- Enough rows are available after creating lags

Good models:

```text
Random Forest
XGBoost
LightGBM
CatBoost
```

### Deep Learning and Seq2Seq

Use deep learning when:

- There is enough data
- There are many related series
- There are many input features
- The output is a future sequence
- Patterns are nonlinear
- Long sequence relationships matter

Typical methods:

| Method | Good for |
|---|---|
| RNN, GRU, LSTM | sequence memory |
| Seq2Seq GRU/LSTM | multi-step future sequence |
| TCN | long sequence modeling |
| Transformer-style models | large multivariate sequence problems |

Avoid deep learning when:

- Very little data is available
- A simple seasonal model already performs well
- Interpretability is more important than accuracy
- Training and tuning resources are limited

### Method Selection Cheat Sheet

| Situation | Good first choices |
|---|---|
| very small data, no seasonality | naive, moving average, ARIMA |
| small data with trend | exponential smoothing, ARIMA |
| clear fixed seasonality | seasonal naive, ETS, SARIMA |
| seasonality plus external variables | SARIMAX |
| many features | tree-based ML with lag features |
| many related series | global ML model or deep learning |
| multi-step sequence output | multi-output ML, Seq2Seq |
| mostly zero demand | intermittent demand methods |

Decision diagram:

```text
Forecasting problem
        |
        v
Build baselines first
        |
        v
Is data small?
        |
   +----+----+
   |         |
  yes        no
   |         |
   v         v
ARIMA,      Many features or many related series?
ETS,        |
SARIMA      +---------+---------+
            |                   |
           yes                  no
            |                   |
            v                   v
   ML with lag features     Classical models may be enough
   or deep learning
            |
            v
   Need sequence output?
            |
       +----+----+
       |         |
      yes        no
       |         |
       v         v
   Seq2Seq,    boosted trees
   TCN,        or direct regression
   Transformer
```

Final rule:

```text
Use the simplest model that gives reliable future-period performance.
```

---

## 16. Why Seq2Seq for Forecasting?

Seq2Seq means:

```text
sequence in -> sequence out
```

In language translation:

```text
French sentence -> English sentence
```

In forecasting:

```text
past values -> future values
```

Seq2Seq is useful when the model must predict a full future sequence.

Examples:

- next 7 days of sales
- next 24 hours of electricity demand
- next 60 minutes of server traffic
- next 14 days of hospital bed demand

---

## 17. Seq2Seq Architecture

Seq2Seq has two main parts:

- encoder
- decoder

### Encoder

The encoder reads the input sequence.

For forecasting:

```text
x1, x2, x3, ..., xL
```

The encoder creates hidden states and a final summary of the past.

### Decoder

The decoder generates the output sequence.

For forecasting:

```text
yhat1, yhat2, yhat3, ..., yhatH
```

Architecture diagram:

```text
Past sequence                          Future sequence

x1 -> x2 -> x3 -> ... -> xL            yhat1 -> yhat2 -> ... -> yhatH
 |    |    |           |                  ^       ^              ^
 v    v    v           v                  |       |              |
[ENC][ENC][ENC] ...   [ENC] --context--> [DEC] -> [DEC] ...    [DEC]

Encoder reads past.
Decoder generates future step by step.
```

Simpler diagram:

```text
past window -> encoder -> context vector -> decoder -> future window
```

---

## 18. Encoder-Decoder Flow

Step-by-step:

```text
1. Input window enters encoder.
2. Encoder updates hidden state at each step.
3. Final encoder hidden state summarizes the past.
4. Decoder starts from this summary.
5. Decoder predicts the first future value.
6. Decoder uses previous information to predict the next value.
7. This continues until the forecast horizon is complete.
```

Unrolled view:

```text
Input window
===================================================
x1        x2        x3                 xL
 |         |         |                  |
 v         v         v                  v
[GRU] --> [GRU] --> [GRU] --> ... --> [GRU]
 h1        h2        h3                 hL
                                      context
                                         |
                                         v
Output window
===================================================
last_y -> [GRU] -> yhat1
          hidden
            |
        yhat1 -> [GRU] -> yhat2
                 hidden
                   |
               yhat2 -> [GRU] -> yhat3 -> ...
```

---

## 19. Teacher Forcing

Teacher forcing is a training technique for decoder models.

During training:

```text
The real future target values are known.
```

During inference:

```text
The real future target values are not known.
```

So the decoder receives different inputs in training and inference.

### Training Mode

The decoder may receive the actual previous target.

```text
last known value -> decoder -> yhat1
actual y1        -> decoder -> yhat2
actual y2        -> decoder -> yhat3
```

### Inference Mode

The decoder must use its own previous prediction.

```text
last known value -> decoder -> yhat1
yhat1            -> decoder -> yhat2
yhat2            -> decoder -> yhat3
```

Numerical example:

```text
True future:
y1 = 10, y2 = 12, y3 = 15

Training:
input = 9       -> yhat1 = 10.3
input = true 10 -> yhat2 = 11.8
input = true 12 -> yhat3 = 14.9

Inference:
input = 9          -> yhat1 = 10.3
input = yhat1 10.3 -> yhat2 = 11.5
input = yhat2 11.5 -> yhat3 = 13.1
```

Key point:

```text
Training can be easier than inference because training may use real previous values.
```

---

## 20. Exposure Bias

Exposure bias is the mismatch between training and inference.

Training:

```text
model often sees clean real previous values
```

Inference:

```text
model sees its own imperfect previous predictions
```

Why this matters:

```text
A small early error can become a larger later error.
```

Example:

```text
true future:      [100, 103, 105, 108]
model prediction: [100, 101,  99,  96]
```

Ways to reduce the problem:

- gradually reduce teacher forcing
- use scheduled sampling
- train with some model-generated inputs
- use attention
- compare with strong baselines

Scheduled sampling idea:

```text
early training: use real targets often
later training: use model predictions more often
```

---

## 21. Attention in Seq2Seq Forecasting

A basic encoder-decoder may compress the whole past into one final hidden state.

For long input windows, that can be too much information for one vector.

Attention allows the decoder to look back at different encoder states.

Intuition:

```text
To predict tomorrow, yesterday may matter most.
To predict next Monday, last Monday may matter more.
```

Diagram:

```text
Encoder states:
h1     h2     h3     h4    ...    hL
 |      |      |      |            |
 +------+------+------+------------+
                 |
              attention
                 |
                 v
           decoder step k
                 |
                 v
             forecast yhat_k
```

Key point:

```text
Attention helps the decoder focus on the most useful parts of the past.
```

---

## 22. Time Features and Cyclical Encoding

Time-related values can become useful input features.

Common features:

- hour of day
- day of week
- weekend flag
- month
- quarter
- holiday flag

### Why Cyclical Encoding?

Hour 23 and hour 0 are close in real life.

But as raw numbers:

```text
23 and 0 look far apart.
```

To fix this, map the cycle onto a circle.

For hour of day:

```text
hour_sin = sin(2 * pi * hour / 24)
hour_cos = cos(2 * pi * hour / 24)
```

Why `2 * pi`?

```text
2 * pi radians = one full circle = 360 degrees
```

So:

```text
hour / 24              -> fraction of the day
2 * pi * hour / 24     -> angle on the full daily circle
sin and cos            -> circular coordinates
```

Clock diagram:

```text
                hour 0
                  |
      hour 18 ----+---- hour 6
                  |
               hour 12

hour 23 is close to hour 0 on the circle.
```

---

## 23. Stationarity

A series is stationary when its statistical behavior is stable.

Simple meaning:

```text
mean and variance do not change dramatically over the series
```

### Stationary Series

```text
value
  ^
  |
  |        *     *       *    *
  |   *       *      *           *
  |------ stable average level ----------------
  |      *       *       *    *
  |  *       *       *
  +--------------------------------------------> order

values fluctuate around a stable level.
spread is also roughly stable.
```

### Non-Stationary Trend

```text
value
  ^
  |
  |                                  *    *
  |                            *   *
  |                      *   *
  |                *   *
  |          *   *
  |    *   *
  +--------------------------------------------> order

average level rises over the series.
```

### Changing Variance

```text
value
  ^
  |
  |                                *       *
  |                          *         *
  |------ average level -------------------------
  |   *  *     *   *       *
  |                    *          *        *
  |                                      *
  +--------------------------------------------> order

average may stay similar, but fluctuations become larger.
```

Common fixes:

| Problem | Possible fix |
|---|---|
| trend | differencing |
| seasonal pattern | seasonal differencing or seasonal features |
| growing variance | log transform |

Differencing example:

```text
Original:    [100, 103, 107, 112, 118]
Difference: [  3,   4,   5,   6]
```

---

## 24. End-to-End Example: Grocery Store Sales

Problem:

```text
Forecast next 7 days of milk sales using previous 30 days.
```

Why it matters:

- avoid empty shelves
- avoid overstock
- reduce waste
- plan ordering
- plan delivery

Available columns:

```text
date
milk_sales
price
discount
is_weekend
is_holiday
temperature
```

Forecasting setup:

```text
target    = milk_sales
frequency = daily
lookback  = 30 days
horizon   = 7 days
features  = milk_sales, price, discount, weekend, holiday, temperature
```

Pipeline:

```text
Raw dated data
      |
      v
Sort by order
      |
      v
Create useful features
      |
      v
Split train, validation, test by order
      |
      v
Fit scaler on train only
      |
      v
Create sliding windows
      |
      v
Train baselines
      |
      v
Train forecasting model
      |
      v
Evaluate MAE, RMSE, horizon-wise error
      |
      v
Use forecast for ordering decision
```

Example forecast:

```text
Predicted milk sales:
Monday:    120
Tuesday:   118
Wednesday: 125
Thursday:  130
Friday:    145
Saturday:  170
Sunday:    165
```

Business decision:

```text
Order more before the weekend because demand is expected to rise.
```

---

## 25. Minimal PyTorch Ideas

Dataset output:

```text
x = previous lookback window
y = future target window
```

Shape example:

```text
x = (lookback, features)
y = (horizon, target_features)
```

Seq2Seq model idea:

```text
encoder_input = x
context = encoder(x)

decoder_input = last known target value

for each future step:
    prediction = decoder(decoder_input, context)
    decoder_input = prediction
```

Training may use teacher forcing:

```text
sometimes decoder_input = true previous target
```

Inference uses no future target:

```text
decoder_input = model's previous prediction
```

---

## 26. Common Mistakes

| Mistake | Why it is wrong | Fix |
|---|---|---|
| random train/test split | future leaks into training | split by order |
| fitting scaler on full data | future distribution leaks | fit scaler on train only |
| skipping baselines | cannot judge model value | compare with naive methods |
| using unknown future features | unrealistic prediction | use only known future features |
| using teacher forcing during inference | uses real future values | use model predictions only |
| reporting only average error | hides horizon weakness | report step-wise error |
| too short lookback | misses seasonality | include enough history |
| too long lookback | adds noise and computation | tune on validation data |

---
