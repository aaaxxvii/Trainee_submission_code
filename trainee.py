# 1. ライブラリのインポート

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import optuna
from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import os
import random
from optuna.samplers import TPESampler


# 再現性のためのシード固定
SEED_VALUE = 42
os.environ['PYTHONHASHSEED'] = str(SEED_VALUE)
random.seed(SEED_VALUE)
np.random.seed(SEED_VALUE)
tf.random.set_seed(SEED_VALUE)
# この行を追加することで、GPU上での計算の再現性が確保されます
tf.config.experimental.enable_op_determinism()



# 2. データの前処理

#データの読み込み
df = pd.read_csv("stock_price.csv")

# カラム名を英語に変更
column_mapping = {
    '日付け': 'Date', '終値': 'Close', '始値': 'Open', '高値': 'High',
    '安値': 'Low', '出来高': 'Volume', '変化率 %': 'Change_Percentage'
}
df.rename(columns=column_mapping, inplace=True)

# Date列をdatetime型に変換
df['Date'] = pd.to_datetime(df['Date'])

# 日付をインデックスに設定し、昇順に並び替え
df.set_index('Date', inplace=True)
df.sort_index(inplace=True)

# データ型を数値に変換
def convert_volume(volume_str):
    if isinstance(volume_str, str):
        volume_str = volume_str.strip()
        if volume_str.endswith('M'): return float(volume_str[:-1]) * 1_000_000
        elif volume_str.endswith('B'): return float(volume_str[:-1]) * 1_000_000_000
    return float(volume_str)
df['Volume'] = df['Volume'].apply(convert_volume)
df['Change_Percentage'] = df['Change_Percentage'].str.replace('%', '').astype(float)



# 3. ベースラインモデルの構築と評価

target_col_base = 'Close'
features_cols_base = df.columns.tolist()
dataset_base = df[features_cols_base].values

# データを訓練用とテスト用に分割
training_data_len_base = int(len(dataset_base) * 0.8)
train_data_base = dataset_base[:training_data_len_base]
test_data_base = dataset_base[training_data_len_base:]

# スケーラーを作成し、訓練データで学習
scaler_base = MinMaxScaler(feature_range=(0, 1))
scaler_base.fit(train_data_base)
scaled_train_data_base = scaler_base.transform(train_data_base)
scaled_test_data_base = scaler_base.transform(test_data_base)

# 目的変数('Close')を元のスケールに戻すためのスケーラーを別途作成
target_col_index_base = features_cols_base.index(target_col_base)
close_scaler_base = MinMaxScaler(feature_range=(0,1))
close_scaler_base.fit(train_data_base[:, target_col_index_base].reshape(-1, 1))

# LSTM用のデータセットを作成する関数
def create_dataset_base(data, time_step=1):
    X, y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:(i + time_step), :])
        y.append(data[i + time_step, target_col_index_base])
    return np.array(X), np.array(y)

time_step_base = 60
X_train_base, y_train_base = create_dataset_base(scaled_train_data_base, time_step_base)
X_test_base, y_test_scaled_base = create_dataset_base(scaled_test_data_base, time_step_base)

# ベースラインLSTMモデルの構築
baseline_model = Sequential()
baseline_model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train_base.shape[1], X_train_base.shape[2])))
baseline_model.add(Dropout(0.2))
baseline_model.add(LSTM(units=50, return_sequences=False))
baseline_model.add(Dropout(0.2))
baseline_model.add(Dense(units=25))
baseline_model.add(Dense(units=1))

baseline_model.compile(optimizer='adam', loss='mean_squared_error')
baseline_model.fit(X_train_base, y_train_base, batch_size=32, epochs=100, validation_split=0.1, verbose=1)

# ベースラインモデルの予測と評価
predictions_scaled_base = baseline_model.predict(X_test_base)
predictions_base = close_scaler_base.inverse_transform(predictions_scaled_base)
y_test_actual_base = close_scaler_base.inverse_transform(y_test_scaled_base.reshape(-1, 1))

# 評価指標の計算
rmse_base = np.sqrt(mean_squared_error(y_test_actual_base, predictions_base))
mae_base = mean_absolute_error(y_test_actual_base, predictions_base)
r2_base = r2_score(y_test_actual_base, predictions_base)
def calculate_mape(y_true, y_pred):
    epsilon = 1e-10
    return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100
mape_base = calculate_mape(y_test_actual_base, predictions_base)

# 方向性精度の計算
valid_base = df[training_data_len_base+time_step_base:].copy()
valid_base['Predictions'] = predictions_base
valid_base['Actual_Prev_Daily'] = valid_base['Close'].shift(1)
valid_base['Actual_Prev_Weekly'] = valid_base['Close'].shift(5)
valid_base.dropna(inplace=True)
valid_base['Predicted_Movement_Daily'] = (valid_base['Predictions'] > valid_base['Actual_Prev_Daily']).astype(int)
valid_base['Actual_Movement_Daily'] = (valid_base['Close'] > valid_base['Actual_Prev_Daily']).astype(int)
directional_accuracy_daily_base = np.mean(valid_base['Predicted_Movement_Daily'] == valid_base['Actual_Movement_Daily']) * 100
valid_base['Predicted_Movement_Weekly'] = (valid_base['Predictions'] > valid_base['Actual_Prev_Weekly']).astype(int)
valid_base['Actual_Movement_Weekly'] = (valid_base['Close'] > valid_base['Actual_Prev_Weekly']).astype(int)
directional_accuracy_weekly_base = np.mean(valid_base['Predicted_Movement_Weekly'] == valid_base['Actual_Movement_Weekly']) * 100

# 結果の表示（方向性精度を追加）
print("\nベースラインモデルの評価指標")
print(f"ベースラインモデル RMSE: {rmse_base:.4f}")
print(f"ベースラインモデル MAE: {mae_base:.4f}")
print(f"ベースラインモデル R2 Score: {r2_base:.4f}")
print(f"ベースラインモデル MAPE: {mape_base:.2f}%")
print(f"ベースラインモデル 方向性精度 (1日単位): {directional_accuracy_daily_base:.2f}%")
print(f"ベースラインモデル 方向性精度 (1週間単位): {directional_accuracy_weekly_base:.2f}%")



# 4. 特徴量エンジニアリング
df['SMA_25'] = df['Close'].rolling(window=25).mean()
df['SMA_75'] = df['Close'].rolling(window=75).mean()
exp12 = df['Close'].ewm(span=12, adjust=False).mean()
exp26 = df['Close'].ewm(span=26, adjust=False).mean()
df['MACD'] = exp12 - exp26
df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
delta = df['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
df['RSI'] = 100 - (100 / (1 + rs))
df['BB_mid'] = df['Close'].rolling(window=20).mean()
df['BB_std'] = df['Close'].rolling(window=20).std()
df['BB_upper'] = df['BB_mid'] + (df['BB_std'] * 2)
df['BB_lower'] = df['BB_mid'] - (df['BB_std'] * 2)
df.dropna(inplace=True)



# 5. 最終モデルのためのデータ準備 (特徴量エンジニアリング後)

target_col = 'Close'
features_cols = df.columns.tolist()
dataset = df[features_cols].values

# データを訓練用とテスト用に分割
training_data_len = int(len(dataset) * 0.8)
train_data = dataset[:training_data_len]
test_data = dataset[training_data_len:]

# スケーラーを作成し、訓練データで学習
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(train_data)
scaled_train_data = scaler.transform(train_data)
scaled_test_data = scaler.transform(test_data)

# 目的変数('Close')を元のスケールに戻すためのスケーラーを別途作成
target_col_index = features_cols.index(target_col)
close_scaler = MinMaxScaler(feature_range=(0,1))
close_scaler.fit(train_data[:, target_col_index].reshape(-1, 1))

# LSTM用のデータセットを作成する関数
def create_dataset(data, time_step=1):
    X, y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:(i + time_step), :])
        y.append(data[i + time_step, target_col_index])
    return np.array(X), np.array(y)

time_step = 60
X_train, y_train = create_dataset(scaled_train_data, time_step)
X_test, y_test_scaled = create_dataset(scaled_test_data, time_step)
print(f"\n最終モデル用の訓練データ形状: {X_train.shape}")


# 6. ハイパーパラメータチューニング (Optunaによるベイズ最適化)

# サンプラーにシードを設定して再現性を確保
sampler = TPESampler(seed=SEED_VALUE)
study = optuna.create_study(direction='minimize', sampler=sampler)

# 最適化を行うための関数を定義
def objective(trial):
    # ハイパーパラメータの探索範囲を定義
    units = trial.suggest_int('units', 32, 128)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    optimizer_name = trial.suggest_categorical('optimizer', ['adam', 'rmsprop'])

    # LSTMモデルの構築
    model = Sequential()
    model.add(LSTM(units=units, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(units=units, return_sequences=False))
    model.add(Dropout(dropout_rate))
    model.add(Dense(units=25))
    model.add(Dense(units=1))

    optimizer = tf.keras.optimizers.get(optimizer_name)
    optimizer.learning_rate = learning_rate
    model.compile(optimizer=optimizer, loss='mean_squared_error')

    # 時系列データ用の交差検証
    tscv = TimeSeriesSplit(n_splits=3)
    val_scores = []
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    for train_index, val_index in tscv.split(X_train):
        x_t, x_v = X_train[train_index], X_train[val_index]
        y_t, y_v = y_train[train_index], y_train[val_index]

        model.fit(x_t, y_t, epochs=100, batch_size=32, validation_data=(x_v, y_v),
                      callbacks=[early_stopping], verbose=0)

        preds = model.predict(x_v)
        rmse = np.sqrt(mean_squared_error(y_v, preds))
        val_scores.append(rmse)

    return np.mean(val_scores)

# 最適化の実行
print("\nベイズ最適化を開始")
study.optimize(objective, n_trials=20)

# 結果の確認
print("\n最適化が完了")
print(f"試行回数: {len(study.trials)}")
print(f"最適なスコア (Validation RMSE): {study.best_value:.4f}")
print("最適なハイパーパラメータ:")
best_params = study.best_params
for key, value in best_params.items():
    print(f"  {key}: {value}")


# 7. 最終モデルの構築と学習

# 最適なパラメータを使ってモデルを構築
final_model = Sequential()
final_model.add(LSTM(units=best_params['units'], return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
final_model.add(Dropout(best_params['dropout_rate']))
final_model.add(LSTM(units=best_params['units'], return_sequences=False))
final_model.add(Dropout(best_params['dropout_rate']))
final_model.add(Dense(units=25))
final_model.add(Dense(units=1))

# 最適なオプティマイザと学習率を設定
optimizer_name = best_params['optimizer']
learning_rate = best_params['learning_rate']
optimizer = tf.keras.optimizers.get(optimizer_name)
optimizer.learning_rate = learning_rate

final_model.compile(optimizer=optimizer, loss='mean_squared_error')

print("\n最終モデルの構造:")
final_model.summary()

# 最終モデルを再学習
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

print("\n最終モデルのトレーニングを開始...")
history = final_model.fit(
    X_train,
    y_train,
    epochs=150,
    batch_size=32,
    validation_split=0.1,
    callbacks=[early_stopping],
    verbose=1
)
print("最終モデルのトレーニングが完了")


# 8. 最終モデルの予測と評価

# テストデータで予測を実行
predictions_scaled = final_model.predict(X_test)

# 予測値と正解ラベルを元のスケールに戻す
predictions = close_scaler.inverse_transform(predictions_scaled)
y_test_actual = close_scaler.inverse_transform(y_test_scaled.reshape(-1, 1))

# 評価指標の計算
rmse = np.sqrt(mean_squared_error(y_test_actual, predictions))
mae = mean_absolute_error(y_test_actual, predictions)
r2 = r2_score(y_test_actual, predictions)
mape = calculate_mape(y_test_actual, predictions)

# 方向性精度の計算
valid = df[training_data_len+time_step:].copy()
valid['Predictions'] = predictions
valid['Actual_Prev_Daily'] = valid['Close'].shift(1)
valid['Actual_Prev_Weekly'] = valid['Close'].shift(5)
valid.dropna(inplace=True)
valid['Predicted_Movement_Daily'] = (valid['Predictions'] > valid['Actual_Prev_Daily']).astype(int)
valid['Actual_Movement_Daily'] = (valid['Close'] > valid['Actual_Prev_Daily']).astype(int)
directional_accuracy_daily = np.mean(valid['Predicted_Movement_Daily'] == valid['Actual_Movement_Daily']) * 100
valid['Predicted_Movement_Weekly'] = (valid['Predictions'] > valid['Actual_Prev_Weekly']).astype(int)
valid['Actual_Movement_Weekly'] = (valid['Close'] > valid['Actual_Prev_Weekly']).astype(int)
directional_accuracy_weekly = np.mean(valid['Predicted_Movement_Weekly'] == valid['Actual_Movement_Weekly']) * 100

# 結果の表示
print("\n最終モデルの評価指標")
print(f"最終モデル RMSE: {rmse:.4f}")
print(f"最終モデル MAE: {mae:.4f}")
print(f"最終モデル R2 Score: {r2:.4f}")
print(f"最終モデル MAPE: {mape:.2f}%")
print(f"最終モデル 方向性精度 (1日単位): {directional_accuracy_daily:.2f}%")
print(f"最終モデル 方向性精度 (1週間単位): {directional_accuracy_weekly:.2f}%")


# 予測結果の可視化
train_plot = df[:training_data_len]
plt.figure(figsize=(16, 8))
plt.title('Final Model - Prediction vs Actual')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price (JPY)', fontsize=18)
plt.plot(train_plot['Close'], label='Train')
plt.plot(valid['Close'], label='Actual')
plt.plot(valid['Predictions'], label='Predictions')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()