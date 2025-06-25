import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow.keras.backend as K

# 1) Carregar e preparar a série temporal mensal
df10 = pd.read_csv('/IMP_2010.csv', sep=';')
df11 = pd.read_csv('/IMP_2011.csv', sep=';')
df12 = pd.read_csv('/IMP_2012.csv', sep=';')
df13 = pd.read_csv('/IMP_2013.csv', sep=';')
df14 = pd.read_csv('/IMP_2014.csv', sep=';')
df15 = pd.read_csv('/IMP_2015.csv', sep=';')
df16 = pd.read_csv('/IMP_2016.csv', sep=';')
df17 = pd.read_csv('/IMP_2017.csv', sep=';')
df18 = pd.read_csv('/IMP_2018.csv', sep=';')
df19 = pd.read_csv('/IMP_2019.csv', sep=';')
df20 = pd.read_csv('/IMP_2020.csv', sep=';')
df21 = pd.read_csv('/IMP_2021.csv', sep=';')
df22 = pd.read_csv('/IMP_2022.csv', sep=';')
df23 = pd.read_csv('/IMP_2023.csv', sep=';')
df24 = pd.read_csv('/IMP_2024.csv', sep=';')
df = pd.concat([df10, df11, df12, df13, df14, df15, df16, df17, df18, df19, df20, df21, df22, df23, df24], ignore_index=True)

#31052000
#84713012
df_main = df[(df['CO_NCM'] == 84713012)].copy()
df_main['CO_ANO'] = df['CO_ANO'].astype(int)
df_main['CO_MES'] = df['CO_MES'].astype(int)
df_main['date'] = pd.to_datetime(df_main['CO_ANO'].astype(str) + '-' + df_main['CO_MES'].astype(str) + '-01')
ts = df_main.groupby('date')['KG_LIQUIDO'].sum().sort_index()

# 2) Normalizar (MinMax)
values = ts.values.reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)

# 3) Gerar janelas (window = 12 meses)
window = 12
X, y = [], []
for i in range(window, len(scaled)):
    X.append(scaled[i-window:i, 0])
    y.append(scaled[i, 0])
X, y = np.array(X), np.array(y)

# 4) Reshape para [samples, timesteps, features]
X = X.reshape((X.shape[0], X.shape[1], 1))

# 5) Separar treino e teste (80% treino)
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# --- 1. Definir a métrica R² ---

def r2_keras(y_true, y_pred):
    ss_res = K.sum(K.square(y_true - y_pred))
    ss_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - ss_res / (ss_tot + K.epsilon())


# 6) Definir o modelo LSTM
model = Sequential([
    LSTM(50, activation='sigmoid', input_shape=(window, 1),
         dropout=0.20,            # 20% de dropout nas entradas
         recurrent_dropout=0.20   # 20% de dropout nos estados recorrentes
         ),
    Dropout(0.20),
    Dense(10),
    Dense(1)
])

opt = tf.keras.optimizers.Adam(learning_rate=0.0005)
rmse_metric = tf.keras.metrics.RootMeanSquaredError(name="rmse")


def rmse_loss(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_true - y_pred)))

#mse
model.compile(optimizer=opt, loss=rmse_loss, metrics=['mape', 'mae',
                                                  'mse', rmse_metric])

#model.load_weights('/content/melhor_modelo_MAPE.h5')

checkpoint_cb = ModelCheckpoint(
    'melhor_modelo_.h5',
    monitor='val_loss',
    save_best_only=True,
    save_weights_only=False,
    mode='min'
)

## 7) Treinar
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=25,
    validation_data=(X_test, y_test),
    callbacks=[checkpoint_cb],
    verbose=2
)

# 8) Previsão e inversão de escala
y_pred = model.predict(X_test)
y_pred_inv = scaler.inverse_transform(y_pred)
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

# 9) Plotar resultados
plt.figure(figsize=(8, 4))
plt.plot(ts.index[window + split:], y_test_inv, label='Real')
plt.plot(ts.index[window + split:], y_pred_inv, label='Previsto', linestyle='--')
plt.title('LSTM: Previsão de KG_LIQUIDO (NCM 84713012)')
plt.xlabel('Data')
plt.ylabel('KG Líquido')
plt.legend()
plt.tight_layout()
plt.show()