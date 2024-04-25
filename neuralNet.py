#IMPORTS
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials
import io

#function declaration for error loss plot
def plot_loss(history):
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label='val_loss')
  plt.xlabel('Epoch')
  plt.ylabel('Error ')
  plt.legend()
  plt.grid(True)

#VARIABLE DECLARATION
Var1 = 'tailpipe_co2_in_grams_mile_ft1'
Var2 = 'year'
#Var3 = 'fuel_economy_score'
#Var4 = 'engine_cylinders'

#GOOGLE AUTH
auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)

#Streaming CSV file from google drive
downloaded = drive.CreateFile({'id': '1yC29V2fDnIy1465Qe4WtjI77FBjR2eR8'}) # String in the URL when view/downloading the file
df = pd.read_csv(io.StringIO(downloaded.GetContentString()), delimiter=',', on_bad_lines='warn')
df.describe()

#Visualization test
print(df['year'])
print(df.columns)

#Creating Avg_mpg and simple data conversion
df['avg_mpg'] = ((df['city_mpg_ft1'] + df['highway_mpg_ft1'])/2)
print(df['avg_mpg'])
df.convert_dtypes().dtypes

#Dropping non numbers of Variables from data set
print(f"Pre drop-na {len(df)} rows")
df2 = df
df2 = df.dropna(subset=[Var1, Var2])
print(f"After drop {len(df2)} rows")

#Verify column types are numeric
df2 = df2.astype({Var1: float})
df2 = df2.astype({Var2: float})
#df2 = df2.astype({Var3: float})
#df2 = df2.astype({Var4: float})
print(df2.dtypes)

#Data set training ratios, split into different sets
train_dataset = df2.sample(frac=0.8, random_state=0)
test_dataset = df2.drop(train_dataset.index)
#Split data visialization
print("Full dataset size: ", df.size)
print("Training rows", train_dataset.size)
print("Testing rows", test_dataset.size)

#Create features and label dataframes
train_features_df = train_dataset.copy()
test_features_df = test_dataset.copy()
train_labels = train_features_df.pop('avg_mpg')
test_labels = test_features_df.pop('avg_mpg')
print(train_labels)

#Choosing features that will be used for the Neural Network
train_features = train_features_df[[Var1, Var2]]
print(train_features)

#Running nomralizer on training features
normalizer = layers.Normalization(input_shape=[2,], axis=None)
normalizer.adapt(np.array(train_features))
print(normalizer.mean.numpy())
print(normalizer.get_weights())

#Normalized model inspection
model = tf.keras.Sequential([
    normalizer,
    layers.Dense(units=2),
    layers.Dense(units=1)
])
model.summary()

#Pre-prediction test
model.predict(train_features[:10])

#Compiling Model
model.compile(
    #optimizer=tf.optimizers.Adam(learning_rate=0.05),
    optimizer=tf.optimizers.Adam(learning_rate=0.0005),
    loss='mean_squared_error')
#mean_absolute_error
#MeanSquaredLogarithmicError
#mean_squared_error

#Fitting Model and running nerual network
%%time
history = model.fit(
    train_features,
    train_labels,
    epochs=20,
    verbose=1,
    # Calculate validation results on 20% of the training data.
    validation_split = 0.2)

#Plotting Data
plot_loss(history)


