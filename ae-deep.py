from pydub import AudioSegment
import numpy as np
from keras.layers import Input, Dense
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.callbacks import LambdaCallback
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib 
import os
import sys

songlen = 2500000
# Create an empty np array of size n where n is the total num of songs
song_list = os.listdir("TestSongs")
num_songs = len(song_list)
song_data = np.zeros(shape=(num_songs,songlen))
count = 0
# Get the samples from each song
for fname in os.listdir("TestSongs/"):
    song = AudioSegment.from_wav("TestSongs/"+fname)
    samples = np.array(song.get_array_of_samples())
    samples = samples[:songlen]
    song_data[count] = np.array(samples)
    count = count + 1

encoding_dim = 2

input_song = Input(shape=(songlen,))
encoded = Dense(32, activation='linear')(input_song)
encoded = Dense(8, activation='linear')(encoded)
encoded = Dense(encoding_dim, activation='linear')(encoded)

decoded = Dense(8, activation='linear')(encoded)
decoded = Dense(32, activation='linear')(decoded)
decoded = Dense(songlen, activation='sigmoid')(decoded)
# Map input to its reconstruction
autoencoder = Model(input_song, decoded)

# Map input to its encoded representation
encoder = Model(input_song, encoded)

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

x_train = song_data
x_test = song_data
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
#print(x_train.dtype)
#scaler = MinMaxScaler()
scaler = MinMaxScaler(feature_range=(0,1))
scaler.fit(x_train)
joblib.dump(scaler, 'songScaler.pkl') 

x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
#print(np.min(x_train))
#print(np.max(x_train))

#print(x_train.shape)
#print(x_test.shape)

# define checkpoint callback                                                     
filepath = 'model-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'        
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')                    

def on_epoch_end(epoch, _):
  encoder.save('model-ep'+str(epoch)+'.h5')

# Training the data for e epochs
autoencoder.fit(x_train, x_train,
                epochs=int(sys.argv[1]),
                batch_size=2,
                callbacks=[checkpoint,LambdaCallback(on_epoch_end=on_epoch_end)],
                shuffle=True,
                validation_data=(x_test, x_test))

encoded_songs = encoder.predict(x_test)

print(encoded_songs)

np.save(str(num_songs)+'_deep_encoded_songs_'+sys.argv[1]+'_epochs',encoded_songs)


import matplotlib.pyplot as plt

scaler = MinMaxScaler(feature_range=(0,1))
#scaler = MinMaxScaler()
count = 0
encoded_songs = scaler.fit_transform(encoded_songs)
joblib.dump(scaler, 'displayScaler.pkl') 
print(encoded_songs)
plt.figure()
for fname in os.listdir("TestSongs/"):
  plt.scatter(encoded_songs[count][0], encoded_songs[count][1], s=700,
              c=(int(encoded_songs[count][1]/10.0),0,int(1-encoded_songs[count][1]/10.0)),
              marker=r"$ {} $".format(fname[:4]), edgecolors='none')
  count += 1
plt.show()
