from pydub import AudioSegment
import numpy as np
from keras.layers import Input, Dense
from keras.models import Model, load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib 
import os
import sys

songlen = 2500000
# Create an empty np array of size n where n is the total num of songs
song_list = os.listdir("TestSongs")
num_songs = len(song_list)
song_data = np.zeros(shape=(num_songs,songlen))
# Get the samples from each song
fname = "africa-toto.wav"
song = AudioSegment.from_wav("TestSongs/"+fname)
samples = np.array(song.get_array_of_samples())
samples = samples[:songlen]
song_data = np.array(samples)
encoding_dim = 2

input_song = Input(shape=(songlen,))
print(input_song.shape)
#encoded = Dense(32, activation='linear')(input_song)
#encoded = Dense(8, activation='linear')(encoded)
#encoded = Dense(encoding_dim, activation='linear')(input_song)
#print(encoded)
#decoded = Dense(8, activation='linear')(encoded)
#decoded = Dense(32, activation='linear')(decoded)
#decoded = Dense(songlen, activation='sigmoid')(encoded)
# Map input to its reconstruction
import glob
list_of_files = glob.glob('model*.h5')
filename = max(list_of_files, key=os.path.getctime)
print(filename)
autoencoder = load_model(filename)

# Map input to its encoded representation
#encoder = Model(input_song, encoded)

#autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

#x_train = song_data.reshape(1,-1)
x_test = song_data.reshape(1,-1)
#x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
#print(x_train.dtype)
#scaler = MinMaxScaler()
scaler = joblib.load('songScaler.pkl') 
#x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
#print(np.min(x_train))
#print(np.max(x_train))

#print(x_train.shape)
#print(x_test.shape)


encoded_song = autoencoder.predict(x_test)
print(encoded_song)

scaler = joblib.load('displayScaler.pkl') 
encoded_song = scaler.transform(encoded_song)
print(encoded_song)


