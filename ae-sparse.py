from pydub import AudioSegment
import numpy as np
from keras.layers import Input, Dense
from keras.models import Model
from keras import regularizers
from sklearn.preprocessing import MinMaxScaler
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
encoded = Dense(encoding_dim, activation='relu', activity_regularizer=regularizers.l1(10e-5))(input_song)
decoded = Dense(songlen, activation='sigmoid')(encoded)
# Map input to its reconstruction
autoencoder = Model(input_song, decoded)

# Map input to its encoded representation
encoder = Model(input_song, encoded)

encoded_input = Input(shape=(encoding_dim,))
decoder_layer = autoencoder.layers[-1]
decoder = Model(encoded_input, decoder_layer(encoded_input))

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

x_train = song_data
x_test = song_data
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
#print(x_train.dtype)
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
#print(np.min(x_train))
#print(np.max(x_train))

#print(x_train.shape)
#print(x_test.shape)

# Training the data for e epochs
autoencoder.fit(x_train, x_train,
                epochs=int(sys.argv[1]),
                batch_size=2,
                shuffle=True,
                validation_data=(x_test, x_test))

encoded_songs = encoder.predict(x_test)
decoded_songs = decoder.predict(encoded_songs)

print(encoded_songs)
print(decoded_songs)

np.save(str(num_songs)+'_sparse_encoded_songs_'+sys.argv[1]+'_epochs',encoded_songs)


import matplotlib.pyplot as plt

count = 0
plt.figure()
for fname in os.listdir("TestSongs/"):
  plt.scatter(encoded_songs[count][0], encoded_songs[count][1], s=700,
              c=(encoded_songs[count][1]/10.0,0,1-encoded_songs[count][1]/10.0),
              marker=r"$ {} $".format(fname[:2]), edgecolors='none')
plt.show()
