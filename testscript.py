from pydub import AudioSegment
import numpy as np
from keras.layers import Input, Dense
from keras.models import Model

# Get samples from song1
song1 = AudioSegment.from_wav("TestSongs/California_Gurls.wav")
samples1 = np.array(song1.get_array_of_samples())
print(samples1.shape)

# Get samples from song2
song2 = AudioSegment.from_wav("TestSongs/Take_My_Bones_Away.wav")
samples2 = np.array(song2.get_array_of_samples())
print(samples2.shape)

encoding_dim = 2

input_song = Input(shape=(2883550,))
encoded = Dense(encoding_dim, activation='relu')(input_song)
decoded = Dense(2883550, activation='sigmoid')(encoded)
# Map input to its reconstruction
autoencoder = Model(input_song, decoded)

# Map input to its encoded representation
encoder = Model(input_song, encoded)

encoded_input = Input(shape=(encoding_dim,))
decoder_layer = autoencoder.layers[-1]
decoder = Model(encoded_input, decoder_layer(encoded_input))

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

#x_train = samples1
x_train = np.array([samples1, samples2])
x_test = np.array([samples1, samples2])

print(x_train.shape)
print(x_test.shape)

# Training the data for 50 epochs
autoencoder.fit(x_train, x_train,
                epochs=50,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))

encoded_songs = encoder.predict(x_test)
decoded_songs = decoder.predict(encoded_songs)

print(encoded_songs)
print(decoded_songs)
