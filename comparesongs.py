from pydub import AudioSegment
import numpy as np
from keras.layers import Input, Dense
from keras.models import Model, load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib 
import os
import sys
import mysql.connector

songlen = 2500000
# Create an empty np array of size n where n is the total num of songs
song_list = os.listdir("TestSongs")
num_songs = len(song_list)
song_data = np.zeros(shape=(num_songs,songlen))
# Get the samples from each song
fname = "Battery.wav"
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
autoencoder = load_model(filename)

x_test = song_data.reshape(1,-1)

x_test = x_test.astype('float32')

scaler = joblib.load('songScaler.pkl') 

x_test = scaler.transform(x_test)

encoded_song = autoencoder.predict(x_test)
print(encoded_song)

scaler = joblib.load('displayScaler.pkl') 
encoded_song = scaler.transform(encoded_song)
print(encoded_song)

# Time to compare the new song to the database

mydb = mysql.connector.connect(
    host="localhost",
    user='root',
    database="honors"
)
mycursor = mydb.cursor()
sql = "SELECT id, name FROM songs WHERE id = (SELECT id FROM song_data ORDER BY ABS (xdata - " + str(encoded_song[0][0]) + ") + ABS (ydata - " + str(encoded_song[0][1]) + ") LIMIT 1)"
print(sql)
mycursor.execute(sql)
result = mycursor.fetchone()
print(result[0], result[1])
mycursor.execute("SELECT xdata, ydata FROM song_data")
result = mycursor.fetchall()

xdata = []
ydata = []
for val in result:
    print(val[0],val[1])
    xdata.append(val[0])
    ydata.append(val[1])
   
"""closest_song = (min(xdata, key=lambda x:abs(x-encoded_song[0][0])), min(ydata, key=lambda x:abs(x-encoded_song[0][1])))
print(closest_song)
sql = "SELECT name from songs INNER JOIN song_data ON songs.id = song_data.id WHERE xdata = " + str(closest_song[0]) + " AND ydata = " + str(closest_song[1])
print(sql)
mycursor.execute(sql)
result = mycursor.fetchone()
print(result[0])"""
