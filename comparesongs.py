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
#song_list = os.listdir("TestSongs")
#num_songs = len(song_list)
#song_data = np.zeros(shape=(num_songs,songlen))
# Get the samples from each song
#fname = "Wokeuplikethis.wav"
import glob
list_of_files = glob.glob('Uploads/*.wav')
fname = max(list_of_files, key=os.path.getctime)
#print(fname)
song = AudioSegment.from_wav(fname)
samples = np.array(song.get_array_of_samples())
samples = samples[:songlen]
song_data = np.array(samples)
encoding_dim = 2

input_song = Input(shape=(songlen,))
#print(input_song.shape)

# Map input to its reconstruction
"""import glob
list_of_files = glob.glob('model*.h5')
filename = max(list_of_files, key=os.path.getctime)"""
filename = 'this-is-the-model.h5'
autoencoder = load_model(filename)
#print(filename)
x_test = song_data.reshape(1,-1)

x_test = x_test.astype('float32')

scaler = joblib.load('songScaler.pkl') 

x_test = scaler.transform(x_test)

encoded_song = autoencoder.predict(x_test)
#print(encoded_song)

scaler = joblib.load('displayScaler.pkl') 
encoded_song = scaler.transform(encoded_song)
#print(encoded_song)

# Time to compare the new song to the database

mydb = mysql.connector.connect(
    host="localhost",
    user='root',
    database="honors"
)
mycursor = mydb.cursor()
#sql = "SELECT name FROM songs WHERE id = (SELECT id FROM song_data ORDER BY ABS (xdata - " + str(encoded_song[0][0]) + ") + ABS (ydata - " + str(encoded_song[0][1]) + ") LIMIT 1)"
#print(sql)
#mycursor.execute(sql)
#result = mycursor.fetchone()
#data = result[0]
#print(data)
#sys.stdout.flush()
mycursor.execute("SELECT name, xdata, ydata FROM song_data")
result = mycursor.fetchall()
import matplotlib.pyplot as plt
import math
alls = {}
plt.figure(figsize=(10,10))
for val in result:
    plt.plot(val[1],val[2],'k.')
    alls[val[0]] = [val[1],val[2],math.sqrt(((val[1]-encoded_song[0][0])**2)+((val[2]-encoded_song[0][1])**2))]
#print(encoded_song)
bestk = ''
bestv = []
bestd = 1000000000000000

for key in alls.keys():
    if alls[key][2] < bestd:
        bestd=alls[key][2]
        bestv=alls[key]
        bestk=key
        #print(bestv)

plt.plot(encoded_song[0][0], encoded_song[0][1], 'go')
plt.plot(bestv[0],bestv[1],'ro')
plt.title('Distance to song:' + str(bestd) + '. Recommended Song: ' + bestk)
plt.xlabel("First autoencoder dimension")
plt.ylabel("Second autoencoder dimension")
plt.savefig('result-'+bestk+'-.png',dpi=350)
print('result-'+bestk+'-.png');
#plt.savefig('result.png',dpi=350)
#plt.show()
#print(bestk)
plt.clf()
plt.close()

   
"""closest_song = (min(xdata, key=lambda x:abs(x-encoded_song[0][0])), min(ydata, key=lambda x:abs(x-encoded_song[0][1])))
print(closest_song)
sql = "SELECT name from songs INNER JOIN song_data ON songs.id = song_data.id WHERE xdata = " + str(closest_song[0]) + " AND ydata = " + str(closest_song[1])
print(sql)
mycursor.execute(sql)
result = mycursor.fetchone()
print(result[0])"""
