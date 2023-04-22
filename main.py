from time import sleep
import pyaudio
import wave
import sys
import librosa
import numpy
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image


CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100

def audio_recorder():
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    
    for i in range(0,100):
        print("Start recording...")
        frame =[]
        second = 0
        progress = 0
        while True:
            progress+=1
            data = numpy.fromstring(stream.read(CHUNK),dtype=numpy.int16)
            frame.append(data)
            second = progress*CHUNK/RATE
            if second >= 1:
                break
        audio_data = numpy.concatenate(frame).astype(numpy.float32)
        onset_env = librosa.onset.onset_strength(y=audio_data, sr=RATE) 
        peaks = librosa.util.peak_pick(onset_env, pre_max=1, post_max=1, pre_avg=1, post_avg=1, delta=2, wait=5)
        if len(peaks)>0:
            getspectrogram(audio_data)
            reconizer()
            
            
    stream.stop_stream()
    stream.close()
    p.terminate()
    
def getspectrogram(audio_data):
    print('Generating spectrogram...')
    plt.figure(figsize=(10, 10))
    D = numpy.abs(librosa.stft(audio_data))
    librosa.display.specshow(librosa.amplitude_to_db(D, ref=numpy.max))
    plt.style.use('dark_background')
    plt.tight_layout()
    plt.savefig(os.path.join("temp/",'{}.png'.format('temp')) ,bbox_inches='tight',pad_inches=0)
    plt.close()

def reconizer():
    print('Recognizing...')
    img = Image.open('temp/temp.png')
    img = img.resize((224, 224))
    img_array = numpy.array(img)
    if img_array.shape[-1] == 4:
        img_array = img_array[..., :3]
    img_array = numpy.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.0
    model = tf.keras.models.load_model('converted_keras/keras_model.h5')
    prediction = model.predict(img_array)
    labels = ['non-cough', 'cough']
    predicted_label = labels[numpy.argmax(prediction)]
    print('Predicted label:', predicted_label, 'with a confidence of', prediction[0][numpy.argmax(prediction)]*100, '%')
    # clear command line
    sleep(2)
    os.system('cls' if os.name == 'nt' else 'clear')
    
audio_recorder()