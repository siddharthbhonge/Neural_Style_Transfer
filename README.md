# Improvising Beethoven using LSTM (Jetson TX2)

Most imporvisation networks are pretty heavy and use Sequence to sequence models.Here,midi library in pythonn is used to convert the notes into musical information such as Note on ,Set tempo and Control Change.
This text file is later used by LSTM network to train.We use this network to predict the next feature.

[Demo (Original MP3)](https://soundcloud.com/siddharth-bhonge/original?in=siddharth-bhonge/sets/lstm-output) | 
[Demo (Generated MP3)](https://soundcloud.com/siddharth-bhonge/generated?in=siddharth-bhonge/sets/lstm-output)

[![screenshot](https://github.com/siddharthbhonge/Piano_music_generation_using_LSTM/blob/master/img/images.jpeg)
## Installation

 - Keras
 - TensorFlow
 - Python MIDI
-  Numpy

## Details

  #### 1. Learn MIDI Library  details
  
  Python Midi library is not available for pip install.It is maintained here:https://github.com/vishnubob/python-midi <br />
  Midi supports total 27 events.We have used 3: <br/>
  1."Note On" <br />
  2."Set Tempo" <br />
  3."Control Change"<br />

  These are sufficient for Beethoven Classical Piano midi files.http://www.piano-midi.de/beeth.htm <br />
  For other types of music please use other events this library supports.<br />

  #### 2. Parse MIDI file to text

  ```
  python3 parse_midi_to_text.py

  ```

  #### 3. Train the model (GPU recommended)

  ```
  python3 training.py

  ```

  #### 4. Generate music
  This is done on Nvidia Jetson TX2. But any GPU is fine.The aim to record files on fly and improvise them. 

  ```
  python generate_music.py
  ```

## Note

 - Change the file `original_song.mid` to the file you want.We added an USB microphone to record on Jetson<br />
 - You may have to choose track as index of the pattern in `parse_midi_to_text.py`.However,the midi library does have this feature.<br />


## Acknowledgemnts 

*Siddharth Bhonge https://github.com/siddharthbhonge 




## Reference

Sequnce to Sequnce Model from Andrew Ng's Deep Learning Specialization.<br />
But personally this is a better one.