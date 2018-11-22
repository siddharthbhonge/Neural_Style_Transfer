# What if Statue of Liberty was painted by Picasso?

Neural Style learning is one of the most exciting sides of deep learning.Ever wondered if Angelina Jolie was painted by Da Vinci?
 
[Demo (Original MP3)](https://soundcloud.com/siddharth-bhonge/original?in=siddharth-bhonge/sets/lstm-output) | 
[Demo (Generated MP3)](https://soundcloud.com/siddharth-bhonge/generated?in=siddharth-bhonge/sets/lstm-output)

[![screenshot](https://github.com/siddharthbhonge/Piano_music_generation_using_LSTM/blob/master/img/images.jpeg)
## Installation

 - Keras
 - TensorFlow
 - Python MIDI
-  Numpy

## Details

  #### What is actually happening?
  
  Neural Style Transfer (NST) uses a previously trained convolutional network, and builds on top of that. The idea of using a network trained on a different task and applying it to a new task is called transfer learning.  <br />

    Neural Style Transfer is an algorithm that given a content image C and a style image S can generate an artistic image <br />
    It uses representations (hidden layer activations) based on a pretrained ConvNet.
    The content cost function is computed using one hidden layer's activations.<br />
    The style cost function for one layer is computed using the Gram matrix of that layer's activations. 
The overall style cost function is obtained using several hidden layers.<br/>
    Optimizing the total cost function results in synthesizing new images.<br />
  
  

  ####  Content Cost
[![screenshot](https://github.com/siddharthbhonge/Piano_music_generation_using_LSTM/blob/master/img/images.jpeg)

  



 #### Style Cost

[![screenshot](https://github.com/siddharthbhonge/Piano_music_generation_using_LSTM/blob/master/img/images.jpeg)


 #### Total Cost

[![screenshot](https://github.com/siddharthbhonge/Piano_music_generation_using_LSTM/blob/master/img/images.jpeg)



## Note

The style of an image can be represented using the Gram matrix of a hidden layer's activations. However, we get even better results combining this representation from multiple different layers. This is in contrast to the content representation, where usually using just a single hidden layer is sufficient.<br />
Minimizing the style cost will cause the image GG to follow the style of the image SS. <br />
 
Please Download VGG19 weights and keep in /model folder.https://gist.github.com/baraldilorenzo/8d096f48a1be4a2d660d
## Acknowledgemnts 

*Siddharth Bhonge https://github.com/siddharthbhonge 




## Reference

Andrew Ng's Deep Learning Specialization.<br />

