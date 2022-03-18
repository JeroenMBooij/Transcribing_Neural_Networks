# Keras Tensorflow Neural Network Recognition API

This repository contains the source code for neural networks with transcribing capabilities in a Django API. The API is described with Swagger OpenAPI 3.0. See deployment below on how to open the swagger documentation page. `</br></br>`

The goal for this project was to learn about the math involved with neural network and how to construct the layers in a neural network. This learning goal was part of my data science minor, where I build a handwriting recognition neural network based on a research paper from Graves A. & Schmidhuber J. called `<a href='https://people.idsia.ch/~juergen/nips2009.pdf' target='_blank'>`Offline Handwriting Recognition with Multidimensional Recurrent Neural Networks `</a>`

`<img src="https://github.com/JeroenMBooij/Transcribing_Neural_Networks/blob/main/images/htr.png"></img>`

The goal for this project was to learn about the math involved with neural network and how to construct the layers in a neural network. This learning goal was part of my data science minor, where I build a handwriting recognition neural network based on a research paper from Graves A. & Schmidhuber J. called `<a href='https://people.idsia.ch/~juergen/nips2009.pdf' target='_blank'>`Offline Handwriting Recognition with Multidimensional Recurrent Neural Networks `</a>`

`<img src="https://github.com/JeroenMBooij/Transcribing_Neural_Networks/blob/main/images/htr%20results.png"></img>`

<h1>Deployment </h1>

* prerequisite - docker installed `<br/><br/>`
* optionally: `<br/>`
  * add a keras checkpoint hdf5 file called handwritten_text_model.hdf5 inside src/apps/computer_vision_services_text_reader/files if you do not have this file you will have to use the computer vision train text endpoint to generate the file. This will take a couple of hours. The file is also too large too upload to Github. `<br/>`
  * add a keras checkpoint hdf5 file called all_handwritten_characters_model.hdf5 inside src/apps/computer_vision_services_text_reader/files/models


`<b>`steps `</b>`

<ol>
  <li>Run "docker-compose up -d" from the root directory of this project</li>
  <li>Open localhost:8000</li>
 </ol>
</br>
