# Actor Recognition with Custom Dataset using PyTorch and Google Colab

### Overview

This repository contains a simple machine learning project for recognizing five different actors based on custom images. Using PyTorch and Google Colab, I trained multiple models with different techniques and evaluated their performance. The dataset consists of 500 images of five well-known actors: Brad Pitt, Johnny Depp, Leonardo DiCaprio, Tom Cruise, and Will Smith.

### Dataset

#### Actors:
- Brad Pitt
- Johnny Depp
- Leonardo DiCaprio
- Tom Cruise
- Will Smith

#### Images:
- 100 images of each actor (total: 500 images)

#### Data split:
- 80% for training
- 20% for testing

### Models and Techniques

#### Model 0: TinyVGG
In the first model, I used TinyVGG, which consists of 2 blocks. Each block contains:
- Convolutional Layer
- Activation Function
- Max Pooling (2D)

After creating the model, I trained it using the training set and evaluated the accuracy and loss curves for both training and testing.

#### Model 1: Data Augmentation with TrivialAugmentWide
In the second model, I applied data augmentation using the TrivialAugmentWide technique for the training set. This helped the model generalize better by randomly applying augmentations such as rotations and flips.

#### Model 2: Extended Epochs
For the third model, I extended the number of training epochs from 5 to 9. This experiment showed an improvement in the model's learning capabilities, which contributed to better performance.

### Results

After training and evaluating all models, I tested them by providing images of the five actors obtained from Google. My model achieved an accuracy of 50%, which, although not impressive, serves as a great starting point for learning about machine learning concepts and the workflow involved in building and testing models.

### How to Run

1. Clone the repository.
2. Install the necessary dependencies using `requirements.txt`.
3. Run the Google Colab notebook to execute the training and testing process.

### Conclusion

This project is a great learning experience for anyone starting with machine learning, computer vision, and PyTorch. It demonstrates the process of working with custom datasets, data augmentation, training models, evaluating performance, and making predictions. Although the accuracy isn't perfect, it is a crucial step in understanding how to apply machine learning concepts practically.

Feel free to fork this repo, experiment with different models, or enhance it with more advanced techniques!
