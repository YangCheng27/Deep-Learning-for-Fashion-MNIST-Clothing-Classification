# Deep-Learning-for-Fashion-MNIST-Clothing-Classification
Build a simple deep learning model for predicting labels of 28x28 greyscale images from the Fashion-MNIST dataset.
## Open Source Dataset : [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist)
Fashion-MNIST is a dataset of Zalando's article images—consisting of a training set of 60,000 examples and a test set of 10,000 examples. Each example is a 28x28 grayscale image, associated with a label from 10 classes.
![image](https://user-images.githubusercontent.com/56757171/206940063-23fcbb12-8245-47c5-88a7-02a095f3d420.png)

### Class Labels
Each training and test example is assigned to one of the following labels:

| Label| Description  |    
| ---- |:------------:| 
| 0    | T-shirt/top  | 
| 1    | Trouser      | 
| 2    | Pullover     | 
| 3    | Dress        | 
| 4    | Coat         | 
| 5    | Sandal       |
| 6    | Shirt        | 
| 7    | Sneaker      | 
| 8    | Bag          |
| 9    | Ankle boot   | 

## Build Neural Network Using Pytorch 
For this program, I have included 8 functions (listed below) to transform data, build a neural network, train the network, evaluate its performance, and make preditions on test data.
  * **dataTransform():** Transform data from Fashion-MNIST dataset.
  * **getDataImageSamples():** Display dataset image samples
  <img width="1281" alt="getDataImageSamples" src="https://user-images.githubusercontent.com/56757171/207085495-73faba25-f59d-48ed-9095-75cde8612d34.png">
  
  * **getTrainingDataLoader():** Create a Dataloader object for training.
  * **getTestDataLoader():** Create a Dataloader object for test.
  * **buildModel():** Use LeakyReLU activation function to build a simple neural network model 
  <img width="658" alt="buildModel" src="https://user-images.githubusercontent.com/56757171/207124608-b6e0e67e-77bd-44f3-a544-193f35ed2f91.png">

  
  * **trainModel(model, train_loader, criterion, epoch):** A function to train the neural network, print the model's accuracy per epoch, and print the model's accumulated loss (epoch loss/length of the dataset) per epoch.
  * **evaluateModel(model, test_loader, criterion):** A function that prints the model's average loss (if the show_loss argument is true), and the model's accuracy on the testing data set.
  * **predictLabel(model, test_images, index):** A function that prints the top 3 most likely labels for the image at the given index, along with their probabilities.
  <img width="750" alt="PredictLabels" src="https://user-images.githubusercontent.com/56757171/208151557-bbb83c9d-9454-4495-b871-32c0a6bfadb6.png">

