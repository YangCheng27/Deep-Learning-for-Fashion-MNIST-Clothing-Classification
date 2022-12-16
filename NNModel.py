# Use Pytorch and Fashion-MNIST dataset to build neural network

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from matplotlib.ticker import AutoMinorLocator, FormatStrFormatter

# class labels
class_names = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt'
                   ,'Sneaker','Bag','Ankle Boot']

# transform data from Fashion-MNIST
def dataTransform():
    
    dataTransform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    return dataTransform
    

def getDataImageSamples():
    
    im = None
    
    data = datasets.FashionMNIST(
        './data',
        train=True,
        download=True,
        transform=dataTransform())
    
    images, label = next(iter(torch.utils.data.DataLoader(
                                data,
                                28, shuffle = True)
                             )
                        )
    images = images.reshape(28, 28, 28)
    fig,axes = plt.subplots(2, 14, figsize = (15, 2))  
    axes = axes.flatten()

    
    for i, (ax, image) in enumerate(zip(axes, images)):

        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        
        if torch.is_tensor(image):
            ax.imshow(image.numpy())
        else:
            im = ax.imshow(image)
            
            im
            
    
    print('Dataset FashionMNIST Image Samples:')
    plt.show()
    
    print('Each training and test sample is assigned to one of the ten following labels:')
    print()
    for label in class_names:
        print(label)
    
    print()


    
# Create training data set
def getTrainingDataLoader():

    train_set=datasets.FashionMNIST(
        './data',
        train=True,
        download=True,
        transform=dataTransform())

    return torch.utils.data.DataLoader(train_set, batch_size = 16)

# Create test data set
def getTestDataLoader():
    
    test_set=datasets.FashionMNIST(
        './data',
        train=False,
        transform=dataTransform())
        
    return torch.utils.data.DataLoader(test_set, batch_size = 16)


# build the neural network model
def buildModel():
    
    # Each example is a 28x28 grayscale image, associated with a label from 10 classes.
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 512),
        
        nn.LeakyReLU(0.03), # ELU(), LeakyReLU(0.1), ReLU() 
        
        nn.Linear(512, 256),
        
        nn.LeakyReLU(0.03), #ELU(),
        
        nn.Linear(256, 10),
    )
    
    return model



# Train the neural network model
def trainModel(model, train_loader, criterion, epoch):

    model.train()
    opt = optim.SGD(model.parameters(), lr = 0.001, momentum = 0.9)

    for epoch in range(epoch):

        running_loss = 0.0
        correct = 0
        total = 0

        for i, data in enumerate(train_loader, 0):
            inputs, labels = data

            # zero the parameter gradients
            opt.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            opt.step()

            # print statistics
            running_loss += loss.item()
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Train Accuracy ('+ str(epoch) + '):',#str(correct)+'/'+str(total)+
              f'{100*correct/total:.2f}%',
              f' Loss: {running_loss*0.001:.3f}')
    


"""
    INPUT: 
        model - the the trained model produced by the previous function
        test_loader    - the test DataLoader
        criterion   - cropy-entropy 

    RETURNS:
        None
"""
def evaluateModel(model, test_loader, criterion):
    
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, labels in test_loader:
            outputs = model(data)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * 16
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    loss = running_loss / total
    accuracy = 100 * correct / total
    
    print('Test Accuracy :', #str(correct) + '/' + str(total) + 
          f'{accuracy:.2f}%',
          f' Loss: {loss:.4f}')


"""

    INPUT: 
        model - the trained model
        test_images   -  test image set of shape Nx1x28x28
        index   -  specific index  i of the image to be tested: 0 <= i <= N - 1


    RETURNS:
        None
"""
    
def predictLabel(model, test_images, index):
    
    logits = model(test_images[index])
    prob = F.softmax(logits, dim=1)*100
    top3_prob, top3_class = torch.topk(prob, 3)
    
    images = test_images[index]
    fig,ax = plt.subplots(figsize = (1, 1))  

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    
    if torch.is_tensor(images):
        ax.imshow(images[0,:,:].numpy())
    else:
        ax.imshow(images[0,:,:])
    print()
    print()
    print('Test Sample', index, ':')
    plt.show()
    
    print('Top three most likely class labels')
    for i in range(3):
        c = top3_class[0][i].item()
        p = top3_prob[0][i].item()
        
        print(class_names[c] + ":", f'{p:.2f}%')



if __name__ == '__main__':

    
    criterion = nn.CrossEntropyLoss()
    train_loader = getTrainingDataLoader()
    print(type(train_loader))
    print(train_loader.dataset)
    
    print()
    getDataImageSamples()
    
    test_loader = getTestDataLoader()  
    model = buildModel()
    
    print('\n-------------<Build Nueral Network Model>-------------\n')
    print(model)
    
    print('\n-------------<Train Neural Network Model>-------------\n')
    trainModel(model, train_loader, criterion, epoch = 10)
    
    print('\n---------------<Evaluate Newwork Model>---------------\n')

    evaluateModel(model, test_loader, criterion)
    print('\n-----------------<Predict the Labels>-----------------\n')
    pred_set, _ = next(iter(test_loader))
    predictLabel(model, pred_set, 1)
    predictLabel(model, pred_set, 2)
    predictLabel(model, pred_set, 3)
    predictLabel(model, pred_set, 4)
    predictLabel(model, pred_set, 5)
    predictLabel(model, pred_set, 11)
    predictLabel(model, pred_set, 12)
    predictLabel(model, pred_set, 13)
    predictLabel(model, pred_set, 14)
    predictLabel(model, pred_set, 15)