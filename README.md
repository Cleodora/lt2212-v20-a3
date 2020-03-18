# LT2212 V20 Assignment 3

# Assignment 1

## Part 1
##### Command line Arguments for the file a3_features.py
- In the command line for a3_features.py the first thing is the percentage of test instances that you would like to split the dataset into. For example 20 or 25. It has to be an int. The default is 20. 
`--test 20`
- In the  first part we take the root directpry of the author directories as inpurt. In this case I have assumed that the root directory is a3. So i have used `glob.glob(args.inputdir + '/*/*/*')` which assumes that there are two further directories which in this case are `/enron_sample/AUTHOR_NAME/DOCUMENTS` so the inputfile path to the commmand line should be for example. 
 `/home/zeeshan/Assignment3/lt2212-v20-a3/a3`
- After the input path there should be be the output path and the name of the file for example
 `/home/zeeshan/Assignment3/lt2212-v20-a3/a3/testing`
 You dont have to mention the extension(like csv). It will automatically add the extension.
 - The last part of the command line is the number of dimension that you want to reduce the original features to like "50" or "60" or "100". It has to be an `int`.
 So the final command line input becomes
`python3 a3_features.py --test 20 '/home/zeeshan/Assignment3/lt2212-v20-a3/a3' 'outputfile' 60`
All the documents from each author directory is read one by one. The data/text is passed through the following cleaning process.
##### Cleaning process
- I have used string.punctuation to remove the punctuation as I found that if the punctutation is not removed the number of columns are increased and there are a lot of garbage words that dont make sense.
- All the words are converted into lower case to avoid duplicated values.
- The words are split on the basic of spaces.
- I have also removoed stop words from the text to lower the frequnecy of common words. The list was complied from the link https://gist.github.com/sebleier/554280
- For further lowering the amount of unique words, another check was used that ignored the words which had the length of less than 3. I found that out that it further resulted in the decrease of garbage words.
- All the nans were replaced with zero
##### Label encoding
- Scikit-learn label encoder was used to convert the name of the author to an integer value such that it is interpretable for the machine to understand the categories/authors. 
##### Dimensionality Reduction
- Scikit-learns SVD was used to reduce to dimensions to the umber of the dimensions from the command line argument.
##### Shuffle and Split
The Shuffle split function is used to randomly split the data into training and test set based on the value of the command line argment --test. There is a special variable called tag which is used as an identifier to see whether the instance belong to the training set or the test set. if the tag is 0 it belong to the training set else if its 1 it belongs to the testing set.
##### Writing to the output File
After the shuffle split the dataframe is written to a file given by the command line argumnet outputfile.

### Part 2

In the second part we read the file we created from the first part and design a basic perceptron. 
The command line argument for this part are
`python3 a3_model.py '/home/zeeshan/Assignment3/lt2212-v20-a3/outputfile.csv'`
The argument should be a path to a file that was created in the first part.
- The csv file is read into the pandas dataframe.
- The training and the testing dataset is separated via the tag column with tag = 0 for training and tah = 1 for testing. The labels column contains the integer value mapped to a authors name which represents the classes.
- The extra columns that were written to the csv file are removed which cant be used for the training the model.
- We separate the unique values from the classes which represent each other. As there are fourteen unique authors so there will be fourteen unique values. We return  a dictionary which contains all the indexes of each unique value that can be used in the training loop when we are randomly seleting two instances from either the same class or different class. This is down in the function `separateindexes` which is present in the `utility.py` file.
- Now we move on to the building the model. A simple perceptron with no hidden layer with sigmoid as the final layer. No activation is used. In terms of hyperparameters like learning rate and the loss function these are the options that were used. The optimizer used is ADAM and the learning rate is deafult 0.01 which can be changed in the `config.py`. Same is the case with the number of epochs. As this is a binary classification so i chose the loss BCE.
`optimizer = torch.optim.Adam(model.parameters(), lr = conf.parameters['LEARNING_RATE'])`
`criterion = nn.BCELoss()`
##### Training Loop
- We are gonna loop for 20 epochs which is the default value but can be changed in the `config.py` file.
- We are gonna traverse each class in the dictionary returned from the `separateindexes` which contains the indexes of all the classes. In the loop for each class, we have all the indexes of that  particular class(For example lets assume that we are traversing class 5(Author 5), we have all the indexes of that class) and we get the remaining instances of the rest of classes(exclusing the class 5 in this scenerio) from the `randnumber` function defined in the `utility.py`. Now for selecting either the instance from the same class or the different class. I select the following approach. I randomly generate two number either 0 or 1. If the randomly generated number is 0 it will generate one instanec from the indexes of the current class we are traversing and one instamnec randomly from the rest of the classes (excluding the class we are traversing). 0 is appended into the list which is basically the ground truth meaning they are not from the same classees/authors .If the randomly generated number is 1 it will get both the instances from the same class while making sure they are not the same documents. 1 is appended into the list which is basically the ground truth meaning they are from the same classees/authors. After that we will concat both the instances. The input is passed through the perceptron model that we have created with out any hidden layer or non linearity and the output from the model is appended to the list. After each author list is traversed the loss is calculated using the BCE loss that we have defined.
- The same process is repeated with the testing expect we dont train the model in this part rather we set the torch.no_grad() to make sure the gradient is not calculated and the model is not updated.
- The classification report is printed at the end with accuracy/ precision/ recall / f1-score.

#### Results with dimentionality reduction to 60 percent of the original features with no Hidden layer and non-linearity.
              precision    recall  f1-score   support

           0       0.56      0.57      0.57       320
           1       0.46      0.45      0.45       258

    accuracy                           0.52       578
    macro avg       0.51      0.51     0.51       578
    weighted avg    0.51      0.52     0.52       578

A perceptron with no hidden layer and sigmoid in the final layer is necessarily logistic regression. So in other words we are applying logistic regression to this dataset. It gives pretty bad results but thats a hard problem because as there are 14 classes and the model has to see if they are from the same class or from different class based on text. If it was a multi class problem with the 14 classes then it would have been a relatively simpler problem in my opinion. There is a possibility that the features that are true for two instances belonging to the same class are the same for the two instances bleonging to the different class making it difficult for the model to separate these results.

### Part 3
In the third part we have to introduce two linearities and a single hidden layer with different sizes. 
- The first argument is the name of the csv file we have used in the second part of this assignment. In this example its `'/home/zeeshan/Assignment3/lt2212-v20-a3/outputfile.csv'`
- The second command line parameter is `--hidden`. The default value is 0. We can change and try out different values like 60 in this example `--hidden 60` 
- The third part is non-linearity/activation function which you have a choice between "relu", "elu", "tanh" and "None". The default is None. In this case its "elu". It should be given as a string.
`python3 a3_model.py '/home/zeeshan/Assignment3/lt2212-v20-a3/outputfile.csv' --hidden 60 --activation 'elu'`
- The training and the testing loop used is the same as used in the part 2.

#### Results with dimentionality reduction to 60 percent of the original features with Hidden layer size 60 and relu activation.

              precision    recall  f1-score   support

           0       0.47      0.75      0.58       281
           1       0.47      0.21      0.29       295

    accuracy                           0.47       576
    macro avg       0.47      0.48     0.43       576
    weighted avg    0.47      0.47     0.43       576
    
With hidden layer 60 and relu the Feed forward neural performs worse than the perceptron. 

#### Results with dimentionality reduction to 60 percent of the original features with Hidden layer size 60 and elu activation.

              precision    recall  f1-score   support

           0       0.54      0.66      0.59       308
           1       0.47      0.36      0.41       267

    accuracy                           0.52       575
    macro avg       0.51      0.51     0.50       575
    weighted avg    0.51      0.52     0.51       575

It performs better than the relu activation with the same hidden size.

#### Results with dimentionality reduction to 60 percent of the original features with Hidden layer size 60 and tanh activation.

              precision    recall  f1-score   support

           0       0.49      0.63      0.55       285
           1       0.50      0.35      0.41       292

    accuracy                           0.49       577
    macro avg       0.49      0.49     0.48       577
    weighted avg    0.49      0.49     0.48       577
    
The tanh performs better than the relu activation but worse than the elu activation.

#### Results with dimentionality reduction to 60 percent of the original features with Hidden layer size 100 and elu activation.

              precision    recall  f1-score   support

           0       0.53      0.62      0.57       294
           1       0.52      0.43      0.47       283

    accuracy                           0.53       577
    macro avg       0.53      0.53     0.52       577
    weighted avg    0.53      0.53     0.52       577

The hidden layer with size 100 with activation elu performs better than hidden laer with 60 and the simple perceptron. It seems increaing the hidden layer size helped the model with the decision making.

#### Results with dimentionality reduction to 60 percent of the original features with Hidden layer size 100 and relu activation.

              precision    recall  f1-score   support

           0       0.51      0.70      0.59       308
           1       0.42      0.25      0.31       272

    accuracy                           0.49       580
    macro avg       0.47      0.47     0.45       580
    weighted avg    0.47      0.49     0.46       580

The same is case with the hidden layer size 100 and relu activation, Although it worked better than previous setting of hidden layer with the size 60 and activation relu but not better than the perceptron or the elu activation.

#### Results with dimentionality reduction to 60 percent of the original features with Hidden layer size 100 and tanh activation.

              precision    recall  f1-score   support

           0       0.52      0.67      0.58       294
           1       0.51      0.36      0.42       288

    accuracy                           0.52       582
    macro avg       0.52      0.51     0.50       582
    weighted avg    0.52      0.52     0.50       582

The same is case with the hidden layer size 100 and tanh activation, Although it worked better than previous setting of hidden layer with the size 60 and activation tanh but not better than the perceptron or the elu activation.

#### Results with dimentionality reduction to 60 percent of the original features with Hidden layer size 200 and elu activation.

              precision    recall  f1-score   support

           0       0.53      0.64      0.58       307
           1       0.47      0.36      0.41       272

    accuracy                           0.51       579
    macro avg       0.50      0.50     0.49       579
    weighted avg    0.50      0.51     0.50       579
    
As the hidden layer size is increased the model starts to overfit and we can see the accuracy and the other results starts to decrease/stay the same.

#### Results with dimentionality reduction to 60 percent of the original features with Hidden layer size 200 and relu activation.

              precision    recall  f1-score   support

           0       0.49      0.71      0.58       293
           1       0.47      0.26      0.33       288

    accuracy                           0.49       581
    macro avg       0.48      0.49     0.46       581
    weighted avg    0.48      0.49     0.46       581
    
We can see the same pattern appearing with the increasing hidden size and the ccuracy is decreasing.

#### Results with dimentionality reduction to 60 percent of the original features with Hidden layer size 200 and tanh activation.

              precision    recall  f1-score   support

           0       0.55      0.73      0.63       317
           1       0.47      0.29      0.36       263

    accuracy                           0.53       580
    macro avg       0.51      0.51     0.49       580
    weighted avg    0.51      0.53     0.50       580
    

The accuracy increased with increasing the hidden layer size but the individual accuracy decreased for class 0.

#### Results with dimentionality reduction to 60 percent of the original features with Hidden layer size 300 and elu activation.



              precision    recall  f1-score   support

           0       0.49      0.79      0.61       279
           1       0.55      0.24      0.34       298

    accuracy                           0.51       577
    macro avg       0.52      0.52     0.47       577
    weighted avg    0.53      0.51     0.47       577

As the hidden layer size increases the model starts to favour the class 0 more and the f1 score for the class 1 increases but the class 1 decreases so the combined accurcy stays the same or decreases.


#### Results with dimentionality reduction to 60 percent of the original features with Hidden layer size 300 and relu activation.

              precision    recall  f1-score   support

           0       0.50      0.83      0.63       286
           1       0.55      0.20      0.29       295

    accuracy                           0.51       581
    macro avg       0.53      0.52     0.46       581
    weighted avg    0.53      0.51     0.46       581

We can see the same pattern in the relu activation with the model favuring the class 0 and the accuray of class1 decreasing


#### Results with dimentionality reduction to 60 percent of the original features with Hidden layer size 300 and tanh activation.

              precision    recall  f1-score   support

           0       0.51      0.71      0.60       293
           1       0.52      0.32      0.40       288

    accuracy                           0.52       581
    macro avg       0.52      0.51     0.50       581
    weighted avg    0.52      0.52     0.50       581
    
The tanh performed better but the model is still favouring the class 0. 

After trying out different activations with different hidden sizes. It seems that the FFNN model starts to get better with the increasing hidden size layer but eventually starts to overfit annd starts favouring class 0. The perceptron performs more or less the same with no hidden layers and activation.






