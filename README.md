# Gradient-Descent
- For training, I imported the training data set and converted all the -1 to 0 to accomodate for the version of cross entropy error formula I was using.
- The gradDFit function is basically like the .fit method of sklearn's logistic regression model.
- Gradient Descent is part of the same algorithm mentioned above, prints the cost and classification error.
- The predict method takes in a vector X and predicts the target variable -1 or 1.
- Parameters of the main Logistic Regression class include learning rate and max iterations so you can change that as desired when calling the class.
- For the Cleveland test file, I imported the data set, used a scaler for standardization, and used my model to predict the target variable.
- The X vector of the test dataset needs to transposed which my code does. 
- In the end, I created an empty .csv file and wrote my answers to the file. Note: I had to convert my predict to an array using np.asarray() to make it work for  writing to the file.
