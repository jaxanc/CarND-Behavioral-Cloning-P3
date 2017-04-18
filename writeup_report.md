# Behavioral Cloning Project Writeup

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/Model_Architecture.png "Model Architecture"
[image2]: ./images/center_2017_04_17_21_59_49_062.jpg "Center Image Example"
[image3]: ./images/center_2017_04_17_22_01_30_023.jpg "Recovery Image"
[image4]: ./images/center_2017_04_17_22_01_07_504.jpg "Recovery Image"
[image5]: ./images/center_2017_04_17_22_02_09_240.jpg "Recovery Image"

## Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.

---
## Files Submitted & Code Quality

### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.ipynb containing the script to create and train the model
* model.py which is generated from the model.ipynb for the purpose of submission
* model.html which is generated from the model.ipynb
* model_vgg16.ipynb containing the script to create and train the model using transfer learning (VGG16 model)
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md summarizing the results
* video.mp4 is the recording of driving on track 1

### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

### 3. Submission code is usable and readable

The model.ipynb file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works. A separate model.py file is created to meet the submission requirement.

## Model Architecture and Training Strategy

### 1. An appropriate model architecture has been employed

My model is mainly my intepretation of the [nVidia model](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) (model.py lines 181-204). It has some differences such as there are 4 fully connected layers instead of 3 as documented in the publication. The last connected layer could perhaps be an averaging or a max pooling layer.

The model includes ELU layers to introduce nonlinearity and the data is normalized in the model using a Keras lambda layer (code line 184).

Both ReLu and ELU layers have both been experimented with no noticable difference between them.

### 2. Attempts to reduce overfitting in the model

Dropouts, regularization, augmentation and maxpooling have been introduced during development. In the end the dropout layers between the 4 fully connected layers were implemented to produce consistent result.

The transfer learning approach presented in model_vgg16.ipynb also contain dropout layers between the fully connected layers. They were used instead of the augmentation methods as they seem to produce worse results in this approach.

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 73). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 233).

### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road.

For details about how I created the training data, see the next section.

## Model Architecture and Training Strategy

### 1. Solution Design Approach

The overall strategy for deriving a model architecture was examine different proven techniques and fine tune the parameters for this project.

My first step was to use a convolution neural network model similar to the [nVidia model](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf). I thought this model might be appropriate because it has proven to work.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting.

To combat the overfitting, I modified the model so that there are max pooling layers between the Convolution2D layers similar to the VGG16 architecture. I have also added dropout layers between the 4 fully connected layers.

Then I experimented with image augmentation similar to what was described in the nVidia paper.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track (particulary before and after the bridge). To improve the driving behavior in these cases, I decided to flip images horizontally at random to simulate even amount of left and right turns.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

After the success of track 1 driving, I moved on to experienting with trasfer learning using the VGG16 model. This model is chosen mainly due to my limited processing power. The images were scaled to 64-by-64 so python would not crash and only 32 samples were used per generator.

In the end the VGG16 model with 4 additional fully connected layer was able to drive around track 2 most of the time but the training time was so long it really limited the amount of adjustments I could make.

So I tried another approach. By training the model I have created using track 1 data again with track 2 data (model.py line 212-213). I was able to update the model to be able to drive on track 2. However, I have trained this way a few times but does not always produce a good result. Even though everything is kept the same.

### 2. Final Model Architecture

The final model architecture (model.py lines 181-204) consisted of a convolution neural network with the following layers and layer sizes

![alt text][image1]

### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover when getting to the outter edges of the road. These images show what a recovery looks like starting from left side of the road:

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data set, I also flipped images and angles at random thinking that this would generate even amount of left turn and right turn data.

After the collection process, I had 6507 number of data points. I then preprocessed this data by removing 50 pixels from the top of the images which are not related to the road and 20 pixels of the bottom which contains front of the car. The cropped images were then scaled to 200-by-66 to match the input shape of the nVidia model.

I finally randomly shuffled the data set and put 20% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was between 6 to 10 as evidenced by using the early stop callbacks. This can be viewed from the last cell in the model.ipynb or model.html.

I used an adam optimizer so that manually training the learning rate wasn't necessary.
