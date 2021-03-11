import numpy as np
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.image as implt
from PIL import Image
import warnings
import tensorflow as tf

#Define the amount of pictures
TOTAL_WASP_TRAIN = 2127
TOTAL_BEE_TRAIN = 2469
TOTAL_WASP_TEST = 2816
TOTAL_BEE_TEST = 714

# filter warnings
warnings.filterwarnings('ignore')


#Sort the pics folders
train_wasp = sorted(os.listdir('wasp1'))
train_bee =  sorted(os.listdir('bee1'))
test_wasp = sorted(os.listdir('wasp2'))
test_bee =  sorted(os.listdir('bee2'))

list=["train_wasp","train_bee","test_wasp","test_bee"]

#Start counting the pics
wasp1=0
wasp2=0
bee1=0
bee2=0

for i in train_wasp:
    wasp1=wasp1+1
    
for i in train_bee:
    bee1=bee1+1
   
for i in test_wasp:
    wasp2=wasp2+1
    
for i in test_bee:
    bee2=bee2+1

count_wasp =wasp1+wasp2
count_bee=bee1+bee2

list_count=[count_wasp,count_bee]
list_count_veriable =["wasp","bee"]
list=["train_wasp","train_bee","test_wasp","test_bee"]

 
list1=["train_wasp","test_wasp"]
list2=["train_bee","test_bee"]
 
 #create the actual images lists, and create the Y label - 1 for a wasp and 1 for a bee.
 #This part for training
list_img_wasp=[]
x_list_wasp=[]
list_img_bee=[]
x_list_bee=[]

for x in train_wasp:
    x_list_wasp.append(x)
for i in range(1,2127):
    list_img_wasp.append(implt.imread('wasp1/'+x_list_wasp[i]))

for y in train_bee:
    x_list_bee.append(y)
for a in range(1,2469):
    list_img_bee.append(implt.imread('bee1/'+x_list_bee[a]))

img_size = 50
wasp_insect_train = []
bee_insect_train = [] 
label_train = []

for i in train_wasp:
    if os.path.isfile('wasp1/'+ i):
        insect = Image.open('wasp1/'+ i).convert('L') #converting grey scale            
        insect = insect.resize((180,300), Image.ANTIALIAS) #resizing to 50,50
        insect = np.asarray(insect)/255.0 #normalizing images
        wasp_insect_train.append(insect)  
        label_train.append(1)

for i in train_bee:
    if os.path.isfile('bee1/'+ i):
        insect = Image.open('bee1/'+ i).convert('L')
        insect = insect.resize((180,300), Image.ANTIALIAS)
        insect = np.asarray(insect)/255.0 #normalizing images
        bee_insect_train.append(insect)  
        label_train.append(0)


#Connect the wasp train images and the bees train images to 1.
x_train = np.concatenate((wasp_insect_train,bee_insect_train),axis=0) # training dataset
x_train_label = np.asarray(label_train)# label array containing 0 and 1
x_train_label = x_train_label.reshape(x_train_label.shape[0],1)

 #This part for testing
wasp_insect_test = []
bee_insect_test = [] 
label_test = []

for i in test_wasp:
    if os.path.isfile('wasp2/'+ i):
        insect = Image.open('wasp2/'+ i).convert('L')  #prepare the image, turn it to grey scale          
        insect = insect.resize((180,300), Image.ANTIALIAS) #Resize to 180X300 pixels
        insect = np.asarray(insect)/255.0 #Normalize the pic
        wasp_insect_test.append(insect)  
        label_test.append(1)      #add to the y_label

for i in test_bee:
    if os.path.isfile('bee2/'+ i):
        insect = Image.open('bee2/'+ i).convert('L') #prepare the image, turn it to grey scale  
        insect = insect.resize((180,300), Image.ANTIALIAS) #Resize to 180X300 pixels
        insect = np.asarray(insect)/255.0    #Normalize the pic 
        bee_insect_test.append(insect) 
        label_test.append(0)    #add to the y_label


#Connect the wasp train and the bee train
x_test = np.concatenate((wasp_insect_test,bee_insect_test),axis=0) # test dataset
x_test_label = np.asarray(label_test) # corresponding labels
x_test_label = x_test_label.reshape(x_test_label.shape[0],1)

x_train = np.asarray(x_train)
x_test = np.asarray(x_test)

#Reshape the train part to 2 Dimensions
x_train = x_train.reshape(x_train.shape[0],x_train.shape[1]*x_train.shape[2]) #flatten 3D image array to 2D


#Creating the Neural Network
#With 3 layers of 200,100,50 neurons
(hidden1_size, hidden2_size,hidden3_size) = (200,100,50)
features = 54000 
eps = 1e-12
x = tf.compat.v1.placeholder(tf.float32, [None, features])
y_ = tf.compat.v1.placeholder(tf.float32, [None, 1])
W1 = tf.Variable(tf.random.truncated_normal([features, hidden1_size], stddev=0.1))
b1 = tf.Variable(tf.constant(0.1, shape=[hidden1_size]))
z1 = tf.nn.relu(tf.matmul(x, W1)+b1)
W2 = tf.Variable(tf.random.truncated_normal([hidden1_size, hidden2_size], stddev=0.1))
b2 = tf.Variable(tf.constant(0.1, shape=[hidden2_size]))
z2 = tf.nn.relu(tf.matmul(z1,W2)+b2)
W3 = tf.Variable(tf.random.truncated_normal([hidden2_size, 1], stddev=0.1))
b3 = tf.Variable(tf.constant(0.1, shape=[hidden3_size]))
z3 = tf.matmul(z2, W3) + b3
W4 = tf.Variable(tf.random.truncated_normal([hidden3_size, 1], stddev=0.1))
b4 = tf.Variable(0.)
z4 =tf.matmul(z3, W4) + b4

y1 = 1 / (1.0 + tf.exp(-z4))

loss = -(y_ * tf.math.log(y1 + eps) + (1 - y_) * tf.math.log(1 - y1 + eps))
cross_entropy  = tf.reduce_mean(loss)
train_step  = tf.compat.v1.train.GradientDescentOptimizer(0.00001).minimize(cross_entropy )
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
for i in range(0, 100):
    sess.run(train_step , feed_dict={x: x_train, y_: x_train_label})



##After the NN is created and trained print the train process
#Bee train print
bee_insect_train= np.asarray(bee_insect_train)
bee_insect_train = bee_insect_train.reshape(bee_insect_train.shape[0],bee_insect_train.shape[1]*bee_insect_train.shape[2]) #flatten 3D image array to 2D
Bee_prediction = np.average(y1.eval(session=sess, feed_dict = {x :bee_insect_train}))
print("Prediction train BEE image: ", Bee_prediction)
train_error_bee = 1 - Bee_prediction

#Wasp train print
wasp_insect_train= np.asarray(wasp_insect_train)
wasp_insect_train = wasp_insect_train.reshape(wasp_insect_train.shape[0],wasp_insect_train.shape[1]*wasp_insect_train.shape[2]) #flatten 3D image array to 2D
Wasp_prediction = np.average(y1.eval(session=sess, feed_dict = {x :wasp_insect_train}))
print("Prediction train WASP image: ", Wasp_prediction)
train_error_wasp = Wasp_prediction

#Total train ERROR print
total_train_error = (train_error_bee + train_error_wasp) / 2.
print("Train error: ", total_train_error)

##Finally test it on the test part of the bee and wasp pictures and print results.
wasp_insect_test= np.asarray(wasp_insect_test)
wasp_insect_test = wasp_insect_test.reshape(wasp_insect_test.shape[0],wasp_insect_test.shape[1]*wasp_insect_test.shape[2]) #flatten 3D image array to 2D
bee_insect_test= np.asarray(bee_insect_test)
bee_insect_test = bee_insect_test.reshape(bee_insect_test.shape[0],bee_insect_test.shape[1]*bee_insect_test.shape[2]) #flatten 3D image array to 2D

#Prepare for check and start testing
(classify_waspRight, classify_beeRight) = (0,0)
wasp_predictions_test = y1.eval(session=sess, feed_dict= {x :wasp_insect_test})
for waspPrediction in wasp_predictions_test:
        # There is a 0.5 classifibeeion threshold
    if (waspPrediction > 0.5): # wasp classify predicted if probability > 0.5
        classify_waspRight += 1
   
wasp_Prediction_mean = np.average(wasp_predictions_test)
print("Prediction wasp Test: ", wasp_Prediction_mean)

bee_predictions_test = y1.eval(session=sess, feed_dict= {x :bee_insect_test})
for beePrediction in bee_predictions_test:
    if (beePrediction < 0.5): # bee classify predicted if probability < 0.5
        classify_beeRight += 1

bee_Prediction_mean = np.average(bee_predictions_test)
print("Prediction bee Test: ", bee_Prediction_mean)

#Print finall result with accuracy
accuracy = (classify_beeRight + classify_waspRight) / (TOTAL_BEE_TEST + TOTAL_WASP_TEST) # how often the classify is correct
test_error = (1 - wasp_Prediction_mean + bee_Prediction_mean) / 2. # how often the classify is incorrect
print("Test Error: %.4f" % test_error)
print("Accuracy: %.4f" % accuracy)
