# help from- https://vijayabhaskar96.medium.com/tutorial-on-keras-flow-from-dataframe-1fd4493d237c
import numpy as np
import pandas as pd 
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
import os
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

FAST_RUN = False
IMAGE_WIDTH= 50
IMAGE_HEIGHT= 50
IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS=3




filenames = os.listdir("train")
categories = []
for filename in filenames:
    category = filename.split(' ')[0]
    if category == 'wasp1':
        categories.append(1)
    else:
        categories.append(0)

# DataFrame created from lists. It is a two-dimensional data structure for analysis and processing data.
df = pd.DataFrame({
    'filename': filenames,
    'category': categories
})



# Input Layer: It represent input image data. It will reshape image into single diminsion array. Example your image is 64x64 = 4096, it will convert to (4096,1) array.
# Conv Layer: This layer will extract features from image.
# Pooling Layer: This layerreduce the spatial volume of input image after convolution.
# Fully Connected Layer: It connect the network from a layer to another layer
# Output Layer: It is the predicted values layer.

model = Sequential() #Sequential provides training and inference features on this model.
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25)) #Dropout is a technique where randomly selected neurons are ignored during training.
                         # They are “dropped-out” randomly. This means that their contribution to the activation 
                         # of downstream neurons is temporally removed on the forward pass and any weight updates are not
                         # applied to the neuron on the backward pass.

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten()) #from matrix to array
model.add(Dense(512, activation='relu')) # 512 in the hidden layer
model.add(Dense(100, activation='relu'))


model.add(Dense(2, activation='softmax')) #the output layer= 2 because we have bee and wasp classes 

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])



earlystop = EarlyStopping(patience=40) #Early stopping is a method that allows you to specify an arbitrary 
                                    # large number of training epochs and stop training once the model performance 
                                    # stops improving on a hold out validation dataset.

 #ReduceLROnPlateau = Reduce learning rate when a metric has stopped improving.                                   
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', 
                                            patience=2, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)
callbacks = [earlystop, learning_rate_reduction]

df["category"] = df["category"].replace({0: 'bee', 1: 'wasp'})

# train_df = df
train_df, validate_df = train_test_split(df, test_size=0.20, random_state=42)
train_df = train_df.reset_index(drop=True) #fixxing the indexs  
validate_df = validate_df.reset_index(drop=True)
total_train = train_df.shape[0]
total_validate = validate_df.shape[0]
batch_size= 100

#Image data augmentation is a technique that can be used to artificially
#  expand the size of a training dataset by creating modified versions of images in the dataset.
train_datagen = ImageDataGenerator(
    rotation_range=15,
    rescale=1./255,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
)

train_generator = train_datagen.flow_from_dataframe(
    train_df, 
    "train", #Path to the directory which contains all the images.
    x_col='filename',#The name of the column which contains the filenames of the images.
    y_col='category', #If class_mode is not “raw” or not “input” you should pass the name of the column which contains the class names.
    target_size=IMAGE_SIZE,
    class_mode='categorical',
    batch_size=batch_size
)
validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_dataframe(
    validate_df, 
    "train", 
    x_col='filename',
    y_col='category',
    target_size=IMAGE_SIZE,
    class_mode='categorical',
    batch_size=batch_size
)
example_df = train_df.sample(n=1).reset_index(drop=True)
example_generator = train_datagen.flow_from_dataframe(
    example_df, 
    "train", 
    x_col='filename',
    y_col='category',
    target_size=IMAGE_SIZE,
    class_mode='categorical'
)

epochs=3 if FAST_RUN else 150
history = model.fit_generator(
    train_generator, 
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=total_validate//batch_size,
    steps_per_epoch=total_train//batch_size,
    # callbacks=callbacks
)




test_filenames = os.listdir("test")
test_df = pd.DataFrame({
    'filename': test_filenames
})
nb_samples = test_df.shape[0]

test_gen = ImageDataGenerator(rescale=1./255)
test_generator = test_gen.flow_from_dataframe(
    test_df, 
    "test", 
    x_col='filename',
    y_col=None,
    class_mode=None,
    target_size=IMAGE_SIZE,
    batch_size=batch_size,
    shuffle=False
)

predict = model.predict_generator(test_generator, steps=np.ceil(nb_samples/batch_size)) #Predict the output
test_df['category'] = np.argmax(predict, axis=-1) #reset the test_generator before whenever you call the predict_generator.
label_map = dict((v,k) for k,v in train_generator.class_indices.items())
test_df['category'] = test_df['category'].replace(label_map)
test_df['category'] = test_df['category'].replace({ 'wasp': 1, 'bee': 0 })



