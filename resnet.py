import tensorflow as tf
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Dense,Flatten

BATCH = 64
NUM_EPOCHS = 5

# create a new generator
imagegen = ImageDataGenerator(rescale=1./255)
# load train data
train = imagegen.flow_from_directory("imagenette2/train/", class_mode="categorical", shuffle=True, batch_size = BATCH, target_size=(224, 224))
# load val data
val = imagegen.flow_from_directory("imagenette2/val/", class_mode="categorical", shuffle=True, batch_size = BATCH, target_size=(224, 224))
print(len(train[0]), train[0][0].shape, train[0][1].shape, train[0][0][0].shape)
model1 = tf.keras.applications.resnet.ResNet50(weights='imagenet', input_tensor=None,input_shape=train[0][0][0].shape, include_top=False)
flatten = Flatten()
print(train[0][1].shape[1])
new_layer1 = Dense(train[0][1].shape[1], activation='softmax', name='transfer_lr')
input1 = model1.input
output1 = new_layer1(flatten(model1.output))
model = Model(input1, output1)
model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train, epochs=NUM_EPOCHS, validation_data = val)


t1 = time.time()
model.evaluate(val)
print("the inference takes {} seconds".format(time.time() - t1))
