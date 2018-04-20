'''
Using Bottleneck Features for Multi-Class Classification in Keras
We use this technique to build powerful (high accuracy without overfitting) Image Classification systems with small
amount of training data.
The full tutorial to get this code working can be found at the "Codes of Interest" Blog at the following link,
http://www.codesofinterest.com/2017/08/bottleneck-features-multi-class-classification-keras.html
Please go through the tutorial before attempting to run this code, as it explains how to setup your training data.
The code was tested on Python 3.5, with the following library versions,
Keras 2.0.6
TensorFlow 1.2.1
OpenCV 3.2.0
This should work with Theano as well, but untested.
'''
import numpy as np
import keras
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt
import math
import cv2
import threading
from PIL import Image
import os

# dimensions of our images.
img_width, img_height = 320, 240

top_model_weights_path = 'bottleneck_fc_model.h5'
train_data_dir = 'data/train'
validation_data_dir = 'data/validation'

# number of epochs to train top model
epochs = 100
# batch size used by flow_from_directory and predict_generator
batch_size = 16


def save_bottlebeck_features():
    # build the VGG16 network
    model = applications.VGG16(include_top=False, weights='imagenet')

    datagen = ImageDataGenerator(rescale=1. / 255)

    generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)

    print(len(generator.filenames))
    print(generator.class_indices)
    print(len(generator.class_indices))

    nb_train_samples = len(generator.filenames)
    num_classes = len(generator.class_indices)

    predict_size_train = int(math.ceil(nb_train_samples / batch_size))

    bottleneck_features_train = model.predict_generator(
        generator, predict_size_train)

    np.save('bottleneck_features_train.npy', bottleneck_features_train)

    generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)

    nb_validation_samples = len(generator.filenames)

    predict_size_validation = int(
        math.ceil(nb_validation_samples / batch_size))

    bottleneck_features_validation = model.predict_generator(
        generator, predict_size_validation)

    np.save('bottleneck_features_validation.npy',
            bottleneck_features_validation)


def train_top_model():
    datagen_top = ImageDataGenerator(rescale=1. / 255)
    generator_top = datagen_top.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False)

    nb_train_samples = len(generator_top.filenames)
    num_classes = len(generator_top.class_indices)

    # save the class indices to use use later in predictions
    np.save('class_indices.npy', generator_top.class_indices)

    # load the bottleneck features saved earlier
    train_data = np.load('bottleneck_features_train.npy')

    # get the class lebels for the training data, in the original order
    train_labels = generator_top.classes

    # https://github.com/fchollet/keras/issues/3467
    # convert the training labels to categorical vectors
    train_labels = to_categorical(train_labels, num_classes=num_classes)

    generator_top = datagen_top.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)

    nb_validation_samples = len(generator_top.filenames)

    validation_data = np.load('bottleneck_features_validation.npy')

    validation_labels = generator_top.classes
    validation_labels = to_categorical(
        validation_labels, num_classes=num_classes)

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    opt = keras.optimizers.SGD(
        lr=0.0001, momentum=0.9, decay=0.0, nesterov=False)

    model.compile(optimizer=opt, loss='categorical_crossentropy',
                  metrics=['accuracy'])
    #model.compile(optimizer='rmsprop',loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(train_data, train_labels,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(validation_data, validation_labels))

    model.save_weights(top_model_weights_path)

    (eval_loss, eval_accuracy) = model.evaluate(
        validation_data, validation_labels, batch_size=batch_size, verbose=1)

    print("[INFO] accuracy: {:.2f}%".format(eval_accuracy * 100))
    print("[INFO] Loss: {}".format(eval_loss))

    plt.figure(1)

    # summarize history for accuracy

    plt.subplot(211)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    # summarize history for loss

    plt.subplot(212)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

#save_bottlebeck_features()
#train_top_model()


# load the class_indices saved in the earlier step
class_dictionary = np.load('class_indices.npy').item()
num_classes = len(class_dictionary)
frame_small = None
label = ""
classify = True

def predict():
    global frame_small
    global label
    path = "test.jpg"
    cv2.namedWindow("preview")
    vc = cv2.VideoCapture(0)
    if vc.isOpened() :
        rval, frame = vc.read()
    else :
        rval =False
    #every alternate frame
    frame_count = 2
    label = ""
    while rval :
        rval, frame = vc.read()
        key = cv2.waitKey(20)
        if key == 27 :
            break
        frame_small = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
        if frame_count == 2 :
            #label = runtime(frame_small,num_classes,class_dictionary)
            # get the prediction label
            #print("Label: {}".format(label))
            # display the predictions with the image
            cv2.putText(frame_small, "Predicted: {}".format(label), (10, 30),
                        cv2.FONT_HERSHEY_PLAIN, 1.5, (43, 99, 255), 2)
            cv2.imshow("preview", frame_small)
            frame_count = 1
        else :
            frame_count+=1
            # get the prediction label
            #print("Label: {}".format(label))
            # display the predictions with the image
            cv2.putText(frame_small, "Predicted: {}".format(label), (10, 30),
                        cv2.FONT_HERSHEY_PLAIN, 1.5, (43, 99, 255), 2)
            cv2.imshow("preview", frame_small)

            
class Classification_Thread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
    def run(self):
        global frame_small
        global num_classes
        global class_dictionary
        global classify
        global label

        while classify is True :
            if np.all(frame_small) != None :
                orig = frame_small
                frame_small = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame_small)
                image = img_to_array(image)
                
                #cv2.imwrite("test.jpg",frame_small)
                #orig = cv2.imread(path)
                #print("[INFO] loading and preprocessing image...")
                #image = load_img(path, target_size=(img_width, img_height))
                #image = img_to_array(image)

                # important! otherwise the predictions will be '0'
                image = image / 255
                image = np.expand_dims(image, axis=0)

                # build the VGG16 network
                model = applications.VGG16(include_top=False, weights='imagenet')

                # get the bottleneck prediction from the pre-trained VGG16 model
                bottleneck_prediction = model.predict(image)

                # build top model
                model = Sequential()
                model.add(Flatten(input_shape=bottleneck_prediction.shape[1:]))
                model.add(Dense(256, activation='relu'))
                model.add(Dropout(0.5))
                model.add(Dense(num_classes, activation='softmax'))

                model.load_weights(top_model_weights_path)

                # use the bottleneck prediction on the top model to get the final
                # classification
                class_predicted = model.predict_classes(bottleneck_prediction)

                probabilities = model.predict_proba(bottleneck_prediction)
                
                print (probabilities)
                print (class_predicted)
                inID = class_predicted[0]

                inv_map = {v: k for k, v in class_dictionary.items()}

                if probabilities[0][0] <= 0.80 and probabilities[0][1] <= 0.80  :
                    label = "No Gesture"
                else :
                    label = inv_map[inID]
classify_thread = Classification_Thread()
classify_thread.start()

def runtime(frame_small,num_classes,class_dictionary) :
    orig = frame_small
    frame_small = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame_small)
    image = img_to_array(image)
    
    #cv2.imwrite("test.jpg",frame_small)
    #orig = cv2.imread(path)
    #print("[INFO] loading and preprocessing image...")
    #image = load_img(path, target_size=(img_width, img_height))
    #image = img_to_array(image)

    # important! otherwise the predictions will be '0'
    image = image / 255
    image = np.expand_dims(image, axis=0)

    # build the VGG16 network
    model = applications.VGG16(include_top=False, weights='imagenet')

    # get the bottleneck prediction from the pre-trained VGG16 model
    bottleneck_prediction = model.predict(image)

    # build top model
    model = Sequential()
    model.add(Flatten(input_shape=bottleneck_prediction.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.load_weights(top_model_weights_path)

    # use the bottleneck prediction on the top model to get the final
    # classification
    class_predicted = model.predict_classes(bottleneck_prediction)

    probabilities = model.predict_proba(bottleneck_prediction)
    print (probabilities)
    print (class_predicted)
    inID = class_predicted[0]

    inv_map = {v: k for k, v in class_dictionary.items()}

    label = inv_map[inID]

    return label
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()




predict()

cv2.destroyAllWindows()

'''
def predict():
    # load the class_indices saved in the earlier step
    class_dictionary = np.load('class_indices.npy').item()
    num_classes = len(class_dictionary)
    path = "test.jpg"
    cv2.namedWindow("preview")
    vc = cv2.VideoCapture(0)
    if vc.isOpened() :
        rval, frame = vc.read()
    else :
        rval =False
    #every alternate frame
    frame_count = 54
    label = ""
    while rval :
        rval, frame = vc.read()
        key = cv2.waitKey(20)
        if key == 27 :
            break
        frame_small = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
        if frame_count == 54 :
            label = runtime(frame_small,num_classes,class_dictionary)
            # get the prediction label
            print("Label: {}".format(label))
            # display the predictions with the image
            cv2.putText(frame_small, "Predicted: {}".format(label), (10, 30),
                        cv2.FONT_HERSHEY_PLAIN, 1.5, (43, 99, 255), 2)
            cv2.imshow("preview", frame_small)
            frame_count = 1
        else :
            frame_count+=1
            # get the prediction label
            #print("Label: {}".format(label))
            # display the predictions with the image
            cv2.putText(frame_small, "Predicted: {}".format(label), (10, 30),
                        cv2.FONT_HERSHEY_PLAIN, 1.5, (43, 99, 255), 2)
            cv2.imshow("preview", frame_small)
    

    image_path = ['rtest.jpg']
    for path in image_path :
        orig = cv2.imread(path)

        print("[INFO] loading and preprocessing image...")
        image = load_img(path, target_size=(img_width, img_height))
        image = img_to_array(image)

        # important! otherwise the predictions will be '0'
        image = image / 255

        image = np.expand_dims(image, axis=0)

        # build the VGG16 network
        model = applications.VGG16(include_top=False, weights='imagenet')

        # get the bottleneck prediction from the pre-trained VGG16 model
        bottleneck_prediction = model.predict(image)

        # build top model
        model = Sequential()
        model.add(Flatten(input_shape=bottleneck_prediction.shape[1:]))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation='softmax'))

        model.load_weights(top_model_weights_path)

        # use the bottleneck prediction on the top model to get the final
        # classification
        class_predicted = model.predict_classes(bottleneck_prediction)

        probabilities = model.predict_proba(bottleneck_prediction)

        inID = class_predicted[0]

        inv_map = {v: k for k, v in class_dictionary.items()}

        label = inv_map[inID]

        # get the prediction label
        print("Image ID: {}, Label: {}".format(inID, label))

        # display the predictions with the image
        cv2.putText(orig, "Predicted: {}".format(label), (10, 30),
                    cv2.FONT_HERSHEY_PLAIN, 1.5, (43, 99, 255), 2)

        cv2.imshow("Classification"+path, orig)
        cv2.waitKey(0)
        #cv2.destroyAllWindows()

count_one = 0
count_two = 0
count_three = 0
miss_class = []
def predict():
    global count_one
    global count_two
    global count_three
    global miss_class
    # load the class_indices saved in the earlier step
    class_dictionary = np.load('class_indices.npy').item()
    num_classes = len(class_dictionary)
    image_path = []
    directory = "data\\validation\\gesture_three"
    for filename in os.listdir(directory):
        if filename.endswith(".jpg"):
            image_path.append("data\\validation\\gesture_three\\" + filename)

    #image_path = ['rtest.jpg']
    for path in image_path:
        orig = cv2.imread(path)

        print("[INFO] loading and preprocessing image...")
        image = load_img(path, target_size=(img_width, img_height))
        image = img_to_array(image)

        # important! otherwise the predictions will be '0'
        image = image / 255

        image = np.expand_dims(image, axis=0)

        # build the VGG16 network
        model = applications.VGG16(include_top=False, weights='imagenet')

        # get the bottleneck prediction from the pre-trained VGG16 model
        bottleneck_prediction = model.predict(image)

        # build top model
        model = Sequential()
        model.add(Flatten(input_shape=bottleneck_prediction.shape[1:]))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation='softmax'))

        model.load_weights(top_model_weights_path)

        # use the bottleneck prediction on the top model to get the final
        # classification
        class_predicted = model.predict_classes(bottleneck_prediction)

        probabilities = model.predict_proba(bottleneck_prediction)

        inID = class_predicted[0]

        inv_map = {v: k for k, v in class_dictionary.items()}

        label = inv_map[inID]

        # get the prediction label
        #print("Image ID: {}, Label: {}".format(inID, label))

        if label == "gesture_one":
            miss_class.append((path,label))
            count_one+=1
        elif label == "gesture_two":
            miss_class.append((path,label))
            count_two+=1
        elif label == "gesture_three":
            count_three+=1
        # display the predictions with the image
        #cv2.putText(orig, "Predicted: {}".format(label), (10, 30),
        #            cv2.FONT_HERSHEY_PLAIN, 1.5, (43, 99, 255), 2)

        #cv2.imshow("Classification" + path, orig)
        #cv2.waitKey(0)
        # cv2.destroyAllWindows()

predict()
print (miss_class)
print (count_one, count_two,count_three)
'''
