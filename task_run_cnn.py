import math

import tensorflow as tf
import os
import IPython.display as display
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
import cv2

# set gpu no for traning
from PIL.Image import Image

os.environ["CUDA_VISIBLE_DEVICES"] = '-1'


# Complete code at ...

# functin with ignoring the param
def main(_):
    save_path = ''
    model = CNN()  # declare model with cnn()

    x = model.getInput()  # get IMG, open tf record
    y = model.inference(x)  # give IMG, apply gusian filter

    saver = tf.train.Saver()  # save the model! so you dont have to train it every code run

    global_init = tf.global_variables_initializer()  # globale initialisierung
    local_init = tf.local_variables_initializer()  # local initialisierung

    with tf.compat.v1.Session() as sess:  # ts.Session is sees shortcut and safe tf variables until you close the session
        sess.run(global_init)  # evaluate the global variable
        sess.run(local_init)  # evaluate the local variable

        coord = tf.train.Coordinator()  # coordination for threads (multitreading)
        threads = tf.train.start_queue_runners(coord=coord)  # start threads and give a list with threads back
        tf.train.start_queue_runners(sess=sess)  # start threads and give a list with threads back

        for i in range(model.N_SAMPLES):  # model.N_Samples = 3
            x_val, y_val = sess.run([x, y])

            # Task 3:
            # - Apply Gausian filter to image of Homer
            # - Save image
            # - Push snapshot and image to GIT
            # ...

        # saver
        tf.train.save(sess, 'my_test_model')  # save the training from ki

        coord.request_stop()  # stop threads
        coord.join(threads)  # wait for the threads to terminate
        print('done')


class CNN():  # Model
    def __init__(self):  # initialise vaiables
        self.N_SAMPLES = 3
        self.DATASET = tf.data.TFRecordDataset(
            "C:\\Users\\Q447230\\Coden\\gitRepositorys\\Coding_Task\\record_test.tfrecord")
        self.IN_SHAPE = ...
        pass

    # read indiviual images
    def getInput(self):
        def _parse_image_function(serialized):
            # dict with data we expect in the tf.record file
            features = {
                'img': tf.io.FixedLenFeature([], tf.string),
                'x': tf.io.FixedLenFeature([], tf.int64),
                'y': tf.io.FixedLenFeature([], tf.int64),
                'z': tf.io.FixedLenFeature([], tf.int64),
            }

            # where does serialized comes from
            # parse the serialized data to get a dict with our data
            parsed_example = tf.io.parse_single_example(serialized=serialized, features=features)

            # what are raw bytes
            # get the image as raw Bytes
            image_raw = parsed_example["img"]

            # Decode the raw bytes so it becomes a tensor with type
            image = tf.decode_raw(image_raw, tf.uint8)

            # only float if i want a tensor
            # image = tf.cast(image, tf.float32)

            return image

        # applies _parse_image_functionc to each element of this data set
        dataset = self.DATASET.map(_parse_image_function)

        # set the dimensions
        dataset = dataset.batch(3)

        # go through the data set
        iterator = dataset.make_one_shot_iterator()

        # get the next element in the dataset
        tensor = iterator.get_next()

        rawBytesArray = tensor.eval(session=tf.compat.v1.Session())

        # set new tensor variable with rawBytes and type tf.float32
        allImageTensor = tf.constant(rawBytesArray, tf.float32)


        CNN.inference(self, allImageTensor)

        # for oneRawBytesArray in rawBytesArray:
        #
        #     oneRawBytesArray = oneRawBytesArray.reshape((303, 303, 3))
        #     oneImageTensor = tf.constant(oneRawBytesArray, dtype=tf.float32)
        #
        #     # Visiualisation + blur
        #     # blur = cv2.GaussianBlur(oneRawBytesArray, (5, 5), 0)
        #     #
        #     # CNN.visualisation(self, data1=oneRawBytesArray, data2=blur)

        return

    def inference(self, allImageTensor):
        with tf.name_scope('conv1'):
            output = 1
            k_w = 5
            k_h = 5
            s = [1, 1, 1, 1]
            pad = 'VALID'
            w_shape = [k_w, k_h, 3, output]
            num_channels = 3
            img_size = 303

            x_image = tf.reshape(allImageTensor, [-1, img_size, img_size, num_channels])

            # set filter variable and the dimension (5, 5, 3, 1) like the input data
            filterExamnple = [[1, 4, 7, 4, 1], [4, 16, 26, 16, 4], [7, 26, 41, 26, 7], [4, 16, 26, 16, 4],
                              [1, 4, 7, 4, 1]]
            filter = np.repeat(filterExamnple, 3)
            filter = np.reshape(filter, w_shape)
            filter = tf.compat.v1.convert_to_tensor(filter, dtype=tf.float32)

            # calculate the convolutional layer and the pictures
            layer = tf.nn.conv2d(input=x_image, filter=filter, strides=s, padding=pad)

            # set negative values to zero
            layer = tf.nn.relu(layer)

            # Dimensions with size 1 will deleted
            layer = tf.squeeze(layer)

            print(
                "___________________________________________________normal_______________________________________\n",
                tf.compat.v1.Session().run(layer),
                "\n______________________________________________________________________________________________\n")

            # run the session and calculate everything what is in the graph
            test = tf.compat.v1.Session().run(layer)

            # visualize the pictures
            visualisation(self, data1=test)

            intLayer = layer
            # layer1 = tf.cast(layer1, tf.int8)
            intLayer = tf.divide(intLayer, 273)
            test1 = tf.compat.v1.Session().run(intLayer)
            print(
                "_____________________________________________________273_________________________________________\n",
                tf.compat.v1.Session().run(intLayer),
                "\n_______________________________________________________________________________________________\n")
            # visualisation(self, data1=rawBytesArray, data2=test1)


def visualisation(self, data1):
    f, axarr = plt.subplots(2, 2)
    axarr[0, 0].imshow(data1[0])
    axarr[0, 1].imshow(data1[1])
    axarr[1, 0].imshow(data1[2])
    plt.show()


# if __name__ == '__main__':
# tf.app.run()
testGetInput = CNN()
testGetInput.getInput()
