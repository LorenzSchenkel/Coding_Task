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

    with tf.Session() as sess:  # ts.Session is sees shortcut and safe tf variables until you close the session
        sess.run(global_init)  # evaluealte the global variable
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
        filename = "C:\\Users\\Q447230\\Coden\\gitRepositorys\\Coding_Task\\record_test.tfrecord"
        dataset = tf.data.TFRecordDataset(
            "C:\\Users\\Q447230\\Coden\\gitRepositorys\\Coding_Task\\record_test.tfrecord")

        def _parse_image_function(serialized):
            features = {
                'img': tf.io.FixedLenFeature([], tf.string),
                'x': tf.io.FixedLenFeature([], tf.int64),
                'y': tf.io.FixedLenFeature([], tf.int64),
                'z': tf.io.FixedLenFeature([], tf.int64),
            }

            parsed_example = tf.parse_single_example(serialized=serialized, features=features)

            image_raw = parsed_example["img"]
            image = tf.decode_raw(image_raw, tf.uint8)

            # only float if we want a tensor
            # image = tf.cast(image, tf.float32)

            return image

        # dataset = self.DATASET.map(lambda x: tf.py_function(process_path, [x], [tf.string]))
        dataset = self.DATASET.map(_parse_image_function)
        # print("dataset", dataset)

        # dataset = dataset.repeat(1)

        dataset = dataset.batch(3)
        # print("dataset_batch", dataset)

        iterator = dataset.make_one_shot_iterator()
        # print("iterator", iterator)

        tensor = iterator.get_next()
        # print("tensor", tensor)

        # x = {"img": images_batch}

        rawBytesArray = tensor.eval(session=tf.compat.v1.Session())
        # print("rawBytesArray", rawBytesArray)

        for oneRawBytesArray in rawBytesArray:
            # print("oneRawBytesArray", oneRawBytesArray)
            # print("tensor", rawBytesArray)

            #oneRawBytesArray = oneRawBytesArray.reshape((303, 303, 3))
            # oneRawBytesArray = np.reshape(a=oneRawBytesArray, newshape=[303, 303, 3])
            # print("oneRawBytesArray_reshape", oneRawBytesArray)

            # blur = cv2.GaussianBlur(oneRawBytesArray, (5, 5), 0)
            #
            # plt.subplot(121), plt.imshow(oneRawBytesArray), plt.title('without filter')
            # plt.xticks([]), plt.yticks([])
            # plt.subplot(122), plt.imshow(blur), plt.title('with filter')
            # plt.xticks([]), plt.yticks([])
            # plt.show()

            CNN.inference(self, tensor, oneRawBytesArray, rawBytesArray)
        return

    def inference(self, tensor, oneRawBytesArray, rawBytesArray):
        with tf.name_scope('conv1'):
            input = 1
            output = 1
            k_w = 5
            k_h = 5
            s = [1, 1, 1, 1]
            pad = 'VALID'
            w_shape = [k_w, k_h, 3, output]
            num_channels = 3
            img_size = 303
            img_size_flat = 275427  # how many elements are in one picture array

            print(tensor)

            arr = np.array([(1, 4, 7, 4, 1), (4, 16, 26, 16, 4), (7, 26, 41, 26, 7), (4, 16, 26, 16, 4), (1, 4, 7, 4, 1)])
            gausian = tf.convert_to_tensor(arr)
            #guasian = tf.cast(gausian, tf.float32)


            print("gausian", gausian)


            # input
            print(img_size_flat)
            x = tf.compat.v1.placeholder(tf.float32, shape=[None, img_size_flat], name='x')
            print("x", x)
            x_image = tf.reshape(x, [3, img_size, img_size, num_channels], [4]) # eigentlich minus 1 statt 3 | expected string or bytes like object -> x have to be a tensor?

            # weights with random numbers i am not sure if i want this
            weights = tf.Variable(tf.truncated_normal(w_shape, stddev=0.05))

            #filter
            # filter = tf.constant(w_shape)
            # filter = tf.cast(filter, tf.float32)
            # print("filter", filter)


            guasianFiltertPicture = tf.nn.conv2d(input=x_image, filter=weights, padding=pad, strides=s)
            print(guasianFiltertPicture)



            # TODO: Task 2
            # - Apply a 5x5 Gausian filter to the input
            # - with graph and session


# if __name__ == '__main__':
# tf.app.run()
testGetInput = CNN()
testGetInput.getInput()