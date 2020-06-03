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

#Complete code at ...

# functin with ignoring the param
def main(_):
    save_path = ''
    model = CNN()                                                   # declare model with cnn()

    x = model.getInput()                                            # get IMG, open tf record
    y = model.inference(x)                                          # give IMG, apply gusian filter

    saver = tf.train.Saver()                                        # save the model! so you dont have to train it every code run

    global_init = tf.global_variables_initializer()                 # globale initialisierung
    local_init = tf.local_variables_initializer()                   # local initialisierung

    with tf.Session() as sess:                                      # ts.Session is sees shortcut and safe tf variables until you close the session
        sess.run(global_init)                                       # evaluealte the global variable
        sess.run(local_init)                                        # evaluate the local variable

        coord = tf.train.Coordinator()                              # coordination for threads (multitreading)
        threads = tf.train.start_queue_runners(coord=coord)         # start threads and give a list with threads back
        tf.train.start_queue_runners(sess=sess)                     # start threads and give a list with threads back

        for i in range(model.N_SAMPLES):                            # model.N_Samples = 3
            x_val, y_val = sess.run([x, y])

            #Task 3:
            # - Apply Gausian filter to image of Homer
            # - Save image
            # - Push snapshot and image to GIT
            #...

        # saver
        tf.train.save(sess, 'my_test_model')                        # save the training from ki

        coord.request_stop()                                        # stop threads
        coord.join(threads)                                         # wait for the threads to terminate
        print('done')

class CNN():                                                        # Model
    def __init__(self):                                             # initialise vaiables
        self.N_SAMPLES = 3
        self.DATASET = tf.data.TFRecordDataset("C:\\Users\\Q447230\\Coden\\gitRepositorys\\Coding_Task\\record_test.tfrecord")
        self.IN_SHAPE = ...
        pass


    # read indiviual images
    def getInput(self):

        filename = "C:\\Users\\Q447230\\Coden\\gitRepositorys\\Coding_Task\\record_test.tfrecord"
        dataset = tf.data.TFRecordDataset("C:\\Users\\Q447230\\Coden\\gitRepositorys\\Coding_Task\\record_test.tfrecord")

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
            #image = tf.cast(image, tf.float32)

            return image


        #dataset = self.DATASET.map(lambda x: tf.py_function(process_path, [x], [tf.string]))
        dataset = self.DATASET.map(_parse_image_function)

        #dataset = dataset.repeat(1)

        dataset = dataset.batch(3)

        iterator = dataset.make_one_shot_iterator()

        tensor = iterator.get_next()
        print("tensor", tensor)

        #x = {"img": images_batch}

        rawBytesArray = tensor.eval(session=tf.compat.v1.Session())


        for oneRawBytesArray in rawBytesArray:

            print("oneTensor", oneRawBytesArray)
            # print("tensor", rawBytesArray)

            oneRawBytesArray = oneRawBytesArray.reshape((303, 303, 3))
            #oneRawBytesArray = np.reshape(a=oneRawBytesArray, newshape=[303, 303, 3])

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

        # Gausian filter formula()
        def getGuessValue(kerStd, posX, posY):
            return 1. / (2. * math.pi * (np.power(kerStd, 2))) * math.exp(
                -(np.power(posX, 2) + np.power(posY, 2)) / (2. * (np.power(kerStd, 2))))

        def getGuessKernel(kerStd):
            K00 = np.column_stack(
                (np.row_stack((np.eye(3) * getGuessValue(kerStd, -1, 1), [0., 0., 0.])), np.array([0., 0., 0., 1.])))
            K10 = np.column_stack(
                (np.row_stack((np.eye(3) * getGuessValue(kerStd, 0, 1), [0., 0., 0.])), np.array([0., 0., 0., 1.])))
            K20 = np.column_stack(
                (np.row_stack((np.eye(3) * getGuessValue(kerStd, 1, 1), [0., 0., 0.])), np.array([0., 0., 0., 1.])))
            K30 = np.column_stack(
                (np.row_stack((np.eye(3) * getGuessValue(kerStd, -1, 0), [0., 0., 0.])), np.array([0., 0., 0., 1.])))
            K40 = np.column_stack(
                (np.row_stack((np.eye(3) * getGuessValue(kerStd, 0, 0), [0., 0., 0.])), np.array([0., 0., 0., 1.])))
            K01 = np.column_stack(
                (np.row_stack((np.eye(3) * getGuessValue(kerStd, 1, 0), [0., 0., 0.])), np.array([0., 0., 0., 1.])))
            K11 = np.column_stack(
                (np.row_stack((np.eye(3) * getGuessValue(kerStd, -1, -1), [0., 0., 0.])), np.array([0., 0., 0., 1.])))
            K21 = np.column_stack(
                (np.row_stack((np.eye(3) * getGuessValue(kerStd, 0, -1), [0., 0., 0.])), np.array([0., 0., 0., 1.])))
            K31 = np.column_stack(
                (np.row_stack((np.eye(3) * getGuessValue(kerStd, 1, -1), [0., 0., 0.])), np.array([0., 0., 0., 1.])))
            K41 = np.column_stack(
                (np.row_stack((np.eye(3) * getGuessValue(kerStd, 1, -1), [0., 0., 0.])), np.array([0., 0., 0., 1.])))
            K02 = np.column_stack(
                (np.row_stack((np.eye(3) * getGuessValue(kerStd, 1, -1), [0., 0., 0.])), np.array([0., 0., 0., 1.])))
            K12 = np.column_stack(
                (np.row_stack((np.eye(3) * getGuessValue(kerStd, 1, -1), [0., 0., 0.])), np.array([0., 0., 0., 1.])))
            K22 = np.column_stack(
                (np.row_stack((np.eye(3) * getGuessValue(kerStd, 1, -1), [0., 0., 0.])), np.array([0., 0., 0., 1.])))
            K32 = np.column_stack(
                (np.row_stack((np.eye(3) * getGuessValue(kerStd, 1, -1), [0., 0., 0.])), np.array([0., 0., 0., 1.])))
            K42 = np.column_stack(
                (np.row_stack((np.eye(3) * getGuessValue(kerStd, 1, -1), [0., 0., 0.])), np.array([0., 0., 0., 1.])))
            K03 = np.column_stack(
                (np.row_stack((np.eye(3) * getGuessValue(kerStd, 1, -1), [0., 0., 0.])), np.array([0., 0., 0., 1.])))
            K13 = np.column_stack(
                (np.row_stack((np.eye(3) * getGuessValue(kerStd, 1, -1), [0., 0., 0.])), np.array([0., 0., 0., 1.])))
            K23 = np.column_stack(
                (np.row_stack((np.eye(3) * getGuessValue(kerStd, 1, -1), [0., 0., 0.])), np.array([0., 0., 0., 1.])))
            K33 = np.column_stack(
                (np.row_stack((np.eye(3) * getGuessValue(kerStd, 1, -1), [0., 0., 0.])), np.array([0., 0., 0., 1.])))
            K43 = np.column_stack(
                (np.row_stack((np.eye(3) * getGuessValue(kerStd, 1, -1), [0., 0., 0.])), np.array([0., 0., 0., 1.])))
            K04 = np.column_stack(
                (np.row_stack((np.eye(3) * getGuessValue(kerStd, 1, -1), [0., 0., 0.])), np.array([0., 0., 0., 1.])))
            K14 = np.column_stack(
                (np.row_stack((np.eye(3) * getGuessValue(kerStd, 1, -1), [0., 0., 0.])), np.array([0., 0., 0., 1.])))
            K24 = np.column_stack(
                (np.row_stack((np.eye(3) * getGuessValue(kerStd, 1, -1), [0., 0., 0.])), np.array([0., 0., 0., 1.])))
            K34 = np.column_stack(
                (np.row_stack((np.eye(3) * getGuessValue(kerStd, 1, -1), [0., 0., 0.])), np.array([0., 0., 0., 1.])))
            K44 = np.column_stack(
                (np.row_stack((np.eye(3) * getGuessValue(kerStd, 1, -1), [0., 0., 0.])), np.array([0., 0., 0., 1.])))
            print(K11.shape)
            kernel = tf.constant(np.array(
                [
                    [
                        K00,
                        K10,
                        K20,
                        K30,
                        K40
                    ],
                    [
                        K01,
                        K11,
                        K21,
                        K31,
                        K41
                    ],
                    [
                        K02,
                        K12,
                        K22,
                        K32,
                        K42
                    ],
                    [
                        K03,
                        K13,
                        K23,
                        K33,
                        K43
                    ],
                    [
                        K04,
                        K14,
                        K24,
                        K34,
                        K44
                    ]
                ])
                , dtype=tf.float32)  # 3*3*4*4 | 5,5,4,4
            return kernel

        def getImageData(oneRawBytesArray):
            return np.array(oneRawBytesArray, dtype=np.float32)

        imageData = getImageData(oneRawBytesArray)
        testData = tf.constant(imageData)
        kernel = getGuessKernel(0.8)
        y = tf.cast(tf.nn.conv2d(testData, kernel, strides=[5, 5, 4, 4], padding="SAME"), dtype=tf.int32)
        init_op = tf.global_variables_initializer()

        g = tf.Graph()
        with tf.Session(graph=g) as sess:
            # print(testData.get_shape())
            # print(kernel.eval())
            # print(kernel.get_shape())
            resultData = sess.run(y)[0]
            print(resultData.shape)
            resulImage = Image.fromarray(np.uint8(resultData))
            resulImage.show()
            print(y.get_shape())

        return y


















































        # with tf.name_scope('conv1'):
        #     out = 1
            # k_w = 5
            # k_h = 5
            # s = [1, 1, 1, 1]
            # pad = 'VALID'
            # w_shape = [k_w, k_h, 3, out]
            #
            # print("oneRawBytesArray:", oneRawBytesArray)
            # np.array(oneRawBytesArray)
            #
            #
            # a = tf.constant(oneRawBytesArray)
            # b = cv2.GaussianBlur(a, (5, 5), 0)
            # print("a", a)
            #
            # print("before Session")
            # sess = tf.compat.v1.Session()
            # print(sess.run(b))

            # g = tf.Graph()
            # with g.as_default():
            #     pass

            # def gaussian_blur(img, kernel_size=11, sigma=5):
            #     def gauss_kernel(channels, kernel_size, sigma):
            #         ax = tf.range(-kernel_size // 2 + 1.0, kernel_size // 2 + 1.0)
            #         xx, yy = tf.meshgrid(ax, ax)
            #         kernel = tf.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
            #         kernel = kernel / tf.reduce_sum(kernel)
            #         kernel = tf.tile(kernel[..., tf.newaxis], [1, 1, channels])
            #         return kernel
            #
            #     gaussian_kernel = gauss_kernel(tf.shape(img)[-1], kernel_size, sigma)
            #     gaussian_kernel = gaussian_kernel[..., tf.newaxis]
            #
            #     return tf.nn.depthwise_conv2d(img, gaussian_kernel, [1, 1, 1, 1], padding='SAME', data_format='NHWC')

            #TODO: Task 2
            # - Apply a 5x5 Gausian filter to the input
            # - with graph and session



#if __name__ == '__main__':
    #tf.app.run()
testGetInput = CNN()
testGetInput.getInput()