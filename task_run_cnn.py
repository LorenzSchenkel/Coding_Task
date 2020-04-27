import tensorflow as tf
import os
from IPython.display import Image
import IPython.display as display
import matplotlib.pyplot as plt
import numpy as np
import cv2
import matplotlib.image as mpimg



# set gpu no for traning
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
            x_val,y_val = sess.run([x,y])

            #TODO:Task 3:
            # - Apply Gausian filter to image of Homer
            # - Save image
            # - Push snapshot and image to GIT

        # saver
        tf.train.save(sess, 'my_test_model')                        # save the training from ki

        coord.request_stop()                                        # stop threads
        coord.join(threads)                                         # wait for the threads to terminate
        print('done')

class CNN():                                                        # Model
    def __init__(self):                                             # initialise vaiables
        self.N_SAMPLES = 3
        self.DATASET = ...
        self.IN_SHAPE = ...
        pass


# read indiviual images
    def getInput(self):
        print("in CNN getInput()")

        # TFRecodDataset(filename=tf.string, tf.data.Dataset)
        dataset = tf.data.TFRecordDataset("C:\\Users\\Q447230\\Coden\\gitRepositorys\\Coding_Task\\record_test.tfrecord")

        # dict dataname & types what we expect to see in the tf.recod file
        features = {
            # warum string?
            'img': tf.io.FixedLenFeature([], tf.string),
            'x': tf.io.FixedLenFeature([], tf.int64),
            'y': tf.io.FixedLenFeature([], tf.int64),
            'z': tf.io.FixedLenFeature([], tf.int64),
        }

        # why example
        def _parse_image_function(example):
            #tf.io.parse_single_example(idk; features = dict with feature keys) -> return: get a dict with our data in raw bytes
            return tf.io.parse_single_example(example, features)

        # transform the whole dataset and return the new one
        image_dataset = dataset.map(_parse_image_function)

        # illiterate about the dataset (there are 3 images)
        for image_features in image_dataset:

            #
            image_raw = image_features['img'].numpy()

            # numpy array
            decoded = np.frombuffer(image_raw, dtype=np.uint8)

            # width: 303 height: 303 , 3 color channel
            decoded = decoded.reshape((303, 303, 3))

            # inference(img: numpy Array) -> return nothing
            CNN.inference(self, img=decoded)


    def inference(self, img):
        print("in inference()")

        # with tf.name_scope('conv1'):
        #     out = 1
        #     k_w = 5
        #     k_h = 5
        #     s = [1, 1, 1, 1]
        #     pad = 'VALID'
        #     w_shape = [k_w, k_h, 3, out]

        #
        blur = cv2.GaussianBlur(img, (5, 5), 0)

        plt.subplot(121), plt.imshow(img), plt.title('Original')
        plt.xticks([]), plt.yticks([])
        plt.subplot(122), plt.imshow(blur), plt.title('GaussianBlur')
        plt.xticks([]), plt.yticks([])
        plt.show()
        return

#if __name__ == '__main__':
    #tf.app.run()
testGetInput = CNN()
testGetInput.getInput()