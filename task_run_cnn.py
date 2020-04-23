import tensorflow as tf
import os
import IPython.display as display
import matplotlib.pyplot as plt
import numpy as np
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
        self.DATASET = ...
        self.IN_SHAPE = ...
        pass


# read indiviual images
    def getInput(self):
        # Task 1:
        # Read TfRecord
        print("in CNN getInput()")

        filename = 'C:\\Users\\Q447230\\Coden\\gitRepositorys\\Coding_Task\\128px-Felis_catus-cat_on_snow.jpg' #'128px-Felis_catus-cat_on_snow.jpg',
        dataset = tf.data.TFRecordDataset("C:\\Users\\Q447230\\Coden\\gitRepositorys\\Coding_Task\\record_test.tfrecord")

        features = {
            'img': tf.io.FixedLenFeature([], tf.string),
            'x': tf.io.FixedLenFeature([], tf.int64),
            'y': tf.io.FixedLenFeature([], tf.int64),
            'z': tf.io.FixedLenFeature([], tf.int64),
        }

        def _parse_image_function(example_proto):
            # Parse the input tf.Example proto using the dictionary above.
            return tf.io.parse_single_example(example_proto, features)

        image_dataset = dataset.map(_parse_image_function)

        for image_features in image_dataset:
            image_raw = image_features['img'].numpy()
            display.display(display.Image(data=image_raw))


    def inference(self, img):
        with tf.name_scope('conv1'):
            out = 1
            k_w = 5
            k_h = 5
            s = [1, 1, 1, 1]
            pad = 'VALID'
            w_shape = [k_w, k_h, 3, out]

# Stackoverflow
            # def gaussian_kernel(size: int,
            #                     mean: float,
            #                     std: float,
            #                     ):
            #     """Makes 2D gaussian Kernel for convolution."""
            #
            #     d = tf.distributions.Normal(mean, std)
            #
            #     vals = d.prob(tf.range(start=5, limit=5, dtype=tf.float32))
            #
            #     gauss_kernel = tf.einsum('i,j->ij',
            #                              vals,
            #                              vals)

            #Task 2
            # - Apply a 5x5 Gausian filter to the input
            #...

            #h_conv1 = tf.nn.conv2d(img, w_conv1, strides=s, padding=pad, name='h_conv1')
            #h_conv1 = b_conv1*h_conv1
            #h_conv1
        return

#if __name__ == '__main__':
    #tf.app.run()
testGetInput = CNN()
testGetInput.getInput()