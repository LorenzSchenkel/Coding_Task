import tensorflow as tf
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

os.environ["CUDA_VISIBLE_DEVICES"] = '-1'

#Complete code at ...

def main(_):
    save_path = ''
    model = CNN()
    x = model.getInput()
    y = model.inference(x)

    saver = tf.train.Saver()

    global_init = tf.global_variables_initializer()
    local_init = tf.local_variables_initializer()
    with tf.Session() as sess:
        sess.run(global_init)
        sess.run(local_init)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        tf.train.start_queue_runners(sess=sess)

        for i in range(model.N_SAMPLES):
            x_val,y_val = sess.run([x,y])

            #Task 3:
            # - Apply Gausian filter to image of Homer
            # - Save image
            # - Push snapshot and image to GIT
            #...

        saver ...

        coord.request_stop()
        coord.join(threads)
        print('done')

class CNN():
    def __init__(self):
        self.N_SAMPLES = 3
        self.DATASET = ...
        self.IN_SHAPE = ...
        pass

    def getInput(self):
        # Task 1:
        # Read TfRecord
        tfrecord_file_queue = tf.train.string_input_producer('...', name='queue', num_epochs=None)
        #...
        return img

    def inference(self, img):
        with tf.name_scope('conv1'):
            out = 1
            k_w = 5
            k_h = 5
            s = [1, 1, 1, 1]
            pad = 'VALID'
            w_shape = [k_w, k_h, 3, out]

            #Task 2
            # - Apply a 5x5 Gausian filter to the input
            #...

            h_conv1 = tf.nn.conv2d(img, w_conv1, strides=s, padding=pad, name='h_conv1')
            h_conv1 = b_conv1*h_conv1
        return h_conv1

if __name__ == '__main__':
    tf.app.run()