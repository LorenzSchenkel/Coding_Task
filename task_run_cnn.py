import tensorflow as tf
import os
import IPython.display as display
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
import cv2


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
            image = tf.cast(image, tf.float32)

            return image


        #dataset = self.DATASET.map(lambda x: tf.py_function(process_path, [x], [tf.string]))
        dataset = self.DATASET.map(_parse_image_function)

        #dataset = dataset.repeat(1)
        # dataset = image_features.repeat(1)

        dataset = dataset.batch(3)
        # dataset = image_features.batch(3)

        iterator = dataset.make_one_shot_iterator()
        # iterator = image_features.make_one_shot_iterator()

        images_batch = iterator.get_next()

        #x = {"img": images_batch}

        y = images_batch.eval(session=tf.compat.v1.Session())


        for tensor in y:

            print("type", type(tensor))


            #tensor = bytearray(tensor)

            print("type", type(tensor))


            tensor = np.reshape(tensor, (303, 303, 3))

            # tf.reshape(tensor, (303, 303, 3))

            print(tensor)

            blur = cv2.GaussianBlur(tensor, (5, 5), 0)

            plt.subplot(121), plt.imshow(tensor), plt.title('without filter')
            plt.xticks([]), plt.yticks([])
            plt.subplot(122), plt.imshow(blur), plt.title('with filter')
            plt.xticks([]), plt.yticks([])
            plt.show()

            print(tensor)

        return y


    def inference(self, img):
        with tf.name_scope('conv1'):
            out = 1
            k_w = 5
            k_h = 5
            s = [1, 1, 1, 1]
            pad = 'VALID'
            w_shape = [k_w, k_h, 3, out]

            g = tf.Graph()
            with g.as_default():
                pass

            #Task 2
            # - Apply a 5x5 Gausian filter to the input
            #...

        return

#if __name__ == '__main__':
    #tf.app.run()
testGetInput = CNN()
testGetInput.getInput()