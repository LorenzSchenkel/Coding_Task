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

# zweiter versuch error: unable to read file
#         list_ds = tf.data.Dataset.list_files(str("C:\\Users\\Q447230\\Coden\\gitRepositorys\\Coding_Task\\record_test.tfrecord"))
#         print("list_ds: ", list_ds)
#         file_path = next(iter(list_ds))
#         print("file path:" + file_path)
#         image = CNN.parse(file_path)
#
#         def show(image):
#             plt.figure()
#             plt.imshow(image)
#             plt.axis('off')
#         show(image)

# erster versuch
        # dataset = tf.data.TFRecordDataset(filenames=filenames)
        # dataset = dataset.map(CNN.parse(filenames))
        #
        # dataset = dataset.batch(batch_size=32)
        #
        # dataset = dataset.repeat(1)
        #
        # iterater = dataset.make_one_shot_iterator()
        #
        # images_batch = iterater.get_next()
        #
        # x = {
        #     'img' : images_batch
        # }
        #
        # imgPlt = mpimg.imread(x[0])
        # print(imgPlt)
        # return

#    def parse(filenames):
#        pass
        # Reads an image from a file, decodes
        #it into a dense tensor




# zweiter versuch error: unable to read file
        # parts = tf.strings.split(filenames, os.sep)
        # label = parts[-2]
        #
        #
        # image = tf.io.read_file(filenames)
        # # bytes
        # image = tf.image.decode_image(image, dtype=tf.dtypes.uint8)
        # image = tf.image.decode_image(image, dtype=tf.dtypes.uint8)
        # image = tf.image.convert_image_dtype(image, tf.float32)
        # return image


# erster Versuch error: string_input_produce gibt es nicht mehr
        # print("in CNN parse()")
        #
        # filenames = ["C:\\Users\\Q447230\\Coden\\gitRepositorys\\Coding_Task\\record_test.tfrecord"]
        #
        # features = {
        #     'img': tf.io.FixedLenFeature([], tf.string),
        #     'x': tf.io.FixedLenFeature([], tf.int64),
        #     'y': tf.io.FixedLenFeature([], tf.int64),
        #     'z': tf.io.FixedLenFeature([], tf.int64),
        # }
        #
        # fq = tf.train.string_input_producer(tf.convert_to_tensor(filenames), num_epochs=1)  # edit the string which comes out (make from a python object a tensor object))
        # reader = tf.TFRecordReader()  # define tf reader
        # _, v = reader.read(fq)  # read fq and set iist to _, v |  A scalar string Tensor, a single serialized Example.
        #
        # parsed_example = tf.parse_single_example(v,features)  # prase a single example with( scalar string tensor serialized example, a dict mapping feature key to fixed len feature values)
        #
        # imageRaw = parsed_example["img"]
        #
        # image = tf.image.decode_raw(imageRaw, tf.unit8)
        #
        # image = tf.cast(image, tf.float32)
        #
        # return image


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