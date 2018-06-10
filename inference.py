import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    # Load layers
    img_input = tf.get_default_graph().get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = tf.get_default_graph().get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3_out = tf.get_default_graph().get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4_out = tf.get_default_graph().get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7_out = tf.get_default_graph().get_tensor_by_name(vgg_layer7_out_tensor_name)

    
    return img_input, keep_prob, layer3_out, layer4_out, layer7_out
tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # TODO: Implement function
    # 1x1 conv on layer 7
    layer7_1x1 = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, padding = 'same', kernel_initializer = tf.random_normal_initializer(stddev=0.01),
                kernel_regularizer = tf.contrib.layers.l2_regularizer(1e-3))
    #Upsample
    layer4_in1 = tf.layers.conv2d_transpose(layer7_1x1, num_classes, 4, strides = (2,2), padding = 'same',
                kernel_initializer = tf.random_normal_initializer(stddev=0.01), kernel_regularizer = tf.contrib.layers.l2_regularizer(1e-3))
    #1x1 conv on layer 4
    layer4_1x1 = tf.layers.conv2d(vgg_layer4_out, num_classes, 1, padding = 'same', kernel_initializer = tf.random_normal_initializer(stddev=0.01),
                kernel_regularizer = tf.contrib.layers.l2_regularizer(1e-3))
    #Skip conection 7-4
    layer4_skipped = tf.add(layer4_in1, layer4_1x1)
    #  Upsample
    layer3_in1 = tf.layers.conv2d_transpose(layer4_skipped, num_classes, 4, strides = (2,2), padding = 'same', 
                kernel_initializer = tf.random_normal_initializer(stddev=0.01), kernel_regularizer = tf.contrib.layers.l2_regularizer(1e-3))
    # 1x1 conv on layer 3
    layer3_1x1 = tf.layers.conv2d(vgg_layer3_out, num_classes, 1, padding = 'same', kernel_initializer = tf.random_normal_initializer(stddev = 0.01),
                kernel_regularizer = tf.contrib.layers.l2_regularizer(1e-3))
    # Skip connection 4-3
    layer3_skipped = tf.add(layer3_in1, layer3_1x1)
    #upsample
    nn_last_layer = tf.layers.conv2d_transpose(layer3_skipped, num_classes, 16, strides = (8,8), padding = 'same',
                    kernel_initializer = tf.random_normal_initializer(stddev = 0.01), kernel_regularizer = tf.contrib.layers.l2_regularizer(1e-3))
    return nn_last_layer
tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # TODO: Implement function
    # Reshape as 2 dimension, where row is a pixel and column a class
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    correct_label = tf.reshape(correct_label, (-1, num_classes))
    # Loss function
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = correct_label))
    # Training optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
    train_op = optimizer.minimize(cross_entropy_loss)
    return logits, train_op, cross_entropy_loss

tests.test_optimize(optimize)




def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    model_dir = './trained_model/'


    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/
    saver = tf.train.import_meta_graph('Semantic_seg_trained.meta')
    with tf.Session() as sess:
        '''        
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)


        #TF place holders:
        correct_label = tf.placeholder(tf.int32, [None, None, None, num_classes], name= 'correct_label')
        learning_rate = tf.placeholder(tf.float32, name='learning_rate')
        
        input_image, keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out = load_vgg(sess, vgg_path)
        nn_last_layer = layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes)

        logits, train_op, cross_entropy_loss = optimize(nn_last_layer, correct_label, learning_rate, num_classes)
        '''
        # TODO: Train NN using the train_nn function
        saver.restore(sess, tf.train.latest_checkpoint(model_dir))
        print("Model restored")

#        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image, correct_label, keep_prob, learning_rate)

        # TODO: Save inference data using helper.save_inference_samples
        #  helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)
        global video_library
        print("Running Predictions...")
        #helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)
        predict_video(video_library, sess, image_shape, logits, keep_prob, input_image)

def predict_video(video_library, sess, image_shape, logits, keep_prob, input_image):
    video_dir = r"./test_video//"
    for video_data in video_list[0:1]:
        rect = video_data[1]
        video_output = video_data[0][:-4] +"_out.mp4"
        clip1 = VideoFileClip(video_dir + video_data[0])
        video_clip = clip1.fl_image(lambda frame: predict_frame(frame, sess, image_shape, logits, keep_prob, input_image))
        video_clip.write_videofile(video_output, audio=False)


def predict_frame(im, sess, image_shape, logits, keep_prob, input_image):
    image = scipy.misc.imresize(im, image_shape)

    im_softmax = sess.run(
            [tf.nn.softmax(logits)],
            {keep_prob: 1.0, image_pl: [image]})
    im_softmax = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])
    segmentation = (im_softmax > 0.5).reshape(image_shape[0], image_shape[1], 1)
    mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
    mask = scipy.misc.toimage(mask, mode="RGBA")
    street_im = scipy.misc.toimage(image)
    street_im.paste(mask, box=None, mask=mask)
    return street_im


if __name__ == '__main__':
    video_library =   [["GOPR0706_cut1.mp4", [210, 470]],
                        ["GOPR0706_cut2.mp4", [210, 470]],
                        ["GOPR0707_cut1.mp4", [316, 576]],
                        ["GOPR0708_cut1.mp4", [316, 576]],
                        ["GOPR0732_cut1.mp4", [316, 576]],
                        ["GOPR0732_cut2.mp4", [316, 576]],
                        ["GOPR0732_cut3.mp4", [316, 576]]
                        ]
    run()
