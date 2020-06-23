import os
import shutil
import sys
import time
import uuid
import cv2
import numpy as np
import tensorflow as tf
import logging
import datetime
from tensorflow.contrib import slim
logging.basicConfig(level=logging.DEBUG)
sys.path.append('src/')
from src.nets import model_train as model
from src.utils.prepare import split_label
from src.utils.text_connector.detectors import TextDetector
from src.utils.rpn_msr.proposal_layer import proposal_layer
from src.utils.dataset import data_provider as data_provider
from src.utils.detection_utils.detection import resize_image, show_img, sort_boxes, four_point_transform, order_points

class CTPN:
    '''
    class to automate CTPN basic methods like predict
    '''

    def __init__(self, gpu=0, weights='checkpoints_mlt/', debug=False):
        tf.app.flags.DEFINE_string('gpu', str(gpu), '')
        tf.app.flags.DEFINE_string(
            'checkpoint_path', weights, '')
        tf.app.flags.DEFINE_boolean('debug', debug, '')
        tf.app.flags.DEFINE_string('output_path', 'data/res/', '')
        self.flags = tf.app.flags.FLAGS

    def detect_text(self, images_path, detect_mode='H', to_lines=False):
        if os.path.exists(self.flags.output_path):
            shutil.rmtree(self.flags.output_path)
        os.makedirs(self.flags.output_path)
        os.environ['CUDA_VISIBLE_DEVICES'] = self.flags.gpu
        with tf.get_default_graph().as_default():
            input_image = tf.placeholder(
                tf.float32, shape=[None, None, None, 3], name='input_image')
            input_im_info = tf.placeholder(
                tf.float32, shape=[None, 3], name='input_im_info')

            global_step = tf.get_variable(
                'global_step', [], initializer=tf.constant_initializer(0), trainable=False)

            bbox_pred, cls_pred, cls_prob = model.model(input_image)

            variable_averages = tf.train.ExponentialMovingAverage(
                0.997, global_step)
            saver = tf.train.Saver(variable_averages.variables_to_restore())

            with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
                ckpt_state = tf.train.get_checkpoint_state(
                    self.flags.checkpoint_path)
                model_path = os.path.join(self.flags.checkpoint_path, os.path.basename(
                    ckpt_state.model_checkpoint_path))
                logging.info('Restore from {}'.format(model_path))
                saver.restore(sess, model_path)
                for img in os.listdir(images_path):
                    image = cv2.imread(images_path+img)
                    start = time.time()
                    img, (rh, rw) = resize_image(image)
                    h, w, c = img.shape
                    im_info = np.array([h, w, c]).reshape([1, 3])
                    bbox_pred_val, cls_prob_val = sess.run([bbox_pred, cls_prob],
                                                           feed_dict={input_image: [img],
                                                                      input_im_info: im_info})
                    textsegs, _ = proposal_layer(
                        cls_prob_val, bbox_pred_val, im_info)
                    scores = textsegs[:, 0]
                    textsegs = textsegs[:, 1:5]

                    textdetector = TextDetector(DETECT_MODE=detect_mode)
                    boxes = textdetector.detect(
                        textsegs, scores[:, np.newaxis], img.shape[:2])
                    boxes = np.array(boxes, dtype=np.int)
                    boxes = sort_boxes(boxes)
                    cost_time = (time.time() - start)
                    logging.info("cost time: {:.2f}s".format(cost_time))
                    text_regions = []
                    line_images = []
                    for idx, box in enumerate(boxes):
                        fourpts = box[:8].astype(np.int32).reshape(4, 2)
                        if to_lines:
                            line_images.append(
                                four_point_transform(img, fourpts))
                        text_regions.append(fourpts)
                        points = box[:8].astype(np.int32).reshape((-1, 1, 2))
                        cv2.polylines(img, [points], True, color=(0, 255, 0),
                                      thickness=2)
                    if self.flags.debug:
                        img = cv2.resize(img, None, None, fx=1.0 / rh,
                                         fy=1.0 / rw, interpolation=cv2.INTER_LINEAR)
                        cv2.imwrite(self.flags.output_path +
                                    str(uuid.uuid4())+'.png', img)
                        #show_img(img, 'box preds')
                return text_regions, line_images

    def train(self, lr=1e-5, max_steps=50000, decay_steps=30000, decay_rate=0.1, moving_average_decay=0.997, num_readers=4,
              gpu=0, ckpt_pth='checkpoints_mlt/',
              logs_pth='logs_mlt/', vgg_path='data/vgg_16.ckpt', restore=True, save_ckpts_every=2000):
        '''mainly used for finetuning by training the network for extra n iterations'''
        tf.app.flags.DEFINE_float('learning_rate', lr, '')
        tf.app.flags.DEFINE_integer('max_steps', max_steps, '')
        tf.app.flags.DEFINE_integer('decay_steps', decay_steps, '')
        tf.app.flags.DEFINE_float('decay_rate', decay_rate, '')
        tf.app.flags.DEFINE_float(
            'moving_average_decay', moving_average_decay, '')
        tf.app.flags.DEFINE_integer('num_readers', num_readers, '')
        tf.app.flags.DEFINE_string('gpu', str(gpu), '')
        tf.app.flags.DEFINE_string('checkpoint_path', ckpt_pth, '')
        tf.app.flags.DEFINE_string('logs_path', logs_pth, '')
        tf.app.flags.DEFINE_string(
            'pretrained_model_path', vgg_path, '')
        tf.app.flags.DEFINE_boolean('restore', restore, '')
        tf.app.flags.DEFINE_integer(
            'save_checkpoint_steps', save_ckpts_every, '')
        self.trainflags = tf.app.flags.FLAGS
        os.environ['CUDA_VISIBLE_DEVICES'] = self.trainflags.gpu
        now = datetime.datetime.now()
        StyleTime = now.strftime("%Y-%m-%d-%H-%M-%S")
        os.makedirs(self.trainflags.logs_path + StyleTime)
        if not os.path.exists(self.trainflags.checkpoint_path):
            os.makedirs(self.trainflags.checkpoint_path)

        input_image = tf.placeholder(
            tf.float32, shape=[None, None, None, 3], name='input_image')
        input_bbox = tf.placeholder(
            tf.float32, shape=[None, 5], name='input_bbox')
        input_im_info = tf.placeholder(
            tf.float32, shape=[None, 3], name='input_im_info')

        global_step = tf.get_variable(
            'global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        learning_rate = tf.Variable(
            self.trainflags.learning_rate, trainable=False)
        tf.summary.scalar('learning_rate', learning_rate)
        opt = tf.train.AdamOptimizer(learning_rate)

        gpu_id = int(self.trainflags.gpu)
        with tf.device('/gpu:%d' % gpu_id):
            with tf.name_scope('model_%d' % gpu_id) as scope:
                bbox_pred, cls_pred, cls_prob = model.model(input_image)
                total_loss, model_loss, rpn_cross_entropy, rpn_loss_box = model.loss(bbox_pred, cls_pred, input_bbox,
                                                                                     input_im_info)
                batch_norm_updates_op = tf.group(
                    *tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope))
                grads = opt.compute_gradients(total_loss)

        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

        summary_op = tf.summary.merge_all()
        variable_averages = tf.train.ExponentialMovingAverage(
            self.trainflags.moving_average_decay, global_step)
        variables_averages_op = variable_averages.apply(
            tf.trainable_variables())
        with tf.control_dependencies([variables_averages_op, apply_gradient_op, batch_norm_updates_op]):
            train_op = tf.no_op(name='train_op')

        saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)
        summary_writer = tf.summary.FileWriter(
            self.trainflags.logs_path + StyleTime, tf.get_default_graph())

        init = tf.global_variables_initializer()

        if self.trainflags.pretrained_model_path is not None:
            variable_restore_op = slim.assign_from_checkpoint_fn(self.trainflags.pretrained_model_path,
                                                                 slim.get_trainable_variables(),
                                                                 ignore_missing_vars=True)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.95
        config.allow_soft_placement = True
        with tf.Session(config=config) as sess:
            if self.trainflags.restore:
                ckpt = tf.train.latest_checkpoint(
                    self.trainflags.checkpoint_path)
                restore_step = int(ckpt.split('.')[0].split('_')[-1])
                logging.info("continue training from previous checkpoint {}".format(
                    restore_step))
                saver.restore(sess, ckpt)
            else:
                sess.run(init)
                restore_step = 0
                if self.trainflags.pretrained_model_path is not None:
                    variable_restore_op(sess)

            data_generator = data_provider.get_batch(
                num_workers=self.trainflags.num_readers)
            start = time.time()
            for step in range(restore_step, self.trainflags.max_steps):
                data = next(data_generator)
                ml, tl, _, summary_str = sess.run([model_loss, total_loss, train_op, summary_op],
                                                  feed_dict={input_image: data[0],
                                                             input_bbox: data[1],
                                                             input_im_info: data[2]})

                summary_writer.add_summary(summary_str, global_step=step)

                if step != 0 and step % self.trainflags.decay_steps == 0:
                    sess.run(tf.assign(learning_rate,
                                       learning_rate.eval() * self.trainflags.decay_rate))

                if step % 10 == 0:
                    avg_time_per_step = (time.time() - start) / 10
                    start = time.time()
                    logging.info('Step {:06d}, model loss {:.4f}, total loss {:.4f}, {:.2f} seconds/step, LR: {:.6f}'.format(
                        step, ml, tl, avg_time_per_step, learning_rate.eval()))

                if (step + 1) % self.trainflags.save_checkpoint_steps == 0:
                    filename = ('ctpn_{:d}'.format(step + 1) + '.ckpt')
                    filename = os.path.join(
                        self.trainflags.checkpoint_path, filename)
                    saver.save(sess, filename)
                    logging.info('Write model to: {:s}'.format(filename))
