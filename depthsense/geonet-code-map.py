import os
import time
import random
from datetime import datetime
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import scipy.misc
import scipy.io
import cv2
#from utils_sceneparsing import *

os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
os.environ['CUDA_VISIBLE_DEVICES']=str(np.argmax([int(x.split()[2]) for x in open('tmp','r').readlines()]))
os.system('rm tmp')
print(os.environ['CUDA_VISIBLE_DEVICES'])

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)

crop_size_h = 481
crop_size_w = 641

train_phase = False
# load caffe weight
def weight_from_caffe(caffenet):
    def func(shape, dtype, partition_info=None):
        sc = tf.get_variable_scope()
        name = sc.name.split('/')[-1]
        print ('init: ', name, shape, caffenet[name][0].shape)
        return tf.transpose(caffenet[name][0], perm=[2, 3, 1, 0])
    return func

# load caffe bias
def bias_from_caffe(caffenet):
    def func(shape, dtype, partition_info=None):
        sc = tf.get_variable_scope()
        name = sc.name.split('/')[-1]
        return caffenet[name][1]
    return func

# data process
def myfunc(x):
    try:
        data_dic = scipy.io.loadmat(x)
        data_img = data_dic['img']
        #print "aaaaa"
        data_depth = data_dic['depth']
        depth_mask = np.zeros((crop_size_h, crop_size_w), dtype=np.float32)
        depth_mask[np.where(data_depth < 0.1)] = 0.0
        depth_mask[np.where(data_depth >= 0.1)] = 1.0
        data_norm = data_dic['norm']
        data_mask = data_dic['mask']
        grid = data_dic['grid']
    except:
        data_img = np.zeros((crop_size_h, crop_size_w, 3), dtype=np.float32)
        data_depth = np.zeros((crop_size_h, crop_size_w), dtype=np.float32)
        data_mask = np.zeros((crop_size_h, crop_size_w), dtype=np.float32)
        data_norm = np.zeros((crop_size_h, crop_size_w, 3), dtype=np.float32)
        depth_mask = np.zeros((crop_size_h, crop_size_w), dtype=np.float32)
        grid = np.zeros((crop_size_h, crop_size_w,3), dtype=np.float32)

    return data_img, data_depth, data_norm, data_mask,depth_mask,grid

# canny edge extractor
def myfunc_canny(img):
    img = np.squeeze(img)
    img = img + 128.0
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #print(img.shape())
    img = ((img-img.min())/(img.max()-img.min()))*255.0
    edges = cv2.Canny(img.astype('uint8'), 100, 220)
    edges = edges.astype(np.float32)
    edges = edges.reshape((1,crop_size_h,crop_size_w,1))
    edges = 1 - edges/255.0
    return edges

#edge aware refinement
def propagate(input_data,dlr,drl,dud,ddu,dim):
    # < replaced with refinement.py propagate>
    pass

class DEEPLAB(object):
    def __init__(self, fcn_ver=32):
        self.deeplab_ver = 'largeFOV'
        self.mean_BGR = [104.008, 116.669, 122.675]
        self.pretrain_weight = np.load('./initilization_model/model_denoise_depth_norm.npy',allow_pickle=True).tolist()

        self.crop_size = 320
        self.crop_size_h = 481
        self.crop_size_w = 641
        self.batch_size = 1
        self.max_steps = int(400000)
        self.train_dir = './trainmodel/'
        self.data_list = open('./list/traindata_grid.txt', 'rt').read().splitlines()
        self.starter_learning_rate = 1e-5
        self.global_step = tf.Variable(0, trainable=False, dtype=tf.float32)
        self.end_learning_rate = 1e-6
        self.decay_steps = int(200000)
        self.k = 9
        self.rate = 4
        self.clip_norm = 20.0
        self.thresh = 0.95
        random.shuffle(self.data_list)

    def input_producer(self):
        def read_data():
            image, depth, norm,mask, depth_mask, grid= tf.py_func(myfunc,[self.data_queue[0]],[tf.float32,tf.float32,tf.float32,tf.float32,tf.float32,tf.float32])
            image, depth, norm, mask, depth_mask, grid = preprocessing(image, depth, norm, mask,depth_mask,grid)
            return image, depth, norm, mask, depth_mask,grid

        # data loader + data augmentation
        def preprocessing(image, depth, norm, mask,depth_mask, grid):

            image = tf.cast(image, tf.float32)
            depth = tf.cast(depth, tf.float32)
            norm = tf.cast(norm, tf.float32)
            mask = tf.cast(mask, tf.float32)
            depth_mask = tf.cast(depth_mask, tf.float32)
            grid = tf.cast(grid, tf.float32)
            random_num = tf.random_uniform([], minval=0, maxval=1.0, dtype=tf.float32, seed=None, name=None)

            mirror_cond = tf.less(random_num, 0.5)
            stride = tf.where(mirror_cond, -1, 1)
            image = image[:, ::stride, :]
            depth = depth[:, ::stride]
            mask = mask[:, ::stride]
            depth_mask = depth_mask[:, ::stride]
            norm = norm[:, ::stride, :]
            norm_x, norm_y, norm_z = tf.split(value=norm, num_or_size_splits=3, axis=2)
            norm_x = tf.scalar_mul(tf.cast(stride, dtype=tf.float32), norm_x)
            norm = tf.cast(tf.concat([norm_x, norm_y, norm_z], 2), dtype=tf.float32)


            img_r, img_g, img_b = tf.split(value=image, num_or_size_splits=3, axis=2)
            image = tf.cast(tf.concat([img_b, img_g, img_r], 2), dtype=tf.float32)

            image.set_shape((crop_size_h, crop_size_w, 3))
            depth.set_shape((crop_size_h, crop_size_w))
            norm.set_shape((crop_size_h, crop_size_w, 3))
            mask.set_shape((crop_size_h, crop_size_w))
            depth_mask.set_shape((crop_size_h, crop_size_w))
            grid.set_shape((crop_size_h,crop_size_w,3))
            return image, depth, norm, mask, depth_mask,grid

        with tf.variable_scope('input'):
            imglist = tf.convert_to_tensor(self.data_list, dtype=tf.string)
            self.data_queue = tf.train.slice_input_producer([imglist], capacity=100)
            images, depths,norms,masks,depth_masks, grid = read_data()
            batch_images, batch_depths, batch_norms, batch_masks,batch_depth_masks, grid = tf.train.batch([images, depths, norms, masks,depth_masks, grid], batch_size=self.batch_size, num_threads=4, capacity=60)
        return batch_images, batch_depths, batch_norms, batch_masks, batch_depth_masks, grid

    def forward(self,inputs, grid,is_training=True, reuse=False):
        def preprocessing(inputs):
            dims = inputs.get_shape()
            if len(dims) == 3:
                inputs = tf.expand_dims(inputs, dim=0)
            mean_BGR = tf.reshape(self.mean_BGR, [1, 1, 1, 3])
            inputs = inputs[:, :, :, ::-1] + mean_BGR
            return inputs

         ## -----------------------depth and normal FCN--------------------------
        inputs = preprocessing(inputs)
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose], activation_fn=tf.nn.relu, stride=1,
                                padding='SAME',
                                weights_initializer=weight_from_caffe(self.pretrain_weight),
                                biases_initializer=bias_from_caffe(self.pretrain_weight)):

            with tf.variable_scope('fcn', reuse=reuse):
                ##---------------------vgg depth + norm ------------------------------------
                # Implemented in DepthSenseHead.depth_head and normal_head
                #
                # GeoNet (VGG): VGG16 w/ heavy convolutional stacking + dropout
                # DepthSense (ViT): Vision Transformer (ViT) via DINOv2
                #-------------------------------------vgg norm end---------------------------------------------
                pass

            # ------------- depth to normal + norm refinement---------------------------------------------------
            with tf.variable_scope('noise', reuse=reuse):
                # NormalRefinement
                pass

            # ------------- normal to depth  + depth refinement---------------------------------------------------
            with tf.variable_scope('norm_depth', reuse=reuse):
                # DepthRefinement
                pass

            with tf.variable_scope('edge_refinemet', reuse=reuse):
                # EdgeRefinement
                pass

        return final_depth,fc8_upsample_norm,norm_pred_final,fc8_upsample

    def train_op(self, loss):

        lr = tf.train.polynomial_decay(learning_rate = self.starter_learning_rate, global_step = self.global_step,
                                          decay_steps = self.decay_steps, end_learning_rate = self.end_learning_rate,
                                          power=0.9)

        print (self.decay_steps)
        print (self.end_learning_rate)
        opt = tf.train.AdamOptimizer(learning_rate = lr)
        grads_vars = opt.compute_gradients(loss)
        vars_name = []
        grads_vars_mult = []
        grads_value = []

        for grad, vars in grads_vars:

            t = 1
            if 'fcn' in vars.name:
                t = 0
            # t to 1 if finetune fcn part
            if 'edge_refinemet' in vars.name:
                t = 1

            grad *= t
            grads_vars_mult.append((grad, vars))
            vars_name.append(vars)
            grads_value.append(grad)
        global_norm_tmp = tf.global_norm(grads_value)
        if self.clip_norm > 0 :
            grads_value, _ = tf.clip_by_global_norm(grads_value, self.clip_norm)

        return opt.apply_gradients(zip(grads_value,vars_name), global_step = self.global_step),global_norm_tmp,lr

    def train(self):

        sess = tf.Session()
        self.sess = sess
        print(len(self.data_list))
        inputs, batch_depths, batch_norms, batch_masks,batch_depth_masks, batch_grids = self.input_producer()
        print(inputs)

        final_depth,fc8_upsample_norm, norm_pred_noise,fc8_upsample = self.forward(inputs,batch_grids)

        fc8_upsample = tf.squeeze(fc8_upsample)
        fc8_upsample_norm = tf.reshape(fc8_upsample_norm, [self.batch_size, self.crop_size_h, self.crop_size_w, 3])

        batch_masks = tf.reshape(batch_masks,[self.batch_size,self.crop_size_h,self.crop_size_w,1])
        batch_masks = tf.tile(batch_masks,[1,1,1,3])
        # Use GeoNetLoss.forward(fc8_upsample, final_depth,
        #                       fc8_upsample_norm, norm_pred_noise,
        #                       batch_depths, batch_norms,
        #                       batch_depth_masks, batch_masks)

        # train_op
        train_op,global_norm_tmp,lr = self.train_op(loss)
        #sum_grad = tf.

        sess.run(tf.global_variables_initializer())

        self.saver = tf.train.Saver(max_to_keep=50, keep_checkpoint_every_n_hours=1)
        self.load()


        '''
        saver_a = tf.train.Saver([v for v in tf.trainable_variables()])
        saver_a.restore(sess,'./trainmodel/norm_refine_depth_denoise_conv_depth_complex_v3_edge/checkpoints/SRCNN.model-399999')
        '''
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        loss_matrix = np.zeros((self.max_steps+1,4),dtype=np.float32)



        for step in range(0,self.max_steps):

            start_time = time.time()
            _, loss_value,loss1_value,loss2_value,loss3_value,loss4_value,global_norm_tmp_value,lr_val,global_step_val = sess.run([train_op, loss, loss1, loss2, loss3,loss4,global_norm_tmp,lr,self.global_step])
            duration = time.time() - start_time


            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            if step % 10 == 0:
                num_examples_per_step = self.batch_size
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)

                format_str = ('%s: step %d, loss = %.5f, loss1 = %.5f, loss2 = %.5f, loss3 = %.5f, loss4 = %.5f lr = %.10f (%.1f examples/sec; %.3f '
                              'sec/batch)')
                print (global_norm_tmp_value)
                print (format_str % (datetime.now(), step, loss_value, loss1_value,loss2_value,loss3_value,loss4_value, lr_val,
                                     examples_per_sec, sec_per_batch))

            if step % 10000 == 0 or (step + 1) == self.max_steps:
                checkpoint_path = os.path.join(self.train_dir, 'checkpoints')
                self.save(checkpoint_path, step)
            loss_matrix[step, 0] = loss1_value
            loss_matrix[step, 1] = loss2_value
            loss_matrix[step, 2] = loss3_value
            loss_matrix[step, 3] = loss4_value

    def load(self, checkpoint_dir='checkpoints', step=None):
        print(" [*] Reading checkpoints...")

        checkpoint_dir = os.path.join(self.train_dir, checkpoint_dir)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            print(os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False

    def save(self, checkpoint_dir, step):
        model_name = "SRCNN.model"

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.saver.save(self.sess, os.path.join(checkpoint_dir, model_name), global_step=step)

    def test(self):
        inputs = tf.placeholder(shape=[1, crop_size_h, crop_size_w, 3], dtype=tf.float32)
        grid = tf.placeholder(shape=[1, crop_size_h, crop_size_w, 3], dtype=tf.float32)
        estimate_depth, fc8_upsample_norm, norm_pred_noise, fc8_upsample =  self.forward(inputs, grid, is_training=False)
        exp_depth  = tf.exp(fc8_upsample * 0.69314718056)

        sess = tf.Session()
        self.sess = sess
        sess.run(tf.global_variables_initializer())


        saver_a = tf.train.Saver([v for v in tf.trainable_variables()])
        saver_a.restore(sess, './trainmodel/checkpoints/SRCNN.model-399999')

        list = scipy.io.loadmat('./data/splits.mat')
        list = list['testNdxs']-1

        images = scipy.io.loadmat('./data/images_uint8.mat')
        images = images['images']
        images = images[:,:,:,list]

        grid_dic = scipy.io.loadmat('./data/grid.mat')
        grid_data = grid_dic['grid']
        grid_data = np.expand_dims(grid_data, axis=0)
        num = list.shape[0]
        depths_pred = np.zeros((crop_size_h, crop_size_w, num), dtype=np.float32)
        norms_pred = np.zeros((crop_size_h, crop_size_w, 3, num), dtype=np.float32)

        norms_pred_estimate = np.zeros((crop_size_h, crop_size_w, 3, num), dtype=np.float32)
        depths_pred_estimate = np.zeros((crop_size_h, crop_size_w, num), dtype=np.float32)
        input1 = np.zeros((1, crop_size_h, crop_size_w, 3), dtype=np.float32)

        for i in range(0, images.shape[3]):
            print(i)
            img_data = images[:, :, :, i]

            img_data = np.expand_dims(img_data, axis=0)
            img_data_r = img_data[0, :, :, 0] - 122.675 * 2
            img_data_g = img_data[0, :, :, 1] - 116.669 * 2
            img_data_b = img_data[0, :, :, 2] - 104.008 * 2

            input1[0, 0:crop_size_h-1, 0:crop_size_w-1, 0] = np.squeeze(img_data_r)
            input1[0, 0:crop_size_h-1, 0:crop_size_w-1, 1] = np.squeeze(img_data_g)
            input1[0, 0:crop_size_h-1, 0:crop_size_w-1, 2] = np.squeeze(img_data_b)

            original_depth, original_norm, refined_norm, refined_depth = sess.run(
                [exp_depth, fc8_upsample_norm, norm_pred_noise, estimate_depth],
                feed_dict={inputs: input1, grid: grid_data})
            depths_pred[:, :, i] = np.squeeze(original_depth)
            norms_pred[:, :, :, i] = np.squeeze(original_norm)
            norms_pred_estimate[:, :, :, i] = np.squeeze(refined_norm)
            depths_pred_estimate[:,:,i] = np.squeeze(refined_depth)

        #scipy.io.savemat(self.train_dir+'/depths_pred.mat',{'depths':depths_pred})
        #scipy.io.savemat(self.train_dir + '/norms_pred.mat', {'norms':norms_pred})
        scipy.io.savemat(self.train_dir+'/norms_estimate.mat', {'norms': norms_pred_estimate})
        scipy.io.savemat(self.train_dir + '/depths_estimate.mat', {'depths': depths_pred_estimate})


def main(_):
    model = DEEPLAB()
    if train_phase:
        model.train()
    else:
        model.test()


if __name__ == '__main__':
    tf.app.run()
