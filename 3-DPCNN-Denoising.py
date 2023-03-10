import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import network.PCNN_3D
import time
from util import *
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Parameter settings
data_path = './data/noisy/'
target_path = './data/clean/3D_hyperbolic_clean_126_32_32.bin'
learning_rate = 1e-4
iterations = 100000
test_interval = 1000
breakpoint_training = False
nn_dropped_rate = 0.35                 # Dropped rate of dropouts in 3-DPCNN
bernoulli_sampling_probability = 0.65  # Bernoulli sampling probability
num_ensemble_models = 200              # Number of ensemble models
dimension = (32, 32, 126)              # 3D_hyperbolic_noisy_126_32_32.bin (Xline, Inline, Time) = (32,32,126)
task_name = 'D'                        # D = Denoising
train_times = 3                        # 3-DPCNN is trained K times from scratch

def train(file_path, current_train, data_dimension, dropouts_dropped_rate, data_mask_probability):
    start = time.time()
    tf.reset_default_graph()
    input_volume = load_np_seismic(file_path, data_dimension)
    model_path = file_path[0:file_path.rfind(".")]+"/"+task_name+"_train"+str(current_train)+"/"
    os.makedirs(model_path, exist_ok=True)
    model = network.PCNN_3D.build_denoising_net(input_volume, dropouts_dropped_rate, data_mask_probability)
    loss = model['training_error']
    saver = model['saver']
    our_volume = model['our_volume']
    is_flip_lr = model['is_flip_lr']
    is_flip_ud = model['is_flip_ud']
    is_reverse_polarity = model['is_reverse_polarity']
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    sum_loss = 0
    loss_value_list = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        if breakpoint_training:
            ckpt = tf.train.get_checkpoint_state(model_path)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

        for step in range(iterations):
            feet_dict = {is_flip_lr: np.random.randint(0, 2), is_flip_ud: np.random.randint(0, 2),
                         is_reverse_polarity: np.random.randint(0, 2)}
            _, loss_value, o_volume = sess.run([optimizer, loss, our_volume], feed_dict=feet_dict)
            sum_loss += loss_value
            if (step + 1) % test_interval == 0:
                print("After %d training step(s)" % (step + 1),
                      "loss  is {:.9f}".format(sum_loss / test_interval))
                # save loss value
                loss_value_list = np.append(loss_value_list, sum_loss / test_interval)
                sum_loss = 0
                sum_volume = np.float32(np.zeros(our_volume.shape.as_list()))
                for j in range(num_ensemble_models):
                    feet_dict = {is_flip_lr: np.random.randint(0, 2), is_flip_ud: np.random.randint(0, 2),
                                 is_reverse_polarity: np.random.randint(0, 2)}
                    o_volume = sess.run(our_volume, feed_dict=feet_dict)
                    sum_volume += o_volume
                o_volume = np.squeeze(sum_volume / num_ensemble_models)
                o_volume.transpose(1, 2, 0).tofile(model_path + '3-DPCNN_' + str(step + 1) + '.bin')
                saver.save(sess, model_path + "model.ckpt-" + str(step + 1))
        np.savetxt(model_path + 'train_loss.txt', loss_value_list.tolist())
    end = time.time()
    time_elapsed = end - start
    print('Time cost is {:.2f}m' .format(time_elapsed / 60))
    return model_path

if __name__ == '__main__':
    print_parameter_information(learning_rate,nn_dropped_rate,bernoulli_sampling_probability,num_ensemble_models,train_times,task_name)
    files_name = os.listdir(data_path)
    for file_name in files_name:
        if not os.path.isdir(data_path + file_name):
            best_snr_list=[]
            for tr_times in range(1, train_times+1):
                result_path = train(data_path + file_name, tr_times, dimension, nn_dropped_rate, 1-bernoulli_sampling_probability)
                best_snr_list = np.append(best_snr_list,pick_best_result(result_path,target_path,data_path+file_name,iterations,test_interval))
            print('---Results of the double ensemble learning strategy---')
            cal_ensemble_result(data_path+file_name,best_snr_list,task_name,target_path)




