import numpy as np
import random
from math import log10
from sklearn.metrics import mean_squared_error


def print_parameter_information(learning_rate,nn_dropped_rate,bernoulli_sampling_probability,num_ensemble_models,train_times,task_name):
    print('----------------------------------------------------')
    print('Task name:',task_name)
    print('Learning rate:', learning_rate)
    print('Dropped rate of dropouts in 3-DPCNN:', nn_dropped_rate)
    print('Bernoulli sampling probability:',bernoulli_sampling_probability)
    print('Number of ensemble models:',num_ensemble_models)
    print('3-DPCNN is trained ', train_times,' times from scratch')
    print('----------------------------------------------------')

def load_np_seismic(path, dim):
    seismic=np.fromfile(path,dtype=np.float32)
    seismic=seismic.reshape(dim)
    seismic = np.expand_dims(seismic, axis=0)        # add batch_size=1 for 3D data --> [1, Xline, Inline, Time]
    seismic = np.expand_dims(seismic, axis=4)        # add channel=1 for 3D data    --> [1, Xline, Inline, Time, 1]
    seismic=seismic.transpose(0, 3, 1, 2, 4)         # [batch_size=1, depth=Time, height=Xline, width=Inline, channel=1]
    return seismic

def read_data(path,dim,result_path):
    missed_volume   = load_np_seismic(path + 'missed_volume.bin', dim)
    missed_position = load_np_seismic(path + 'missed_position.bin',dim)
    np.squeeze(missed_position.transpose(0, 2, 3, 1, 4)).tofile(result_path + 'missed_position.bin')
    np.squeeze(missed_volume.transpose(0, 2, 3, 1, 4)).tofile(result_path + 'missed_volume.bin')

    return missed_volume, missed_position


def miss_trace(volume, model_path,missing_type,missing_coefficient):
    missed_volume = volume.copy()
    missed_position = np.ones_like(missed_volume)

    if missing_type=='RM':
        interval = np.int32(np.around(1.0/missing_coefficient))

        for x in range(np.int32(interval/2), np.shape(volume)[3], interval):
            missed_volume[:, :, :, x, :] = 0
            missed_position[:, :, :, x, :] = 0
        for y in range(np.int32(interval/2), np.shape(volume)[2], interval):
            missed_volume[:, :, y, :, :] = 0
            missed_position[:, :, y, :, :] = 0
    elif missing_type=='CM':
        # start_pos=np.random.randint(1,5)
        start_pos = 4
        interval = 4
        for x in range(start_pos, np.shape(volume)[3], missing_coefficient + interval):
            if x + missing_coefficient < np.shape(volume)[3] - 1:
                for y in range(x, x + missing_coefficient, 1):
                    missed_volume[:, :, :, y, :] = 0
                    missed_position[:, :, :, y, :] = 0
            else:
                pass
    elif missing_type=='IM':
        all_index = [i for i in range(np.shape(volume)[2] * np.shape(volume)[3])]
        miss_id = random.sample(all_index, np.int32(np.around(np.shape(volume)[2] * np.shape(volume)[3] * missing_coefficient)))
        missed_volume = missed_volume.reshape(np.shape(volume)[0], np.shape(volume)[1],
                                              np.shape(volume)[2]*np.shape(volume)[3], np.shape(volume)[4])
        missed_position = missed_position.reshape(np.shape(volume)[0], np.shape(volume)[1],
                                              np.shape(volume)[2] * np.shape(volume)[3], np.shape(volume)[4])

        for xy in miss_id:
            missed_volume[:, :, xy, :] = 0
            missed_position[:, :, xy, :] = 0

        missed_volume = missed_volume.reshape(np.shape(volume)[0], np.shape(volume)[1],
                                              np.shape(volume)[2], np.shape(volume)[3], np.shape(volume)[4])
        missed_position = missed_position.reshape(np.shape(volume)[0], np.shape(volume)[1],
                                        np.shape(volume)[2], np.shape(volume)[3], np.shape(volume)[4])
    else:
        pass
    np.squeeze(missed_position.transpose(0, 2, 3, 1, 4)).tofile(model_path + 'missed_position.bin')
    np.squeeze(missed_volume.transpose(0, 2, 3, 1, 4)).tofile(model_path + 'missed_volume.bin')

    return missed_volume, missed_position

def cal_snr(prediction, target):
    prediction = np.array(prediction)
    target     = np.array(target)
    zero       = np.zeros_like(target)
    MSE        = mean_squared_error(target, prediction)
    total      = mean_squared_error(zero, target)
    snr        = 10. * log10(total.item() / MSE.item())
    return snr

# Picking out the best denoised or reconstructed data
def pick_best_result(result_path, target_path, ori_path, iterations, test_interval):
    target = np.fromfile(target_path, dtype=np.float32)
    ori  = np.fromfile(ori_path, dtype=np.float32)
    snr_list = []
    for item in range(test_interval, iterations+1, test_interval):
        result = np.fromfile(result_path+ '3-DPCNN_' + str(item) + '.bin',dtype=np.float32)
        snr_list = np.append(snr_list, cal_snr(result, target))
    max_index=(snr_list.tolist().index(max(snr_list)) + 1) * test_interval
    print('Ori S/N: %.2fdB -- Max S/N from %d iterations: %.2fdB ' % (cal_snr(ori, target),max_index, max(snr_list)))
    return max_index

def  cal_ensemble_result(path, best_snr_list,task_name,target_path):
    target = np.fromfile(target_path, dtype=np.float32)
    for current_train in range(1,len(best_snr_list)+1):
        result = np.zeros_like(target)
        for item in range(current_train):
            result_path_ = path[0:path.rfind(".")] + "/" + task_name + "_train" + str(item+1) + "/"
            result_path = result_path_ + '3-DPCNN_' + str(np.int32(best_snr_list[item])) + '.bin'
            result = result + np.fromfile(result_path, dtype=np.float32)
        result = result / current_train
        result.tofile(result_path_ + '3-DPCNN_K'+str(current_train)+'.bin')
        print('K= %d : %.2fdB' %(current_train,cal_snr(result,target)))






