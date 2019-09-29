# coding: utf-8

# In[1]:


import numpy as np
import tqdm
import pickle
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
# import tensorboard
from tensorboardX import SummaryWriter
# In[2]:
import os
import io
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
N_GPU = 2
flow_size = 300
is_training = 'y'
TRAINING = True if is_training == 'y' else False

# In[4]:


all_runs = {'8872': '192.168.122.117', '8802': '192.168.122.117', '8873': '192.168.122.67', '8803': '192.168.122.67',
            '8874': '192.168.122.113', '8804': '192.168.122.113', '8875': '192.168.122.120',
            '8876': '192.168.122.30', '8877': '192.168.122.208', '8878': '192.168.122.58'}

# In[6]:


dataset = []

for name in all_runs:
    dataset += pickle.load(open('./../%s_tordata300.pickle' % name, 'rb'))

if TRAINING:

    len_tr = len(dataset)
    train_ratio = float(len_tr - 6000) / float(len_tr)
    rr = list(range(len(dataset)))
    np.random.shuffle(rr)

    train_index = rr[:int(len_tr * train_ratio)]
    dev_index = rr[int(len_tr * train_ratio): -5000]  # range(len(dataset_test)) # #
    test_index = rr[-5000:]
    pickle.dump(test_index, open('test_index300.pickle', 'wb'))
else:
    test_index = pickle.load(open('test_index300.pickle', 'rb'))

# In[3]:


negetive_samples_train = 19
negetive_samples_test = 199


# In[4]:


def generate_data(dataset, data_index, flow_size, negetive_samples):
    all_samples = len(data_index)
    labels = np.zeros((all_samples * (negetive_samples + 1), 1), np.bool)
    l2s = np.zeros((all_samples * (negetive_samples + 1), 8, flow_size, 1), np.float32)

    index = 0
    random_ordering = [] + data_index
    # for i in tqdm.tqdm(data_index):
    for i in data_index:
        # []#list(lsh.find_k_nearest_neighbors((Y_train[i]/ np.linalg.norm(Y_train[i])).astype(np.float64),(50)))

        l2s[index, 0, :, 0] = np.array(dataset[i]['here'][0]['<-'][:flow_size]) * 1000.0
        l2s[index, 1, :, 0] = np.array(dataset[i]['there'][0]['->'][:flow_size]) * 1000.0
        l2s[index, 2, :, 0] = np.array(dataset[i]['there'][0]['<-'][:flow_size]) * 1000.0
        l2s[index, 3, :, 0] = np.array(dataset[i]['here'][0]['->'][:flow_size]) * 1000.0

        l2s[index, 4, :, 0] = np.array(dataset[i]['here'][1]['<-'][:flow_size]) / 1000.0
        l2s[index, 5, :, 0] = np.array(dataset[i]['there'][1]['->'][:flow_size]) / 1000.0
        l2s[index, 6, :, 0] = np.array(dataset[i]['there'][1]['<-'][:flow_size]) / 1000.0
        l2s[index, 7, :, 0] = np.array(dataset[i]['here'][1]['->'][:flow_size]) / 1000.0

        if index % (negetive_samples + 1) != 0:
            print(index)
            raise
        labels[index, 0] = 1
        m = 0
        index += 1
        np.random.shuffle(random_ordering)
        for idx in random_ordering:
            if idx == i or m > (negetive_samples - 1):
                continue

            m += 1

            l2s[index, 0, :, 0] = np.array(dataset[idx]['here'][0]['<-'][:flow_size]) * 1000.0
            l2s[index, 1, :, 0] = np.array(dataset[i]['there'][0]['->'][:flow_size]) * 1000.0
            l2s[index, 2, :, 0] = np.array(dataset[i]['there'][0]['<-'][:flow_size]) * 1000.0
            l2s[index, 3, :, 0] = np.array(dataset[idx]['here'][0]['->'][:flow_size]) * 1000.0

            l2s[index, 4, :, 0] = np.array(dataset[idx]['here'][1]['<-'][:flow_size]) / 1000.0
            l2s[index, 5, :, 0] = np.array(dataset[i]['there'][1]['->'][:flow_size]) / 1000.0
            l2s[index, 6, :, 0] = np.array(dataset[i]['there'][1]['<-'][:flow_size]) / 1000.0
            l2s[index, 7, :, 0] = np.array(dataset[idx]['here'][1]['->'][:flow_size]) / 1000.0

            # l2s[index,0,:,0]=Y_train[i]#np.concatenate((Y_train[i],X_train[idx]))#(Y_train[i]*X_train[idx])/(np.linalg.norm(Y_train[i])*np.linalg.norm(X_train[idx]))
            # l2s[index,1,:,0]=X_train[idx]

            labels[index, 0] = 0
            index += 1

    return l2s, labels


# In[5]:


import tensorflow as tf


# In[6]:


def model(flow_before, dropout_keep_prob):
    last_layer = flow_before
    flat_layers_after = [flow_size * 2, 1000, 50, 1]
    for l in range(len(flat_layers_after) - 1):
        flat_weight = tf.get_variable("flat_after_weight%d" % l, [flat_layers_after[l], flat_layers_after[l + 1]],
                                      initializer=tf.random_normal_initializer(stddev=0.01, mean=0.0))

        flat_bias = tf.get_variable("flat_after_bias%d" % l, [flat_layers_after[l + 1]],
                                    initializer=tf.zeros_initializer())

        _x = tf.add(
            tf.matmul(last_layer, flat_weight),
            flat_bias)
        if l < len(flat_layers_after) - 2:
            _x = tf.nn.dropout(tf.nn.relu(_x, name='relu_noise_flat_%d' % l), keep_prob=dropout_keep_prob)
        last_layer = _x
    return last_layer


# In[9]:


def model_cnn(flow_before, dropout_keep_prob):
    last_layer = flow_before

    CNN_LAYERS = [[2, 20, 1, 2000, 5], [4, 10, 2000, 800, 3]]

    for cnn_size in range(len(CNN_LAYERS)):
        cnn_weights = tf.compat.v1.get_variable("cnn_weight%d" % cnn_size, CNN_LAYERS[cnn_size][:-1],
                                                initializer=tf.random_normal_initializer(stddev=0.01))
        cnn_bias = tf.compat.v1.get_variable("cnn_bias%d" % cnn_size, [CNN_LAYERS[cnn_size][-2]],
                                             initializer=tf.zeros_initializer())

        _x = tf.nn.conv2d(last_layer, cnn_weights, strides=[1, 2, 2, 1], padding='VALID')
        _x = tf.nn.bias_add(_x, cnn_bias)
        conv = tf.nn.relu(_x, name='relu_cnn_%d' % cnn_size)
        pool = tf.nn.max_pool2d(conv, ksize=[1, 1, CNN_LAYERS[cnn_size][-1], 1], strides=[1, 1, 1, 1], padding='VALID')
        last_layer = pool
    last_layer = tf.reshape(last_layer, [batch_per_gpu, -1])

    flat_layers_after = [49600, 3000, 800, 100, 1]
    for l in range(len(flat_layers_after) - 1):
        flat_weight = tf.compat.v1.get_variable("flat_after_weight%d" % l,
                                                [flat_layers_after[l], flat_layers_after[l + 1]],
                                                initializer=tf.random_normal_initializer(stddev=0.01, mean=0.0))

        flat_bias = tf.compat.v1.get_variable("flat_after_bias%d" % l, [flat_layers_after[l + 1]],
                                              initializer=tf.zeros_initializer())

        _x = tf.add(
            tf.matmul(last_layer, flat_weight),
            flat_bias)
        if l < len(flat_layers_after) - 2:
            _x = tf.nn.dropout(tf.nn.relu(_x, name='relu_noise_flat_%d' % l), keep_prob=dropout_keep_prob)
        last_layer = _x
    return last_layer


# In[10]:

def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


if TRAINING:
    batch_size = 256
    batch_per_gpu = int(batch_size / N_GPU)
    learn_rate = 0.0001
    train_print_gap = 200
    dev_print_gap = 3000

    graph = tf.Graph()
    with graph.as_default():
        train_flow_before = tf.compat.v1.placeholder(tf.float32, shape=[batch_size, 8, flow_size, 1],
                                                     name='flow_before_placeholder')
        train_label = tf.compat.v1.placeholder(tf.float32, name='label_placeholder', shape=[batch_size, 1])
        dropout_keep_prob = tf.compat.v1.placeholder(tf.float32, name='dropout_placeholder')
        # train_correlated_var = tf.Variable(train_correlated, trainable=False)
        # Look up embeddings for inputs.

        # multi GPU
        optimizer = tf.compat.v1.train.AdamOptimizer(learn_rate)
        tower_grads = []
        y2_sum = None
        reuse_variables = False
        for i in range(N_GPU):
            with tf.device('/gpu:%d' % i):
                with tf.name_scope('GPU_%d' % i) as scope:
                    # y2 = model_cnn(train_flow_before[i*batch_size/N_GPU:(i+1)*batch_size/N_GPU], dropout_keep_prob)
                    with tf.compat.v1.variable_scope(tf.compat.v1.get_variable_scope(), reuse=reuse_variables):
                        y2 = model_cnn(train_flow_before[i * batch_per_gpu:(i + 1) * batch_per_gpu], dropout_keep_prob)
                    # Compute the average NCE loss for the batch.
                    # tf.nce_loss automatically draws a new sample of the negative labels each
                    # time we evaluate the loss.
                    # loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y2, labels=train_label[i*batch_size/N_GPU:(i+1)*batch_size/N_GPU]),
                    #                       name='loss_sigmoid')
                    reuse_variables = True
                    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y2, labels=train_label[
                                                                                                    i * batch_per_gpu:(
                                                                                                                              i + 1) * batch_per_gpu]),
                                          name='loss_sigmoid')
                    grads = optimizer.compute_gradients(loss)
                    tower_grads.append(grads)
                    if i == 0:
                        y2_sum = tf.identity(y2)
                    else:
                        y2_sum = tf.concat([y2_sum, y2], 0)
        grads = average_gradients(tower_grads)
        apply_gradient_op = optimizer.apply_gradients(grads)
        predict = tf.nn.sigmoid(y2_sum)



        # tp = tf.contrib.metrics.streaming_true_positives(predictions=tf.nn.sigmoid(logits), labels=train_correlated)
        # fp = tf.contrib.metrics.streaming_false_positives(predictions=tf.nn.sigmoid(logits), labels=train_correlated)

        # optimizer = tf.train.AdamOptimizer(learn_rate).minimize(loss)

        #    gradients = tf.norm(tf.gradients(logits, weights['w_out']))

        #    w_mean, w_var = tf.nn.moments(weights['w_out'], [0])
        # s_loss = tf.summary.scalar('loss', loss)
        #    tf.summary.scalar('weight_norm', tf.norm(weights['w_out']))
        #    tf.summary.scalar('weight_mean', tf.reduce_mean(w_mean))
        #    tf.summary.scalar('weight_var', tf.reduce_mean(w_var))

        #    tf.summary.scalar('bias', tf.reduce_mean(biases['b_out']))
        #    tf.summary.scalar('logits', tf.reduce_mean(logits))
        #    tf.summary.scalar('gradients', gradients)
        # summary_op = tf.summary.merge_all()

        # Add variable initializer.
        init = tf.global_variables_initializer()

        saver = tf.train.Saver(max_to_keep=None)


else:
    # batch_size = 2804 / 2
    batch_size = 256
    batch_per_gpu = 256
    graph = tf.Graph()
    with graph.as_default():
        train_flow_before = tf.placeholder(tf.float32, shape=[batch_size, 8, flow_size, 1],
                                           name='flow_before_placeholder')
        train_label = tf.placeholder(tf.float32, name='label_placeholder', shape=[batch_size, 1])
        dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_placeholder')
        # train_correlated_var = tf.Variable(train_correlated, trainable=False)
        # Look up embeddings for inputs.

        y2 = model_cnn(train_flow_before, dropout_keep_prob)
        predict = tf.nn.sigmoid(y2)
        # Compute the average NCE loss for the batch.
        # tf.nce_loss automatically draws a new sample of the negative labels each
        # time we evaluate the loss.
        saver = tf.train.Saver()

# In[ ]:


# In[18]:


num_epochs = 200
import datetime

# train_writer = tf.summary.FileWriter('./run/train/' + str(datetime.datetime.now()), graph=graph)
train_writer = SummaryWriter('runs/train/' + str(datetime.datetime.now()))
dev_writer = SummaryWriter('runs/dev/' + str(datetime.datetime.now()))


# In[ ]:


# Launch the graph
# with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
# with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
# saver = tf.train.Saver()
if TRAINING:
    import time

    config = tf.ConfigProto(allow_soft_placement=True)
    # config.gpu_options.per_process_gpu_memory_fraction = 0.5
    with tf.Session(graph=graph, config=config) as session:
        # We must initialize all variables before we use them.
        session.run(init)
        for epoch in range(num_epochs):
            gen_time_before = time.time()
            l2s, labels = generate_data(dataset=dataset, data_index=train_index, flow_size=flow_size,
                                                negetive_samples=negetive_samples_train)
            rr = list(range(len(l2s)))
            np.random.shuffle(rr)
            l2s = l2s[rr]
            labels = labels[rr]

            average_loss = 0
            new_epoch = True
            num_steps = (len(l2s) // batch_size) - 1
            gen_time_after = time.time()
            print("generate_data time:", gen_time_after - gen_time_before)

            for step in range(num_steps):
                start_time = time.time()
                start_ind = step * batch_size
                end_ind = ((step + 1) * batch_size)
                if end_ind < start_ind:
                    print('HOOY')
                    continue

                else:
                    batch_flow_before = l2s[start_ind:end_ind, :]
                    batch_label = labels[start_ind:end_ind]

                feed_dict = {train_flow_before: batch_flow_before,
                             train_label: batch_label,
                             dropout_keep_prob: 0.6}
                # We perform one update step by evaluating the optimizer op (including it
                # in the list of returned values for session.run()

                # _, loss_train, summary = session.run([apply_gradient_op, loss, summary_op], feed_dict=feed_dict)
                _, loss_train = session.run([apply_gradient_op, loss], feed_dict=feed_dict)
                # average_loss += loss_val
                # writer.add_summary(summary, (epoch * num_steps) + step)
                train_time = time.time() - start_time
                print("1 step using time:", train_time)
                # print step, loss_val
                # if step % FLAGS.print_every_n_steps == 0:
                #     if step > 0:
                #         average_loss /= FLAGS.print_every_n_steps
                #     # The average loss is an estimate of the loss over the last 2000 batches.
                #     print("Average loss at step ", step, ": ", average_loss)
                #     average_loss = 0.

                # Note that this is expensive (~20% slowdown if computed every 500 steps)

                if ((epoch * num_steps) + step) % train_print_gap == train_print_gap - 1:
                    predict_train = session.run(predict, feed_dict=feed_dict)
                    predict_labels = np.copy(predict_train)
                    predict_labels[predict_labels < 0.5] = 0
                    predict_labels[predict_labels >= 0.5] = 1
                    acc = accuracy_score(labels[start_ind:end_ind], predict_labels)
                    print("Average loss on training set at step ", (epoch * num_steps) + step, ": ", loss_train, "acc:", acc)
                    train_writer.add_scalar("loss", loss_train, (epoch * num_steps) + step)
                    train_writer.add_scalar("acc", acc, (epoch * num_steps) + step)
                    if np.max(labels[start_ind:end_ind]):
                        cm = confusion_matrix(labels[start_ind:end_ind], predict_labels)
                        # pr-curve
                        precision, recall, thresholds = precision_recall_curve(labels[start_ind:end_ind], predict_train)
                        pr_auc = auc(recall, precision)
                        # roc-curve
                        fpr, tpr, thresholds = roc_curve(labels[start_ind:end_ind], predict_train)
                        roc_auc = roc_auc_score(labels[start_ind:end_ind], predict_train)
                        # roc figure
                        fig = plt.figure()
                        plt.plot([0, 1], [0, 1], linestyle='--')
                        plt.plot(fpr, tpr, marker='.')
                        plt.xlabel("Fpr")
                        plt.ylabel("Tpr")
                        # confusion matrix figure
                        fig_cm = plt.figure()
                        cm_df = pd.DataFrame(cm, index=['false', 'true'], columns=['false', 'true'])
                        sns.heatmap(cm_df, annot=True, fmt="d")
                        plt.title('Accuracy:{0:.3f}'.format(acc))
                        plt.ylabel('True label')
                        plt.xlabel('Predicted label')
                        print(cm)
                        train_writer.add_scalar("pr_auc", pr_auc, (epoch * num_steps) + step)
                        train_writer.add_scalar("roc_auc", roc_auc, (epoch * num_steps) + step)
                        train_writer.add_figure("roc_curve", fig, (epoch * num_steps) + step)
                        train_writer.add_figure("cm", fig_cm, (epoch * num_steps) + step)
                        train_writer.add_pr_curve("pr_curve", labels[start_ind:end_ind], predict_train, (epoch * num_steps) + step)

                if ((epoch * num_steps) + step) % dev_print_gap == dev_print_gap - 1:
                    l2s_test, labels_test = generate_data(dataset=dataset, data_index=dev_index, flow_size=flow_size,
                                                    negetive_samples=negetive_samples_test)
                    test_time_before = time.time()
                    tp = 0
                    fp = 0
                    loss_sum = 0
                    num_steps_test = (len(l2s_test) // batch_size) - 1
                    Y_est = np.zeros((batch_size * (num_steps_test + 1), 1), np.float32)
                    for step_test in range(num_steps_test):
                        start_ind = step_test * batch_size
                        end_ind = ((step_test + 1) * batch_size)
                        test_batch_flow_before = l2s_test[start_ind:end_ind]
                        test_batch_label = labels_test[start_ind:end_ind]
                        test_batch_label = test_batch_label.reshape((-1, 1))
                        feed_dict = {
                            train_flow_before: test_batch_flow_before,
                            train_label: test_batch_label,
                            dropout_keep_prob: 1.0}

                        # est = session.run(predict, feed_dict=feed_dict)
                        est, loss_val = session.run([predict, loss], feed_dict=feed_dict)
                        loss_sum += loss_val
                        # est=np.array([xxx.sum() for xxx in test_batch_flow_before])
                        # Y_est[start_ind:end_ind] = est.reshape((batch_size))
                        Y_est[start_ind:end_ind] = est
                    Y_label = np.copy(Y_est)
                    Y_label[Y_label < 0.5] = 0
                    Y_label[Y_label >= 0.5] = 1

                    cm = confusion_matrix(labels_test[:len(Y_est)], Y_label)
                    acc = accuracy_score(labels_test[:len(Y_est)], Y_label)

                    print("Average loss on dev set at step ", (epoch * num_steps) + step, ": ",
                          loss_sum / num_steps_test, "acc:", acc)
                    print(cm)
                    # pr-curve
                    precision, recall, thresholds = precision_recall_curve(labels_test[:len(Y_est)], Y_est)
                    pr_auc = auc(recall, precision)
                    # roc-curve
                    fpr, tpr, thresholds = roc_curve(labels_test[:len(Y_est)], Y_est)
                    roc_auc = roc_auc_score(labels_test[:len(Y_est)], Y_est)
                    #roc figure
                    fig = plt.figure()
                    plt.plot([0, 1], [0, 1], linestyle='--')
                    plt.plot(fpr, tpr, marker='.')
                    plt.xlabel("Fpr")
                    plt.ylabel("Tpr")
                    # confusion matrix figure
                    fig_cm = plt.figure()
                    cm_df = pd.DataFrame(cm, index=['false', 'true'], columns=['false', 'true'])
                    sns.heatmap(cm_df, annot=True, fmt="d")
                    plt.title('Accuracy:{0:.3f}'.format(acc))
                    plt.ylabel('True label')
                    plt.xlabel('Predicted label')
                    dev_writer.add_scalar("loss", loss_sum / num_steps_test, (epoch * num_steps) + step)
                    dev_writer.add_scalar("acc", acc, (epoch * num_steps) + step)
                    dev_writer.add_scalar("pr_auc", pr_auc, (epoch * num_steps) + step)
                    dev_writer.add_scalar("roc_auc", roc_auc, (epoch * num_steps) + step)
                    dev_writer.add_figure("roc_curve", fig, (epoch * num_steps) + step)
                    dev_writer.add_figure("cm", fig_cm, (epoch * num_steps) + step)
                    dev_writer.add_pr_curve("pr_curve", labels_test[:len(Y_est)], Y_est, (epoch * num_steps) + step)
                    # num_samples_test = len(l2s_test) // (negetive_samples + 1)
                    num_samples_test = len(Y_est) // (negetive_samples_test + 1)
                    for idx in range(num_samples_test - 1):
                        best = np.argmax(Y_est[idx * (negetive_samples_test + 1):(idx + 1) * (negetive_samples_test + 1)])

                        if labels_test[best + (idx * (negetive_samples_test + 1))] == 1:
                            tp += 1
                        else:
                            fp += 1
                    print(tp, fp)
                    acc_2 = float(tp) / float(tp + fp)
                    print("acc_2:", acc_2)

                    print('saving...')
                    save_path = saver.save(session, "./model/%d.ckpt" % ((epoch * num_steps) + step))
                    print('saved')
                    test_time_after = time.time()
                    print("test time:", test_time_after - test_time_before)
        print('Epoch:', epoch)
        # save_path = saver.save(session, "/mnt/nfs/scratch1/milad/model_diff_large_1e4_epoch%d.ckpt"%(epoch))

        # t.join()
else:
    names = ["tor_199_epoch169_step779_acc0.57.ckpt", "tor_199_epoch174_step779_acc0.64.ckpt",
            "tor_199_epoch179_step779_acc0.67.ckpt", "tor_199_epoch183_step779_acc0.65.ckpt",
            "tor_199_epoch188_step779_acc0.58.ckpt", "tor_199_epoch193_step779_acc0.59.ckpt",
             "tor_199_epoch198_step779_acc0.67.ckpt"]
    for name in names:
        with tf.Session(graph=graph) as sess:
            # name = "tor_199_epoch38_step779_acc0.73.ckpt"
            saver.restore(sess, "./tor_model/%s" %name)
            print("model restored")
            test_all = np.zeros(((1 + negetive_samples) * len(test_index), 8, flow_size, 1))
            test_label = np.zeros(((1 + negetive_samples) * len(test_index), 1))
            index = 0
            for i in range(len(test_index)):
                test_all[index, 0, :, 0] = np.array(dataset[i]['here'][0]['<-'][:flow_size]) * 1000.0
                test_all[index, 1, :, 0] = np.array(dataset[i]['there'][0]['->'][:flow_size]) * 1000.0
                test_all[index, 2, :, 0] = np.array(dataset[i]['there'][0]['<-'][:flow_size]) * 1000.0
                test_all[index, 3, :, 0] = np.array(dataset[i]['here'][0]['->'][:flow_size]) * 1000.0

                test_all[index, 4, :, 0] = np.array(dataset[i]['here'][1]['<-'][:flow_size]) / 1000.0
                test_all[index, 5, :, 0] = np.array(dataset[i]['there'][1]['->'][:flow_size]) / 1000.0
                test_all[index, 6, :, 0] = np.array(dataset[i]['there'][1]['<-'][:flow_size]) / 1000.0
                test_all[index, 7, :, 0] = np.array(dataset[i]['here'][1]['->'][:flow_size]) / 1000.0

                test_label[index, 0] = 1
                index += 1
                for j in range(negetive_samples):
                    while True:
                        idx = np.random.randint(0, len(test_index))
                        if idx != i:
                            break
                    test_all[index, 0, :, 0] = np.array(dataset[idx]['here'][0]['<-'][:flow_size]) * 1000.0
                    test_all[index, 1, :, 0] = np.array(dataset[i]['there'][0]['->'][:flow_size]) * 1000.0
                    test_all[index, 2, :, 0] = np.array(dataset[i]['there'][0]['<-'][:flow_size]) * 1000.0
                    test_all[index, 3, :, 0] = np.array(dataset[idx]['here'][0]['->'][:flow_size]) * 1000.0

                    test_all[index, 4, :, 0] = np.array(dataset[idx]['here'][1]['<-'][:flow_size]) / 1000.0
                    test_all[index, 5, :, 0] = np.array(dataset[i]['there'][1]['->'][:flow_size]) / 1000.0
                    test_all[index, 6, :, 0] = np.array(dataset[i]['there'][1]['<-'][:flow_size]) / 1000.0
                    test_all[index, 7, :, 0] = np.array(dataset[idx]['here'][1]['->'][:flow_size]) / 1000.0

                    test_label[index, 0] = 0
                    index += 1
            num_steps_test = (len(test_all) // batch_size) - 1
            Y_est = np.zeros((batch_size * (num_steps_test + 1)))
            for step in range(num_steps_test):
                start_ind = step * batch_size
                end_ind = ((step + 1) * batch_size)
                test_batch_flow_before = test_all[start_ind:end_ind]
                test_batch_label = test_label[start_ind:end_ind]
                # test_batch_label = test_batch_label.reshape((-1, 1))
                feed_dict = {
                    train_flow_before: test_batch_flow_before,
                    train_label: test_batch_label,
                    dropout_keep_prob: 1.0}

                est = sess.run(predict, feed_dict=feed_dict)
                Y_est[start_ind:end_ind] = est.reshape((batch_size))
            Y_label = np.copy(Y_est)
            Y_label[Y_label < 0.5] = 0
            Y_label[Y_label > 0.5] = 1


            cm = confusion_matrix(test_label[:len(Y_est)], Y_label)
            acc = accuracy_score(test_label[:len(Y_est)], Y_label)
            # pr-curve
            precision, recall, thresholds = precision_recall_curve(test_label[:len(Y_est)], Y_est)
            pr_auc = auc(recall, precision)
            # roc-curve
            fpr, tpr, thresholds = roc_curve(test_label[:len(Y_est)], Y_est)
            roc_auc = roc_auc_score(test_label[:len(Y_est)], Y_est)
            print(cm)
            print("acc:%.3f    pr_auc:%.3f  roc_auc:%.3f   " % (acc, pr_auc, roc_auc))
            # plot no skill
            plt.figure(1)
            plt.plot([0, 1], [0.5, 0.5], linestyle='--')
            # plot the precision-recall curve for the model
            plt.plot(recall, precision, marker='.')
            plt.xlabel('recall')
            plt.ylabel('precision')
            # show the plot
            plt.savefig('./tor_model/pr_curve')
            plt.figure(2)
            plt.plot([0, 1], [0, 1], linestyle='--')
            # plot the roc curve for the model
            plt.plot(fpr, tpr, marker='.')
            plt.savefig('./tor_model/roc_curve')
            metrics_figure = [recall, precision, fpr, tpr]
            pickle.dump(metrics_figure, open('./tor_model/metrics_figure_%s.pkl' % name, 'wb'))
            print('dump %s metrics_figure.pkl' % name)
train_writer.close()
dev_writer.close()
