"""
Simple tester for the vgg19_trainable
"""

import tensorflow as tf

import vgg19_bin_trainable as vgg19
import utils
import gc
import time
import random

batch,label = utils.load_files("./test_data/chip-sample/train")
batch2,label2 = utils.load_files("./test_data/chip-sample/test")
sample_count = len(batch)
#img1_true_result = [1 if i == 292 else 0 for i in range(1000)]  # 1-hot result for tiger
#img1_true_result = [[0] for i in range(sample_count)]  # 1-hot result for tiger
img1_true_result = label
slots = [i for i in range(0,sample_count) ]

def print_pred(prob,gt,flag):
  pred_label = []
  for p in prob:
    if p > 0.5:
      pred_label.append(1)
    else:
      pred_label.append(0)
  flat_gt = sum(gt,[])
  res = list(zip(pred_label,flat_gt))
  corrects = 0
  for item in res:
#   print(item)
    if item[0] == item[1]:
      corrects += 1
  acc = corrects/len(gt)
  print(flag,' acc: ', acc)
  return acc

with tf.device('/cpu:0'):
    sess = tf.Session()

#   images = tf.placeholder(tf.float32, [sample_count, 224, 224, 3])
    images = tf.placeholder(tf.float32, [None, 224, 224, 3],name='images')
#   true_out = tf.placeholder(tf.float32, [1, 1000])
#   true_out = tf.placeholder(tf.float32, [sample_count, 1])
    true_out = tf.placeholder(tf.float32, [None, 1])
    train_mode = tf.placeholder(tf.bool,name='mode')

    vgg = vgg19.Vgg19('./vgg19.npy')
    vgg.build(images, train_mode)

    # print number of variables used: 143667240 variables, i.e. ideal size = 548MB
    print(vgg.get_var_count())

    # simple 1-step training
    cost = tf.reduce_sum((vgg.prob - true_out) ** 2)

    #reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    #print('reg_variables ',reg_variables)
    #reg_term = tf.contrib.layers.apply_regularization(vgg19.regularizer, reg_variables)
    reg_term=tf.losses.get_regularization_loss()
    cost += reg_term
    #train = tf.train.GradientDescentOptimizer(0.0001).minimize(cost)
    train = tf.train.AdamOptimizer(1e-4).minimize(cost)
    
    sess.run(tf.global_variables_initializer())

    # test classification
    print('before train')
    prob = sess.run(vgg.prob, feed_dict={images: batch, train_mode: False})
    print_pred(prob, label,'train')
    prob = sess.run(vgg.prob, feed_dict={images: batch2, train_mode: False})
    print_pred(prob, label2, 'test')
    
    saver = tf.train.Saver()

    for i in range(100):
      batch_aug = []
      for img in batch:
        batch_aug.append(utils.aug_img(img))
      gc.collect()
      print('epoch ', i)
      random.shuffle(slots)
      batch_size=64
      mini_batch = []
      mini_label = []
      counts = 0
      for slot in slots:
        mini_batch.append(batch_aug[slot])
        mini_label.append(label[slot])
        counts += 1
        if counts == batch_size:
          sess.run(train, feed_dict={images: mini_batch, true_out: mini_label, train_mode: True})
          print('run mini batch train once')
          mini_batch = []
          mini_label = []
          counts = 0
      if counts > 0:
        sess.run(train, feed_dict={images: mini_batch, true_out: mini_label, train_mode: True})
    #sess.run(train, feed_dict={images: batch_aug, true_out: img1_true_result, train_mode: True})
    # test classification again, should have a higher probability about tiger
      loss_train = sess.run(cost, feed_dict={images: batch, true_out: label, train_mode: False})
      print('train loss ', loss_train)
      loss_test = sess.run(cost, feed_dict={images: batch2, true_out: label2, train_mode: False})
      print('test loss ', loss_test)
      train_prob = sess.run(vgg.prob, feed_dict={images: batch, train_mode: False})
      train_acc = print_pred(train_prob, label,'train')
      test_prob = sess.run(vgg.prob, feed_dict={images: batch2, train_mode: False})
      test_acc = print_pred(test_prob, label2,'test')

      if train_acc>0.9 and test_acc>0.85:
          save_path = saver.save(sess, "vgg19-model/model.ckpt")
          print('stop training, save model')
          break

    # test save
