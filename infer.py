import tensorflow as tf
import skimage
import sys

def load_img(path):
    # load image
    img = skimage.io.imread(path)
    img = img / 255.0
    assert (0 <= img).all() and (img <= 1.0).all()
    # print "Original Image Shape: ", img.shape
    # we crop image from center
    # resize to 224, 224
    resized_img = skimage.transform.resize(img, (224, 224))
    return [resized_img]

print('input img: ', sys.argv[1])
batch = load_img(sys.argv[1])

sess=tf.Session() 
saver = tf.train.import_meta_graph('vgg19-model/model.ckpt.meta')
saver.restore(sess,tf.train.latest_checkpoint('vgg19-model/'))

graph = tf.get_default_graph()
prob = graph.get_tensor_by_name("prob:0")
images = graph.get_tensor_by_name("images:0")
mode = graph.get_tensor_by_name("mode:0")

logits=sess.run(prob,feed_dict={images:batch,mode:False})
print('logits: ', logits)
if logits[0][0]>0.5:
  print('normal img')
  exit(1)
else:
  print('defect imag')
  exit(2)


