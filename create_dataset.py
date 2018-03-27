import glob
import io
import tensorflow as tf
from PIL import Image
import random
import xml.etree.ElementTree as ET


flags = tf.app.flags
flags.DEFINE_string('output_train_path', 'data/tfrecord/train.tfrecord', 'Path to output TFRecord')
flags.DEFINE_string('output_eval_path', 'data/tfrecord/eval.tfrecord', 'Path to output TFRecord')
FLAGS = flags.FLAGS


def create_tf_example(image_file_path, anotate_xml_file_path):
  img = Image.open(image_file_path, 'r')
  height = int(img.height)
  width = int(img.width)
  filename = filename_without_extension(image_file_path).encode('utf-8')
  b = io.BytesIO()
  img.save(b, img.format)
  encoded_image_data = b.getvalue()
  image_format = img.format.encode('utf-8')
  with open(anotate_xml_file_path, 'r') as f:
    xmlstr = ''.join(f.readlines())
  root = ET.fromstring(xmlstr)
  data = root.find('object')
  
  xmins = [float(data.find('bndbox').find('xmin').text) / width]
  xmaxs = [float(data.find('bndbox').find('xmax').text) / width]
  ymins = [float(data.find('bndbox').find('ymin').text) / height]
  ymaxs = [float(data.find('bndbox').find('ymax').text) / height]
  classes_text = [data.find('name').text.encode('utf-8')]
  classes = [1]

  tf_example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
      'image/width': tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
      'image/filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[filename])),
      'image/source_id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[filename])),
      'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[encoded_image_data])),
      'image/format': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_format])),
      'image/object/bbox/xmin': tf.train.Feature(float_list=tf.train.FloatList(value=xmins)),
      'image/object/bbox/xmax': tf.train.Feature(float_list=tf.train.FloatList(value=xmaxs)),
      'image/object/bbox/ymin': tf.train.Feature(float_list=tf.train.FloatList(value=ymins)),
      'image/object/bbox/ymax': tf.train.Feature(float_list=tf.train.FloatList(value=ymaxs)),
      'image/object/class/text': tf.train.Feature(bytes_list=tf.train.BytesList(value=classes_text)),
      'image/object/class/label': tf.train.Feature(int64_list=tf.train.Int64List(value=classes)),
  }))
  return tf_example


def main(_):
  examples = _examples('./data/foot/*') + _examples('./data/footmirror/*')
  random.shuffle(examples)
  num_train = int(len(examples) * 0.8)
  train_examples = examples[:num_train]
  eval_examples = examples[num_train:]

  writer = tf.python_io.TFRecordWriter(FLAGS.output_train_path)
  for example in train_examples:
    tf_example = create_tf_example(example[0], example[1])
    writer.write(tf_example.SerializeToString())

  writer.close()

  writer = tf.python_io.TFRecordWriter(FLAGS.output_eval_path)
  for example in eval_examples:
    tf_example = create_tf_example(example[0], example[1])
    writer.write(tf_example.SerializeToString())

  writer.close()



def _examples(dirctory):
  allfiles = glob.glob(dirctory)
  xmls = list(filter(lambda s: '.xml' in s, allfiles))
  images = list(filter(lambda s: '.jpg' in s, allfiles))
  return [make_pair(xml, images) for xml in xmls]


def make_pair(x, lst):
  for e in lst:
    if filename_without_extension(x) == filename_without_extension(e):
      return (e, x)


def filename_without_extension(s):
  return s.split('/')[-1].split('.')[0]


def validate(pair):
  return filename_without_extension(pair[0]) == filename_without_extension(pair[1])


if __name__ == '__main__':
  tf.app.run()