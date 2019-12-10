import flask
import json
from flask import request, jsonify
import numpy as np
import os
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import urllib.request
from flask import Flask, flash, request, redirect, render_template
from werkzeug.utils import secure_filename
import zipfile
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
# sys.path.append("..")
from object_detection.utils import ops as utils_ops
import tensorflow as tf
import glob
import os
import help




app = flask.Flask(__name__)
app.config["DEBUG"] = True

output_directory=""
pb_fname = os.path.join(os.path.abspath(output_directory), "frozen_inference_graph.pb")


UPLOAD_FOLDER = 'C:/Users/Varun Nair/PycharmProjects/twitter/object_detection_twitter/test'
repo_dir_path='C:/Users/Varun Nair/PycharmProjects/twitter/object_detection_twitter'
test_record_fname='C:/Users/Varun Nair/PycharmProjects/twitter/object_detection_twitter/data/annotations/test.record'
train_record_fname='C:/Users/Varun Nair/PycharmProjects/twitter/object_detection_twitter/data/annotations/train.record'
label_map_pbtxt_fname='C:/Users/Varun Nair/PycharmProjects/twitter/object_detection_twitter/data/annotations/label_map.pbtxt'
PATH_TO_CKPT = pb_fname
PATH_TO_LABELS = label_map_pbtxt_fname
PATH_TO_TEST_IMAGES_DIR =  os.path.join(repo_dir_path, "test")


#Assert statements commented
# assert os.path.isfile(pb_fname)
# assert os.path.isfile(PATH_TO_LABELS)

# for file in os.listdir(PATH_TO_TEST_IMAGES_DIR):
#     image = Image.open(PATH_TO_TEST_IMAGES_DIR+file)
#     print(os.path.join(PATH_TO_TEST_IMAGES_DIR, file))
# gen = glob.iglob(PATH_TO_TEST_IMAGES_DIR)
# TEST_IMAGE_PATHS=os.listdir(PATH_TO_TEST_IMAGES_DIR)




def absoluteFilePaths(directory):
   for dirpath,_,filenames in os.walk(directory):
       for f in filenames:
           yield os.path.abspath(os.path.join(dirpath, f))

TEST_IMAGE_PATHS =absoluteFilePaths(PATH_TO_TEST_IMAGES_DIR)

# assert len(TEST_IMAGE_PATHS) > 0, 'No image found in `{}`.'.format(PATH_TO_TEST_IMAGES_DIR)



#assert os.path.isfile(pb_fname), '`{}` not exist'.format(pb_fname)

# Create some test data for our catalog in the form of a list of dictionaries.



detection_graph = tf.Graph()
# num_classes=help.get
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)

categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=help.num_classes, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


IMAGE_SIZE = (13, 18)


def run_inference_for_single_image(image, graph):
    with graph.as_default():
        with tf.Session() as sess:
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {
                output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                        tensor_name)
            if 'detection_masks' in tensor_dict:
                detection_boxes = tf.squeeze(
                    tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(
                    tensor_dict['detection_masks'], [0])
                real_num_detection = tf.cast(
                    tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [
                    real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [
                    real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    detection_masks, detection_boxes, image.shape[0], image.shape[1])
                detection_masks_reframed = tf.cast(
                    tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                tensor_dict['detection_masks'] = tf.expand_dims(
                    detection_masks_reframed, 0)
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            output_dict = sess.run(tensor_dict,
                                   feed_dict={image_tensor: np.expand_dims(image, 0)})

            output_dict['num_detections'] = int(
                output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict[
                'detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict


def run_model():
    for image_path in TEST_IMAGE_PATHS:
        image = Image.open(image_path)
        image_np = load_image_into_numpy_array(image)
        image_np_expanded = np.expand_dims(image_np, axis=0)
        output_dict = run_inference_for_single_image(image_np, detection_graph)
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            output_dict['detection_boxes'],
            output_dict['detection_classes'],
            output_dict['detection_scores'],
            category_index,
            instance_masks=output_dict.get('detection_masks'),
            use_normalized_coordinates=True,
            line_thickness=5)
        plt.figure(figsize=IMAGE_SIZE)
        plt.imshow(image_np)
    boxes = output_dict['detection_boxes']
    max_boxes_to_draw = boxes.shape[0]
    scores = output_dict['detection_scores']
    min_score_thresh = .5
    bcoords = []
    for i in range(min(max_boxes_to_draw, boxes.shape[0])):
        if scores is None or scores[i] > min_score_thresh:
            bcoords.append(boxes[i].tolist())
    print(bcoords)
    return bcoords


# for image_path in TEST_IMAGE_PATHS:
#     image = Image.open(image_path)
#     image_np = load_image_into_numpy_array(image)
#     image_np_expanded = np.expand_dims(image_np, axis=0)
#     output_dict = run_inference_for_single_image(image_np, detection_graph)
#     vis_util.visualize_boxes_and_labels_on_image_array(
#         image_np,
#         output_dict['detection_boxes'],
#         output_dict['detection_classes'],
#         output_dict['detection_scores'],
#         category_index,
#         instance_masks=output_dict.get('detection_masks'),
#         use_normalized_coordinates=True,
#         line_thickness=5)
#     plt.figure(figsize=IMAGE_SIZE)
#     plt.imshow(image_np)
# boxes = output_dict['detection_boxes']
# max_boxes_to_draw = boxes.shape[0]
# scores = output_dict['detection_scores']
# min_score_thresh=.5
# bcoords=[]
# for i in range(min(max_boxes_to_draw, boxes.shape[0])):
#     if scores is None or scores[i] > min_score_thresh:
#
#         bcoords.append(boxes[i].tolist())
# print(bcoords)

###################################################

app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])
# print("---",bcoords)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET'])
def upload_pic():
    return render_template('upload.html')


# A route to return all of the available entries in our catalog.
@app.route('/api/home', methods=['POST'])
def show_coords():
    # print(type(bcoords))

    # print(bcoords)
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No file selected for uploading')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            flash('File successfully uploaded')


            # return json.dumps(run_model())
            return render_template('final.html',data=run_model())
            # return redirect('/api/home')
        else:
            flash('Allowed file types are txt, pdf, png, jpg, jpeg, gif')
            return redirect(request.url)



app.run()
