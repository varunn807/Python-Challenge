import os
UPLOAD_FOLDER = 'C:/Users/Varun Nair/PycharmProjects/twitter/object_detection_twitter/test'
pipeline_fname='C:/Users/Varun Nair/PycharmProjects/twitter/models/research/object_detection/samples/configs/faster_rcnn_inception_v2_pets.config'
repo_dir_path='C:/Users/Varun Nair/PycharmProjects/twitter/object_detection_twitter'
test_record_fname='C:/Users/Varun Nair/PycharmProjects/twitter/object_detection_twitter/data/annotations/test.record'
train_record_fname='C:/Users/Varun Nair/PycharmProjects/twitter/object_detection_twitter/data/annotations/train.record'
label_map_pbtxt_fname='C:/Users/Varun Nair/PycharmProjects/twitter/object_detection_twitter/data/annotations/label_map.pbtxt'
fine_tune_checkpoint='C:/Users/Varun Nair/PycharmProjects/twitter/models/research/pretrained_model/model.ckpt'
repo_url = 'https://github.com/JATINBHORE/object_detection_twitter'
num_steps = 800
num_eval_steps = 50
MODELS_CONFIG = {
    'faster_rcnn_inception_v2': {
        'model_name': 'faster_rcnn_inception_v2_coco_2018_01_28',
        'pipeline_file': 'faster_rcnn_inception_v2_pets.config',
        'batch_size': 12
    }
}
selected_model = 'faster_rcnn_inception_v2'
MODEL = MODELS_CONFIG[selected_model]['model_name']
pipeline_file = MODELS_CONFIG[selected_model]['pipeline_file']
batch_size = MODELS_CONFIG[selected_model]['batch_size']
output_directory=""
pb_fname = os.path.join(os.path.abspath(output_directory), "frozen_inference_graph.pb")
