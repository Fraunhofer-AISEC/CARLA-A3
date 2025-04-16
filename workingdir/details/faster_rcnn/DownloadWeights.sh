MODEL_NAME="faster_rcnn_inception_v2_coco_2017_11_08"
MODEL_FILE="${MODEL_NAME}.tar.gz"
DOWNLOAD_BASE='http://download.tensorflow.org/models/object_detection/'
wget "${DOWNLOAD_BASE}${MODEL_FILE}"
tar -xvz -f $MODEL_FILE --wildcards "*frozen_inference_graph.pb"