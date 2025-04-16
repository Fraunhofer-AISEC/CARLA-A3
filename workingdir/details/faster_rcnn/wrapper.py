import numpy as np
from details.ObjectDetectorBase import ObjectDetectorBase
from object_detection.utils import label_map_util
from details.metric import bbox_iou

tf = None

class FasterRCNN(ObjectDetectorBase):
    def __init__(
        self,
        cudaFlag = False,
        frozenGraphFile = "details/faster_rcnn/faster_rcnn_inception_v2_coco_2017_11_08/frozen_inference_graph.pb",
        labelPath = "details/faster_rcnn/faster_rcnn_inception_v2_coco_2017_11_08/mscoco_label_map.pbtxt",
        inputWidth = 300,
        inputHeight = 300,
    ) -> None:
        super().__init__()
        if cudaFlag == False:
            import os
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        import tensorflow
        global tf
        tf = tensorflow
        self.label_map = label_map_util.load_labelmap(labelPath)
        self.categories = label_map_util.convert_label_map_to_categories(
            self.label_map,
            max_num_classes=len(self.label_map.item),
            use_display_name=True
        )
        self.categoryIndex = label_map_util.create_category_index(self.categories)
        self.inferenceGraph = tf.Graph()
        self.initInferenceGraph(frozenGraphFile, inputWidth, inputHeight)

    def initInferenceGraph(
        self,
        frozenGraphFile,
        inputWidth,
        inputHeight
    ):
        with self.inferenceGraph.as_default():
            image_tensor = tf.placeholder(tf.float32, shape=(None, inputWidth, inputHeight, 3), name='image_tensor')
            inference_graph_def = tf.GraphDef()
            with tf.gfile.GFile(frozenGraphFile, 'rb') as fid:
                serialized_graph = fid.read()
                inference_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(inference_graph_def, name='',
                                    input_map={'Preprocessor/map/TensorArrayStack/TensorArrayGatherV3:0':image_tensor})
        
        self.inferenceSess = tf.Session(graph = self.inferenceGraph)
        self.tensors = [
            self.inferenceGraph.get_tensor_by_name('detection_boxes:0'),
            self.inferenceGraph.get_tensor_by_name('detection_scores:0'),
            self.inferenceGraph.get_tensor_by_name('detection_classes:0'),
            self.inferenceGraph.get_tensor_by_name('num_detections:0'),
            self.inferenceGraph.get_tensor_by_name('SecondStagePostprocessor/Reshape_4:0'),
            self.inferenceGraph.get_tensor_by_name('SecondStagePostprocessor/convert_scores:0')
        ]

        self.inputTensor = self.inferenceGraph.get_tensor_by_name('image_tensor:0')

    def detect(
        self,
        img,
        scoreThreshold = 0.5,
        bboxIOUthreshold = 0.4
    ):
        feed_dict = { self.inputTensor: np.expand_dims(img, axis=0) }

        nms_bboxes, nms_scores, nms_classes, num_detections, bboxes, scores = self.inferenceSess.run(self.tensors, feed_dict)
        
        bboxes = bboxes[0]
        scores = scores[0]

        sorted_classes = np.argsort(scores[:, 1:], axis=1)
        sorted_scores = scores[:, 1:].copy()
        sorted_bboxes = bboxes.copy()

        for i, ordering in enumerate(sorted_classes):
            sorted_scores[i, :] = scores[i, ordering+1]
            sorted_bboxes[i, :] = bboxes[i, ordering, :]

        bboxes = sorted_bboxes[:, -1, :]
        classes = sorted_classes[:, -1].astype(np.int32)
        scores_array = scores[:,1:]
        scores = sorted_scores[:, -1]
        

        if classes.ndim == 1:
            # TF category index starts from 1
            classname = [self.categoryIndex.get(i+1, {}).get("name", "unknown") for i in classes] 
            classname = np.array(classname)
        else:
            # TF category index starts from 1
            classname = [self.categoryIndex.get(j+1, {}).get("name", "unknown") for i in classes for j in i] 
            classname = np.reshape(classname, classes.shape)
        
        # Thresholding
        idx = scores > scoreThreshold
        if idx.ndim == 2:
            idx = (np.sum(idx, axis=1) == 1)
        
        bboxes, classes, scores, classname, scores_array = self.rejectOverlappingBbox(
            bboxes[idx],
            classes[idx],
            scores[idx],
            classname[idx],
            scores_array[idx],
            bboxIOUthreshold
        )

        return bboxes, classes, scores, classname, scores_array


    def rejectOverlappingBbox(self, bboxes, classes, scores, classname, scores_array, bboxIOUthreshold):
        if scores.ndim == 2:
            scores = scores[:,-1]

        # Sort descendingly
        sortedIdx = np.argsort(-scores) 
        bboxes, classes, scores, scores_array, classname = bboxes[sortedIdx], classes[sortedIdx], scores[sortedIdx], scores_array[sortedIdx], classname[sortedIdx]
        
        newBboxes, newClasses, newScores, newScoresArray, newClassname = [], [], [], [], []

        for i in range(0, len(scores)):
            if scores[i] > 0:
                newBboxes.append(bboxes[i])
                newClasses.append(classes[i])
                newScores.append(scores[i])
                newScoresArray.append(scores_array[i])
                newClassname.append(classname[i])

                for j in range(i+1, len(scores)):
                    if bbox_iou(bboxes[i], bboxes[j]) >= bboxIOUthreshold:
                        scores[j] = 0

        return np.array(newBboxes), np.array(newClasses), np.array(newScores), np.array(newClassname), np.array(newScoresArray)