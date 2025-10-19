from leap_binder import *
import tensorflow as tf
import os
import numpy as np
import onnxruntime
from keras.losses import BinaryCrossentropy
from code_loader.contract.datasetclasses import PredictionTypeHandler
from code_loader.inner_leap_binder.leapbinder_decorators import tensorleap_load_model, tensorleap_integration_test
from code_loader.default_losses import categorical_crossentropy

prediction_type1 = PredictionTypeHandler('sentiment', ['positive', 'negative'])

@tensorleap_load_model([prediction_type1])
def load_model():
    dir_path = os.path.dirname(os.path.abspath(__file__))
    model_path = 'model/fabriceyhc-bert-imdb.onnx'
    sess = onnxruntime.InferenceSession(os.path.join(dir_path, model_path))
    return sess

@tensorleap_integration_test()
def check_custom_test(idx, responses_set):
    print("started custom tests")
    input__id = input_ids(idx, responses_set)
    attention__mask = attention_masks(idx, responses_set)
    token_type__id = token_type_ids(idx, responses_set)

    # get input and gt
    gt = gt_sentiment(idx, responses_set)
    #y_true = tf.convert_to_tensor(np.expand_dims(gt, axis=0))

    # model
    sess = load_model()
    # get inputs
    input_name_1 = sess.get_inputs()[0].name #TODO solve
    input_name_2 = sess.get_inputs()[1].name #TODO solve
    input_name_3 = sess.get_inputs()[2].name #TODO solve
    y_pred = sess.run(None, {input_name_1: input__id,
                                     input_name_2: attention__mask,
                                     input_name_3: token_type__id})[0]
    # get loss
    ls = ce_loss(gt, y_pred)

    # get meatdata
    gt_mdata = gt_metadata(idx, responses_set)
    all_raw_md = all_raw_metadata(idx, responses_set)

    # get visualizer
    horizontal_bar_visualizer_with_labels_name(y_pred)
    horizontal_bar_visualizer_with_labels_name(gt)
    text_visualizer_func(input__id)


def check_custom_test_dense():
    print("started custom tests")
    responses = preprocess_func()
    train = responses[0]
    val = responses[1]
    responses_set = train
    dir_path = os.path.dirname(os.path.abspath(__file__))
    model_path = 'model/imdb-dense.h5'

    for idx in range(20):
        # get input and gt

        input__tokens = input_tokens(idx, responses_set)
        concat = np.expand_dims(input__tokens, axis=0)

        gt = gt_sentiment(idx, responses_set)
        y_true = tf.convert_to_tensor(np.expand_dims(gt, axis=0))

        # model
        model = tf.keras.models.load_model(os.path.join(dir_path, model_path))
        y_pred = model([concat])

        # get loss
        ls = BinaryCrossentropy()(y_true, y_pred)

        # get meatdata
        gt_mdata = gt_metadata(idx, responses_set)
        all_raw_md = all_raw_metadata(idx, responses_set)

        # get visualizer
        horizontal_bar_visualizer_with_labels_name(y_pred.numpy())
        horizontal_bar_visualizer_with_labels_name(np.expand_dims(np.array(gt, dtype=np.float32), 0))
        text_visualizer_func_dense_model(np.expand_dims(input__tokens, 0))

        labels_names = [CONFIG['LABELS_NAMES'][index] for index in range(y_pred.shape[-1])]

        # text_gt_visualizer_func
        # ohe = {"pos": [0., 1.0], "neg": [1.0, 0.]}
        ohe = {"pos": [1.0, 0.], "neg": [0., 1.0]}
        text = []
        if (y_true[0].numpy() == np.array(ohe["pos"])).all():
            text.append("pos")
        else:
            text.append("neg")

    print("finish tests")


if __name__ == '__main__':
    responses = preprocess_func()
    train = responses[0]
    check_custom_test(0, train)
