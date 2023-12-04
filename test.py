import os, sys, cv2, csv, argparse
import numpy as np
import keras
import tensorflow as tf

@tf.function
def AAR_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    mae_fun = tf.keras.losses.MeanAbsoluteError()
    mae = mae_fun(y_true, y_pred)

    condition = tf.less_equal(y_true, 0.1)
    indices = tf.where(condition)
    mae_1 = mae_fun(tf.gather(y_true, indices), tf.gather(y_pred, indices))
    mae_diff_1 = tf.square(mae_1-mae)

    condition1 = tf.less_equal(y_true, 0.2)
    condition2 = tf.greater(y_true, 0.1)
    condition = tf.logical_and(condition1, condition2)
    indices = tf.where(condition)
    mae_2 = mae_fun(tf.gather(y_true, indices), tf.gather(y_pred, indices))
    mae_diff_2 = tf.square(mae_2-mae)

    condition1 = tf.less_equal(y_true, 0.3)
    condition2 = tf.greater(y_true, 0.2)
    condition = tf.logical_and(condition1, condition2)
    indices = tf.where(condition)
    mae_3 = mae_fun(tf.gather(y_true, indices), tf.gather(y_pred, indices))
    mae_diff_3 = tf.square(mae_3-mae)

    condition1 = tf.less_equal(y_true, 0.4)
    condition2 = tf.greater(y_true, 0.3)
    condition = tf.logical_and(condition1, condition2)
    indices = tf.where(condition)
    mae_4 = mae_fun(tf.gather(y_true, indices), tf.gather(y_pred, indices))
    mae_diff_4 = tf.square(mae_4-mae)

    condition1 = tf.less_equal(y_true, 0.5)
    condition2 = tf.greater(y_true, 0.4)
    condition = tf.logical_and(condition1, condition2)
    indices = tf.where(condition)
    mae_5 = mae_fun(tf.gather(y_true, indices), tf.gather(y_pred, indices))
    mae_diff_5 = tf.square(mae_5-mae)

    condition1 = tf.less_equal(y_true, 0.6)
    condition2 = tf.greater(y_true, 0.5)
    condition = tf.logical_and(condition1, condition2)
    indices = tf.where(condition)
    mae_6 = mae_fun(tf.gather(y_true, indices), tf.gather(y_pred, indices))
    mae_diff_6 = tf.square(mae_6-mae)

    condition1 = tf.less_equal(y_true, 0.7)
    condition2 = tf.greater(y_true, 0.6)
    condition = tf.logical_and(condition1, condition2)
    indices = tf.where(condition)
    mae_7 = mae_fun(tf.gather(y_true, indices), tf.gather(y_pred, indices))
    mae_diff_7 = tf.square(mae_7-mae)

    condition = tf.greater(y_true, 0.7)
    indices = tf.where(condition)
    mae_8 = mae_fun(tf.gather(y_true, indices), tf.gather(y_pred, indices))
    mae_diff_8 = tf.square(mae_8-mae)

    mae_temp = tf.stack([mae_1, mae_2, mae_3, mae_4, mae_5, mae_6, mae_7, mae_8], axis=0)
    indices = tf.where(tf.logical_not(tf.math.is_nan(mae_temp)))
    mae_total = tf.gather(mae_temp, indices)
    mmae = tf.reduce_mean(mae_total)
    mae_diff_temp = tf.stack([mae_diff_1, mae_diff_2, mae_diff_3, mae_diff_4, mae_diff_5, mae_diff_6, mae_diff_7, mae_diff_8])
    variance_temp = tf.gather(mae_diff_temp, indices)
    variance = tf.sqrt(tf.reduce_mean(variance_temp))

    AAR_loss = 0.5*mmae + 0.5*variance

    return AAR_loss

@tf.function
def AAR_metric(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    mae_fun = tf.keras.losses.MeanAbsoluteError()
    mae = mae_fun(y_true, y_pred)
    
    condition = tf.less_equal(y_true, 0.1)
    indices = tf.where(condition) 
    mae_1 = mae_fun(tf.gather(y_true, indices), tf.gather(y_pred, indices))
    mae_diff_1 = tf.square(mae_1-mae)

    condition1 = tf.less_equal(y_true, 0.2)
    condition2 = tf.greater(y_true, 0.1)
    condition = tf.logical_and(condition1, condition2)
    indices = tf.where(condition)
    mae_2 = mae_fun(tf.gather(y_true, indices), tf.gather(y_pred, indices))
    mae_diff_2 = tf.square(mae_2-mae)

    condition1 = tf.less_equal(y_true, 0.3)
    condition2 = tf.greater(y_true, 0.2)
    condition = tf.logical_and(condition1, condition2)
    indices = tf.where(condition)
    mae_3 = mae_fun(tf.gather(y_true, indices), tf.gather(y_pred, indices))
    mae_diff_3 = tf.square(mae_3-mae)

    condition1 = tf.less_equal(y_true, 0.4)
    condition2 = tf.greater(y_true, 0.3)
    condition = tf.logical_and(condition1, condition2)
    indices = tf.where(condition)
    mae_4 = mae_fun(tf.gather(y_true, indices), tf.gather(y_pred, indices))
    mae_diff_4 = tf.square(mae_4-mae)

    condition1 = tf.less_equal(y_true, 0.5)
    condition2 = tf.greater(y_true, 0.4)
    condition = tf.logical_and(condition1, condition2)
    indices = tf.where(condition)
    mae_5 = mae_fun(tf.gather(y_true, indices), tf.gather(y_pred, indices))
    mae_diff_5 = tf.square(mae_5-mae)

    condition1 = tf.less_equal(y_true, 0.6)
    condition2 = tf.greater(y_true, 0.5)
    condition = tf.logical_and(condition1, condition2)
    indices = tf.where(condition)
    mae_6 = mae_fun(tf.gather(y_true, indices), tf.gather(y_pred, indices))
    mae_diff_6 = tf.square(mae_6-mae)

    condition1 = tf.less_equal(y_true, 0.7)
    condition2 = tf.greater(y_true, 0.6)
    condition = tf.logical_and(condition1, condition2)
    indices = tf.where(condition)
    mae_7 = mae_fun(tf.gather(y_true, indices), tf.gather(y_pred, indices))
    mae_diff_7 = tf.square(mae_7-mae)

    condition = tf.greater(y_true, 0.7)
    indices = tf.where(condition)
    mae_8 = mae_fun(tf.gather(y_true, indices), tf.gather(y_pred, indices))
    mae_diff_8 = tf.square(mae_8-mae)

    mae_temp = tf.stack([mae_1, mae_2, mae_3, mae_4, mae_5, mae_6, mae_7, mae_8], axis=0)
    indices = tf.where(tf.logical_not(tf.math.is_nan(mae_temp)))
    mae_total = tf.gather(mae_temp, indices)
    mmae = tf.reduce_mean(mae_total)
    mae_diff_temp = tf.stack([mae_diff_1, mae_diff_2, mae_diff_3, mae_diff_4, mae_diff_5, mae_diff_6, mae_diff_7, mae_diff_8])
    variance_temp = tf.gather(mae_diff_temp, indices)
    variance = tf.sqrt(tf.reduce_mean(variance_temp))

    #𝐴𝐴𝑅 = max(0; 5 − 𝑚𝑀𝐴𝐸) + max(0; 5 − 𝜎)
    AAR = tf.math.maximum(tf.zeros((1,)),tf.constant(5, dtype=tf.float32)-mmae*100)+tf.math.maximum(tf.zeros((1,)),tf.constant(5, dtype=tf.float32)-variance*100)

    return AAR


def init_parameter():   
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument("--data", type=str, default='foo_test.csv', help="Dataset labels")
    parser.add_argument("--images", type=str, default='foo_test/', help="Dataset folder")
    parser.add_argument("--results", type=str, default='foo_results.csv', help="CSV file of the results")
    args = parser.parse_args()
    return args

args = init_parameter()

# Reading CSV test file
with open(args.data, mode='r') as csv_file:
    gt = csv.reader(csv_file, delimiter=',')
    gt_num = 0
    gt_dict = {}
    for row in gt:
        gt_dict.update({row[0]: int(round(float(row[1])))})
        gt_num += 1
print(gt_num)

# Loading model
model = keras.models.load_model('xception09_final', custom_objects={'AAR_metric': AAR_metric, 'AAR_loss': AAR_loss}, compile=False)
feature_extractor = keras.Model(inputs=model.layers[0].input, outputs=model.layers[2].output)
regressor = keras.models.load_model('xception_double_finetuning', custom_objects={'AAR_metric': AAR_metric, 'AAR_loss': AAR_loss}, compile=False)
img_width, img_height = 299, 299

# Opening CSV results file
with open(args.results, 'w', newline='') as res_file:
    writer = csv.writer(res_file)
    # Processing all the images
    for image in gt_dict.keys():
        img = tf.keras.preprocessing.image.load_img(args.images+image)
        if img.size == 0:
            print("Error")
        # Preprocessing image
        img = tf.keras.preprocessing.image.img_to_array(img)
        img = tf.image.resize(img,(img_width, img_height)).numpy()
        img = img/255.
        img = np.expand_dims(img, 0)
        
        # Here you should add your code for applying your DCNN
        feature_vector = feature_extractor.predict(img)
        age = regressor.predict(feature_vector)
        # Cast age to integer
        age = round(float(age[0])*100)
        # Writing a row in the CSV file
        writer.writerow([image, age])