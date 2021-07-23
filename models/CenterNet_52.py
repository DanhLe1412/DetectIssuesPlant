from tensorflow.keras import layers
import tensorflow as tf

from utils.loss import neg_loss, regr_loss, ae_loss
from utils.module import hourglass_module, base_module, cascade_br_pool, cascade_tl_pool, center_pool
from utils.module import heat_layer
import tensorflow.keras.models 


def head(inputs, fouts, nbclass):
    out = []
    x_tl = cascade_tl_pool(inputs, fouts)
    x_br = cascade_br_pool(inputs, fouts)
    x_center = center_pool(inputs, fouts)

    tl_heatmap = heat_layer(x_tl, fouts, nbclass)
    tl_heatmap = layers.Activation('sigmoid')(tl_heatmap)
    out.append(tl_heatmap)

    br_heatmap = heat_layer(x_br, fouts, nbclass)
    br_heatmap = layers.Activation('sigmoid')(br_heatmap)
    out.append(br_heatmap)

    center_heatmap = heat_layer(x_center, fouts, nbclass)
    center_heatmap = layers.Activation('sigmoid')(center_heatmap)
    out.append(center_heatmap)

    tl_tag = heat_layer(x_tl, fouts, 1)
    tl_tag = layers.Flatten()(tl_tag)
    out.append(tl_tag)

    br_tag = heat_layer(x_br, fouts, 1)
    br_tag = layers.Flatten()(br_tag)
    out.append(br_tag)

    tl_reg = heat_layer(x_tl, fouts, 2)
    tl_reg = layers.Activation('sigmoid')(tl_reg)
    out.append(tl_reg)
    br_reg = heat_layer(x_br, fouts, 2)
    br_reg = layers.Activation('sigmoid')(br_reg)
    out.append(br_reg)
    cent_reg = heat_layer(x_center, fouts, 2)
    cent_reg = layers.Activation('sigmoid')(cent_reg)
    out.append(cent_reg)

    return out


class CenterNet52():
    def __init__(self, config, weight_dir):
        self.config = config
        self.weight_dir = weight_dir
        self.model = self.build()

    def build(self):

        # Inputs
        input_image = layers.Input(
            shape=self.config.VIEW_SIZE+[3], name="input_image")
        input_image_meta = layers.Input(shape=[self.config.MAX_NUMS, self.config.META_SHAPE],
                                    name="input_image_meta")

        input_tl_heatmaps = layers.Input(
            shape=self.config.OUTPUT_SIZE+[self.config.CLASSES], name="input_tl_heatmaps", dtype=tf.float32)
        input_br_heatmaps = layers.Input(
            shape=self.config.OUTPUT_SIZE+[self.config.CLASSES], name="input_br_heatmaps", dtype=tf.float32)
        input_ct_heatmaps = layers.Input(
            shape=self.config.OUTPUT_SIZE+[self.config.CLASSES], name="input_ct_heatmaps", dtype=tf.float32)

        input_tl_reg = layers.Input(
            shape=self.config.OUTPUT_SIZE+[2], name="input_tl_reg", dtype=tf.float32)
        input_br_reg = layers.Input(
            shape=self.config.OUTPUT_SIZE+[2], name="input_br_reg", dtype=tf.float32)
        input_ct_reg = layers.Input(
            shape=self.config.OUTPUT_SIZE+[2], name="input_ct_reg", dtype=tf.float32)

        input_mask = layers.Input(
            shape=[3] + self.config.OUTPUT_SIZE, name="input_mask", dtype=tf.float32)
        input_tag_mask = layers.Input(
            shape=[self.config.MAX_NUMS], name="input_tag_mask", dtype=tf.float32)
        input_tl_tag = layers.Input(
            shape=[self.config.MAX_NUMS], name="input_tl_tag", dtype=tf.int64)
        input_br_tag = layers.Input(
            shape=[self.config.MAX_NUMS], name="input_br_tag", dtype=tf.int64)
        input_gt_bbox = layers.Input(
            shape=[self.config.MAX_NUMS, 4], name="input_gt_bbox", dtype=tf.float32)
        input_gt_class_id = layers.Input(
            shape=[self.config.MAX_NUMS], name="input_gt_class_id", dtype=tf.int64)

        # Build the center network graph
        x = base_module(input_image, self.config.INTER_CHANNELS[0])
        backbone_feat = hourglass_module(
            x, self.config.INTER_CHANNELS, self.config.NUM_FEATS)
        outs = head(backbone_feat, self.config.NUM_FEATS, self.config.CLASSES)

        tl_map_loss = layers.Lambda(lambda x: neg_loss(
            *x), name="tl_map_loss")([outs[0], input_tl_heatmaps])
        br_map_loss = layers.Lambda(lambda x: neg_loss(
            *x), name="br_map_loss")([outs[1], input_br_heatmaps])
        ct_map_loss = layers.Lambda(lambda x: neg_loss(
            *x), name="ct_map_loss")([outs[2], input_ct_heatmaps])

        # regression loss
        tl_mask, br_mask, ct_mask = layers.Lambda(
            lambda x: tf.unstack(x, axis=1), name="unstack_mask")(input_mask)
        lt_reg_loss = layers.Lambda(lambda x: regr_loss(
            *x), name="tl_reg_loss")([outs[5], input_tl_reg, tl_mask])
        br_reg_loss = layers.Lambda(lambda x: regr_loss(
            *x), name="br_reg_loss")([outs[6], input_br_reg, br_mask])
        ct_reg_loss = layers.Lambda(lambda x: regr_loss(
            *x), name="ct_reg_loss")([outs[7], input_ct_reg, ct_mask])

        # embedding loss
        pull_push_loss = layers.Lambda(lambda x: ae_loss(*x),
                                   name="ae_loss")([outs[3], outs[4], input_tl_tag, input_br_tag, input_tag_mask])

        # Model
        inputs = [input_image, input_image_meta, input_tl_heatmaps, input_br_heatmaps,
                  input_ct_heatmaps, input_tl_reg, input_br_reg, input_ct_reg, input_mask, input_tl_tag,
                  input_br_tag, input_tag_mask, input_gt_bbox, input_gt_class_id]

        outputs = outs + [tl_map_loss, br_map_loss, ct_map_loss, lt_reg_loss, br_reg_loss,
                          ct_reg_loss, pull_push_loss]
        model = tensorflow.keras.models.Model(inputs, outputs, name='centernet52')

        return model
