

from models.utils.loss import neg_loss, regr_loss, ae_loss
from models.utils.module import hourglass_module, base_module, cascade_br_pool, cascade_tl_pool, center_pool
from models.utils.module import heat_layer
from data.prepareData import data_generator

import tensorflow.keras.models
from tensorflow.keras import layers
import tensorflow as tf

import multiprocessing
import os


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
        backbone_feat = hourglass_module(x, self.config.INTER_CHANNELS, self.config.NUM_FEATS)
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
        model = tensorflow.keras.models.Model(
            inputs, outputs, name='centernet52')

        return model

    def set_log_dir(self, model_path=None):
        """Sets the model log directory and epoch counter.
        model_path: If None, or a format different from what this code uses
            then set a new log directory and start epochs from 0. Otherwise,
            extract the log directory and the epoch counter from the file
            name.
        """
        # Set date and epoch counter as if starting a new model
        self.epoch = 0
        # Directory for training logs
        self.log_dir = os.path.join(
            self.weight_dir, self.config.NET_NAME.lower())

        # Path to save after each epoch. Include placeholders that get filled by Keras.
        self.checkpoint_path = os.path.join(
            self.log_dir, "centernet52_{epoch:02d}.h5")

    def find_last(self):
        """Finds the last checkpoint file of the last trained model in the
        model directory.
        Returns:
            The path of the last checkpoint file
        """
        # Get directory names. Each directory corresponds to a model
        dir_names = next(os.walk(self.weight_dir))[1]
        key = self.config.NET_NAME.lower()
        dir_names = filter(lambda f: f.startswith(key), dir_names)
        dir_names = sorted(dir_names)
        if not dir_names:
            import errno
            raise FileNotFoundError(
                errno.ENOENT,
                "Could not find model directory under {}".format(self.weight_dir))
        # Pick last directory
        dir_name = os.path.join(self.weight_dir, dir_names[-1])
        # Find the last checkpoint
        checkpoints = next(os.walk(dir_name))[2]
        checkpoints = filter(lambda f: f.startswith(
            "centernet52"), checkpoints)
        checkpoints = sorted(checkpoints)
        if not checkpoints:
            import errno
            raise FileNotFoundError(
                errno.ENOENT, "Could not find weight files in {}".format(dir_name))
        checkpoint = os.path.join(dir_name, checkpoints[-1])
        return checkpoint

    def log(text, array=None):
        """Prints a text message. And, optionally, if a Numpy array is provided it
            prints it's shape, min, and max values.
        """
        if array is not None:
            text = text.ljust(25)
            text += ("shape: {:20}  ".format(str(array.shape)))
            if array.size:
                text += ("min: {:10.5f}  max: {:10.5f}".format(array.min(), array.max()))
            else:
                text += ("min: {:10}  max: {:10}".format("", ""))
            text += "  {}".format(array.dtype)
        print(text)


    def compile(self, learning_rate, momentum):
        """Gets the model ready for training. Adds losses, regularization, and
        metrics. Then calls the Keras compile() function.
        """
        # Optimizer object
        optimizer = tf.keras.optimizers.SGD(
            lr=learning_rate, momentum=momentum,
            clipnorm=self.config.GRADIENT_CLIP_NORM)
        # Add Losses
        # First, clear previously set losses to avoid duplication
        self.model._losses = []
        self.model._per_input_losses = {}
        loss_names = ["tl_map_loss",  "br_map_loss", "ct_map_loss",
                      "tl_reg_loss", "br_reg_loss", "ct_reg_loss", "ae_loss"]
        for name in loss_names:
            layer = self.model.get_layer(name)
            if layer.output in self.model.losses:
                continue
            loss = (
                tf.reduce_mean(layer.output, keepdims=True)
                * self.config.LOSS_WEIGHTS.get(name, 1.))
            self.model.add_loss(loss)

        # Add L2 Regularization
        # Skip gamma and beta weights of batch normalization layers.
        reg_losses = [
            tf.keras.regularizers.l2(self.config.WEIGHT_DECAY)(w) / tf.cast(tf.size(w), tf.float32)
            for w in self.model.trainable_weights
            if 'gamma' not in w.name and 'beta' not in w.name]
        self.model.add_loss(tf.add_n(reg_losses))

        # Compile
        self.model.compile(
            optimizer=optimizer,
            loss=[None] * len(self.model.outputs))

        # Add metrics for losses
        for name in loss_names:
            if name in self.model.metrics_names:
                continue
            layer = self.model.get_layer(name)
            self.model.metrics_names.append(name)
            loss = (layer.output * self.config.LOSS_WEIGHTS.get(name, 1.))
            self.model.metrics_tensors.append(loss)

    def train(self, train_dataset, val_dataset, learning_rate, epochs,
              augment=None, custom_callbacks=None):
        """Train the model.
        train_dataset, val_dataset: Training and validation Dataset objects.
        learning_rate: The learning rate to train with
        epochs: Number of training epochs.
        augment: Optional.
        """
        

        # Data generators
        train_generator = data_generator(train_dataset, self.config, shuffle=True,
                                         augment=augment)
        val_generator = data_generator(val_dataset, self.config)

        # Create log_dir if it does not exist
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # Callbacks
        callbacks = [
            tf.keras.callbacks.TensorBoard(log_dir=self.log_dir,
                                           histogram_freq=0, write_graph=True, write_images=False),
            tf.keras.callbacks.ModelCheckpoint(self.checkpoint_path,
                                               verbose=0, save_weights_only=True),
        ]

        # Add custom callbacks to the list
        if custom_callbacks:
            callbacks += custom_callbacks

        # Train
        self.log("\nStarting at epoch {}. LR={}\n".format(self.epoch, learning_rate))
        self.log("Checkpoint Path: {}".format(self.checkpoint_path))
        self.compile(learning_rate, self.config.MOMENTUM)

        # Work-around for Windows: Keras fails on Windows when using
        # multiprocessing workers. See discussion here:
        # https://github.com/matterport/Mask_RCNN/issues/13#issuecomment-353124009
        if os.name is 'nt':
            workers = 0
        else:
            workers = multiprocessing.cpu_count()

        self.model.fit_generator(
            train_generator,
            initial_epoch=self.epoch,
            epochs=epochs,
            steps_per_epoch=self.config.STEPS_PER_EPOCH,
            callbacks=callbacks,
            validation_data=val_generator,
            validation_steps=self.config.VAL_STEPS,
            max_queue_size=100,
            workers=workers,
            use_multiprocessing=True,
        )
        self.epoch = max(self.epoch, epochs)