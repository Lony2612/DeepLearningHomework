import tensorflow as tf
import tensorflow.contrib.slim as slim
from collections import namedtuple
from cnn_utils import *


def _make_divisible(v, divisor, min_value=None):
    """make `v` is divided exactly by `divisor`, but keep the min_value"""
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


@slim.add_arg_scope
def _depth_multiplier_func(params,
                           multiplier,
                           divisible_by=8,
                           min_depth=8):
    """get the new channles"""
    if 'num_outputs' not in params:
        return
    d = params['num_outputs']
    params['num_outputs'] = _make_divisible(d * multiplier, divisible_by,
                                                   min_depth)


#@slim.add_arg_scope
def expanded_conv(x,
                  num_outputs,
                  expansion=6,
                  stride=1,
                  rate=1,
                  normalizer_fn=slim.batch_norm,
                  project_activation_fn=tf.identity,
                  padding="SAME",
                  scope=None):
    """The expand conv op in MobileNetv2
        1x1 conv -> depthwise 3x3 conv -> 1x1 linear conv
    """
    with tf.variable_scope(scope, default_name="expanded_conv") as s, \
       tf.name_scope(s.original_name_scope):
        prev_depth = x.get_shape().as_list()[3]
        # the filters of expanded conv
        inner_size = prev_depth * expansion
        net = x

        ########################To Do#################################

        # only inner_size > prev_depth, use expanded conv

        # depthwise conv

        # projection


        # residual connection
        ########################To Do#################################

        return net


_Op = namedtuple("Op", ['op', 'params', 'multiplier_func'])

def op(op_func, **params):
    return _Op(op=op_func, params=params,
               multiplier_func=_depth_multiplier_func)

########################To Do#################################
CONV_DEF = [
            ]
########################To Do#################################

def mobilenetv2(x,
                num_classes=1001,
                prediction_fn=slim.softmax,
                depth_multiplier=1.4,
                scope='MobilenetV2',
                finegrain_classification_mode=False,
                min_depth=8,
                divisible_by=8,
                output_stride=None,
                ):
    """Mobilenet v2
    Args:
        x: The input tensor
        num_classes: number of classes
        depth_multiplier: The multiplier applied to scale number of
            channels in each layer. Note: this is called depth multiplier in the
            paper but the name is kept for consistency with slim's model builder.
        scope: Scope of the operator
        finegrain_classification_mode: When set to True, the model
            will keep the last layer large even for small multipliers.
            The paper suggests that it improves performance for ImageNet-type of problems.
        min_depth: If provided, will ensure that all layers will have that
          many channels after application of depth multiplier.
       divisible_by: If provided will ensure that all layers # channels
          will be divisible by this number.
    """
    conv_defs = CONV_DEF

    # keep the last conv layer very larger channel
    if finegrain_classification_mode:
        conv_defs = copy.deepcopy(conv_defs)
        if depth_multiplier < 1:
            conv_defs[-1].params['num_outputs'] /= depth_multiplier

    depth_args = {}
    # NB: do not set depth_args unless they are provided to avoid overriding
    # whatever default depth_multiplier might have thanks to arg_scope.
    if min_depth is not None:
        depth_args['min_depth'] = min_depth
    if divisible_by is not None:
        depth_args['divisible_by'] = divisible_by

    with slim.arg_scope([_depth_multiplier_func], **depth_args):
        with tf.variable_scope(scope, default_name='Mobilenet'):
            # The current_stride variable keeps track of the output stride of the
            # activations, i.e., the running product of convolution strides up to the
            # current network layer. This allows us to invoke atrous convolution
            # whenever applying the next convolution would result in the activations
            # having output stride larger than the target output_stride.
            current_stride = 1

            # The atrous convolution rate parameter.
            rate = 1

            net = x
            # Insert default parameters before the base scope which includes
            # any custom overrides set in mobilenet.
            end_points = {}
            scopes = {}
            for i, opdef in enumerate(conv_defs):
                params = dict(opdef.params)
                opdef.multiplier_func(params, depth_multiplier)
                stride = params.get('stride', 1)
                if output_stride is not None and current_stride == output_stride:
                    # If we have reached the target output_stride, then we need to employ
                    # atrous convolution with stride=1 and multiply the atrous rate by the
                    # current unit's stride for use in subsequent layers.
                    layer_stride = 1
                    layer_rate = rate
                    rate *= stride
                else:
                    layer_stride = stride
                    layer_rate = 1
                    current_stride *= stride
                # Update params.
                params['stride'] = layer_stride
                # Only insert rate to params if rate > 1.
                if layer_rate > 1:
                    params['rate'] = layer_rate

                try:
                     ########################To Do#################################
                     #######build layer################
                     ########################To Do#################################
                except Exception:
                    raise ValueError('Failed to create op %i: %r params: %r' % (i, opdef, params))

            with tf.variable_scope('Logits'):

                ########################To Do#################################
                #######global_pool################
                ########################To Do#################################
                end_points['global_pool'] = net
                if not num_classes:
                    return net, end_points
                ########################To Do#################################
                #######dropout################
                #
                # and 
                # FC to classes
                # 1 x 1 x num_classes
                # Note: legacy scope name.
                # scope='Conv2d_1c_1x1'
                # output: logits
                ########################To Do#################################

            end_points['Logits'] = logits
            if prediction_fn:
              end_points['Predictions'] = prediction_fn(logits, 'Predictions')
            return logits,end_points

def mobilenet_arg_scope(is_training=True,
                        weight_decay=0.00004,
                        stddev=0.09,
                        dropout_keep_prob=0.8,
                        bn_decay=0.997):
    """Defines Mobilenet default arg scope.
    Usage:
     with tf.contrib.slim.arg_scope(mobilenet.training_scope()):
       logits, endpoints = mobilenet_v2.mobilenet(input_tensor)
     # the network created will be trainble with dropout/batch norm
     # initialized appropriately.
    Args:
        is_training: if set to False this will ensure that all customizations are
            set to non-training mode. This might be helpful for code that is reused
        across both training/evaluation, but most of the time training_scope with
        value False is not needed. If this is set to None, the parameters is not
        added to the batch_norm arg_scope.
        weight_decay: The weight decay to use for regularizing the model.
        stddev: Standard deviation for initialization, if negative uses xavier.
        dropout_keep_prob: dropout keep probability (not set if equals to None).
        bn_decay: decay for the batch norm moving averages (not set if equals to
            None).
    Returns:
        An argument scope to use via arg_scope.
    """
    # Note: do not introduce parameters that would change the inference
    # model here (for example whether to use bias), modify conv_def instead.
    batch_norm_params = {
        'center': True,
        'scale': True,
        'decay': bn_decay,
        'is_training': is_training
    }
    if stddev < 0:
        weight_intitializer = slim.initializers.xavier_initializer()
    else:
        weight_intitializer = tf.truncated_normal_initializer(stddev=stddev)

    # Set weight_decay for weights in Conv and FC layers.
    with slim.arg_scope(
        [slim.conv2d, slim.fully_connected, slim.separable_conv2d],
        weights_initializer=weight_intitializer,
        normalizer_fn=slim.batch_norm,
        activation_fn=tf.nn.relu6), \
        slim.arg_scope([slim.batch_norm], **batch_norm_params), \
        slim.arg_scope([slim.dropout], is_training=is_training,
                     keep_prob=dropout_keep_prob), \
        slim.arg_scope([slim.conv2d, slim.separable_conv2d],
                       biases_initializer=None,
                       padding="SAME"), \
        slim.arg_scope([slim.conv2d],
                     weights_regularizer=slim.l2_regularizer(weight_decay)), \
        slim.arg_scope([slim.separable_conv2d], weights_regularizer=None) as s:
        return s


