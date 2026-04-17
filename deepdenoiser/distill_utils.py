import tensorflow as tf


def _to_float32_scalar(value, name):
    return tf.constant(float(value), dtype=tf.float32, name=name)


def distill_soft_ce(student_logits, teacher_logits, temperature=2.0, name='distill_soft_ce'):
    """Temperature-scaled soft cross-entropy distillation loss.

    Args:
        student_logits: Tensor [..., n_class], float32.
        teacher_logits: Tensor with same shape as student_logits, float32.
        temperature: Distillation temperature T.
        name: Op scope name.

    Returns:
        Scalar tensor: mean soft CE * (T^2).
    """
    with tf.compat.v1.variable_scope(name):
        T = _to_float32_scalar(temperature, 'temperature')
        student_logits = tf.convert_to_tensor(value=student_logits, dtype=tf.float32)
        teacher_logits = tf.stop_gradient(tf.convert_to_tensor(value=teacher_logits, dtype=tf.float32))

        teacher_prob_T = tf.nn.softmax(teacher_logits / T, axis=-1, name='teacher_prob_T')
        student_logits_T = student_logits / T

        flat_student = tf.reshape(student_logits_T, [-1, tf.shape(input=student_logits_T)[-1]], name='flat_student_logits_T')
        flat_teacher = tf.reshape(teacher_prob_T, [-1, tf.shape(input=teacher_prob_T)[-1]], name='flat_teacher_prob_T')

        loss_map = tf.nn.softmax_cross_entropy_with_logits(labels=flat_teacher, logits=flat_student)
        loss = tf.reduce_mean(input_tensor=loss_map, name='soft_ce_mean')
        loss = tf.multiply(loss, T * T, name='soft_ce_scaled')
        return loss


def distill_logits_mse(student_logits, teacher_logits, name='distill_logits_mse'):
    """Optional auxiliary loss for quick ablations."""
    with tf.compat.v1.variable_scope(name):
        student_logits = tf.convert_to_tensor(value=student_logits, dtype=tf.float32)
        teacher_logits = tf.stop_gradient(tf.convert_to_tensor(value=teacher_logits, dtype=tf.float32))
        return tf.reduce_mean(input_tensor=tf.square(student_logits - teacher_logits), name='mse')


def combine_losses(hard_loss, distill_loss, alpha=0.5, beta=0.5, reg_loss=None, name='distill_total_loss'):
    """Combine hard-label loss and distillation loss.

    total = alpha * hard_loss + beta * distill_loss + reg_loss
    """
    with tf.compat.v1.variable_scope(name):
        alpha_t = _to_float32_scalar(alpha, 'alpha')
        beta_t = _to_float32_scalar(beta, 'beta')
        total = alpha_t * tf.cast(hard_loss, tf.float32) + beta_t * tf.cast(distill_loss, tf.float32)
        if reg_loss is not None:
            total = total + tf.cast(reg_loss, tf.float32)
        return tf.identity(total, name='total')
