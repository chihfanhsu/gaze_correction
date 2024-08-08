import tensorflow as tf

def repeat(x, num_repeats):
    with tf.name_scope("repeat"):
        ones = tf.ones((1, num_repeats), dtype=tf.int32)
        x = tf.reshape(x, shape=(-1, 1))
        x = tf.matmul(x, ones)
        return tf.reshape(x, [-1])

def interpolate(image, x, y, output_size):
    with tf.name_scope("interpolate"):
        batch_size, height, width, num_channels = tf.unstack(tf.shape(image))
        
        x, y = tf.cast(x, tf.float32), tf.cast(y, tf.float32)
        height_float, width_float = tf.cast(height, tf.float32), tf.cast(width, tf.float32)
        
        x = 0.5 * (x + 1.0) * width_float
        y = 0.5 * (y + 1.0) * height_float
        
        x0 = tf.cast(tf.floor(x), tf.int32)
        x1 = x0 + 1
        y0 = tf.cast(tf.floor(y), tf.int32)
        y1 = y0 + 1

        max_y, max_x = height - 1, width - 1

        x0 = tf.clip_by_value(x0, 0, max_x)
        x1 = tf.clip_by_value(x1, 0, max_x)
        y0 = tf.clip_by_value(y0, 0, max_y)
        y1 = tf.clip_by_value(y1, 0, max_y)

        flat_image_dimensions = height * width
        pixels_batch = tf.range(batch_size) * flat_image_dimensions
        flat_output_dimensions = output_size[0] * output_size[1]
        base = repeat(pixels_batch, flat_output_dimensions)
        base_y0 = base + y0 * width
        base_y1 = base + y1 * width
        indices_a = base_y0 + x0
        indices_b = base_y1 + x0
        indices_c = base_y0 + x1
        indices_d = base_y1 + x1

        flat_image = tf.reshape(image, shape=(-1, num_channels))
        flat_image = tf.cast(flat_image, tf.float32)
        pixel_values_a = tf.gather(flat_image, indices_a)
        pixel_values_b = tf.gather(flat_image, indices_b)
        pixel_values_c = tf.gather(flat_image, indices_c)
        pixel_values_d = tf.gather(flat_image, indices_d)

        x0, x1, y0, y1 = tf.cast(x0, tf.float32), tf.cast(x1, tf.float32), tf.cast(y0, tf.float32), tf.cast(y1, tf.float32)

        area_a = tf.expand_dims((x1 - x) * (y1 - y), 1)
        area_b = tf.expand_dims((x1 - x) * (y - y0), 1)
        area_c = tf.expand_dims((x - x0) * (y1 - y), 1)
        area_d = tf.expand_dims((x - x0) * (y - y0), 1)

        output = tf.add_n([area_a * pixel_values_a,
                           area_b * pixel_values_b,
                           area_c * pixel_values_c,
                           area_d * pixel_values_d])
        return output

def meshgrid(height, width):
    with tf.name_scope("meshgrid"):
        y_linspace = tf.linspace(-1.0, 1.0, height)
        x_linspace = tf.linspace(-1.0, 1.0, width)
        x_coordinates, y_coordinates = tf.meshgrid(x_linspace, y_linspace)
        y_coordinates = tf.expand_dims(tf.reshape(y_coordinates, [-1]), 0)
        x_coordinates = tf.expand_dims(tf.reshape(x_coordinates, [-1]), 0)
        return tf.concat([x_coordinates, y_coordinates], 0)

def apply_transformation(flows, img):
    with tf.name_scope("apply_transformation"):
        batch_size, height, width, num_channels = tf.unstack(tf.shape(img))
        output_size = (height, width)

        flows = tf.reshape(tf.transpose(flows, [0, 3, 1, 2]), [batch_size, -1, height * width])
        indices_grid = meshgrid(height, width)
        transformed_grid = flows + indices_grid
        x_s, y_s = tf.unstack(transformed_grid, axis=1)

        transformed_image = interpolate(img, tf.reshape(x_s, [-1]), tf.reshape(y_s, [-1]), output_size)
        return tf.reshape(transformed_image, [batch_size, height, width, num_channels])
