
import tensorflow as tf
from tensorflow.keras import layers # type: ignore


def sparse_tuple_from(y_batch, num_classes):
    """
    Convert batch of labels to sparse tensor format.
    """
    indices = []
    values = []
    dense_shape = [y_batch.shape[0], 0]
    
    for batch_idx, label in enumerate(y_batch):
        for time_idx, value in enumerate(label):
            if value != num_classes:  # Assuming num_classes is the padding token
                indices.append([batch_idx, time_idx])
                values.append(value)
        dense_shape[1] = max(dense_shape[1], len(label))
    
    return tf.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # Convert labels to SparseTensor
    sparse_labels = sparse_tuple_from(labels, num_classes=29)  # Adjust num_classes as needed
    return tf.compat.v1.nn.ctc_loss(labels=sparse_labels, inputs=y_pred, sequence_length=tf.squeeze(input_length))

# Example usage of the Lambda layer with specified output_shape
def build_model(input_dim, output_dim):
    # Define your model architecture here
    inputs = layers.Input(shape=input_dim, name='input')
    # Example layers
    x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    y_pred = layers.Dense(output_dim, activation='softmax', name='output')(x)

    # Define the CTC loss
    labels = layers.Input(name='labels', shape=[None], dtype='int32')
    input_length = layers.Input(name='input_length', shape=[1], dtype='int32')
    label_length = layers.Input(name='label_length', shape=[1], dtype='int32')

    loss_out = layers.Lambda(ctc_lambda_func, output_shape=(1,), name='ctc_loss')([y_pred, labels, input_length, label_length])

    # Define the model
    model = tf.keras.Model(inputs=[inputs, labels, input_length, label_length], outputs=loss_out)

    # Compile the model
    model.compile(optimizer='adam', loss={'ctc_loss': lambda y_true, y_pred: y_pred})

    return model, model

# Example usage
if __name__ == "__main__":
    train_model, pred_model = build_model(input_dim=(128, 128, 1), output_dim=29)
    print(train_model.summary())