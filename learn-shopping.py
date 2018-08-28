


import tensorflow as tf
from sklearn.cross_validation import train_test_split
from sklearn.datasets import make_classification
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np



TRAIN_DATA_FILE = 'data/interview_dataset_train'
TEST_DATA_FILE = 'data/interview_dataset_test_no_tags'

def process_data_file(file_name):
    df_data = pd.read_csv(file_name, sep='\t')

    # encode categorized features
    lb_make = LabelEncoder()
    colums = ['target_product_category', 'shopper_segment', 'delivery_time']
    for c in colums:
        df_data[c + "_code"] = lb_make.fit_transform(df_data[c])
    # remove non-numerical
    data_num_df = df_data.select_dtypes(include=[np.number])

    return data_num_df

# Parameters
learning_rate = 0.01
training_epochs = 100
batch_size = 100
display_step = 5



data_num_df = process_data_file(TRAIN_DATA_FILE)
y_num_df = data_num_df['tag']
x_num_df = data_num_df.drop(['tag'], axis=1)
X = x_num_df.fillna(0)
Y = np.array([y_num_df, -(y_num_df - 1)]).T  # The model currently needs one column for each class
X, X_dev, Y, Y_dev = train_test_split(X, Y)



# Network Parameters
n_hidden_1 = 25 # 1st layer number of features
n_hidden_2 = 10 # 2nd layer number of features
n_input = len(x_num_df.columns) # Number of feature
n_classes = 2 # Number of classes to predict


# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}

biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = multilayer_perceptron(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()


with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(len(X) / batch_size)
        X_batches = np.array_split(X, total_batch)
        Y_batches = np.array_split(Y, total_batch)
        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = X_batches[i], Y_batches[i]
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                          y: batch_y})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch + 1), "cost={:.9f}".format(avg_cost))
            # Test model
            correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
            # Calculate accuracy
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            print("Accuracy:", accuracy.eval({x: X_dev, y: Y_dev}))

    print("Optimization Finished!")

    X_test = process_data_file(TEST_DATA_FILE)
    result = tf.argmax(pred, 1).eval({x: X_test, y: np.zeros([X_test.size,n_classes])})
    with open('pred','w') as f:
        for r in result:
            f.writelines(str(r)+'\n')
