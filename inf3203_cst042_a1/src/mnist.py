#Assignment 1 - Camilla Stormoen

from __future__ import print_function

import tensorflow as tf
import sys
import time


#input flags
tf.app.flags.DEFINE_string("job_name", "", "Either 'ps' or 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
tf.app.flags.DEFINE_string("para_serv", "", "Parameter server")
tf.app.flags.DEFINE_string("worker", "", "Worker node")
FLAGS = tf.app.flags.FLAGS

all_workers = FLAGS.worker.split(",")
all_ps = FLAGS.para_serv.split(",")

cluster = tf.train.ClusterSpec({"ps": all_ps, "worker": all_workers})
server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)



#config
batch_size = 100 #change this -> will go faster
learning_rate = 0.5 #0.001
training_epochs = 3 #20
logs_path = "/tmp/mnist/1"

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

if FLAGS.job_name == "ps":
	server.join()
elif FLAGS.job_name == "worker":
	#Between-graph replication
	with tf.device(tf.train.replica_device_setter(worker_device="/job:worker/task:%d" % FLAGS.task_index, cluster=cluster)):

		#Count the number of updates
		global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

	
		#input images
		with tf.name_scope('input'):
			#None -> batch size can be any size, 784 -> flattened mnist image
			x = tf.placeholder(tf.float32, shape=[None, 784], name="x-input")
			# target 10 output classes
			y_ = tf.placeholder(tf.float32, shape=[None, 10], name="y-input")

		#model parameters will change during training so we use tf.Variable
		tf.set_random_seed(1)
		with tf.name_scope("weights"):
			w1 = tf.Variable(tf.random_normal([784, 100]))
			w2 = tf.Variable(tf.random_normal([100, 10]))

		#bias
		with tf.name_scope("biases"):
			b1 = tf.Variable(tf.zeros([100]))
			b2 = tf.Variable(tf.zeros([10]))


		#implement model
		with tf.name_scope("softmax"):
			#y is our prediction
			z2 = tf.add(tf.matmul(x, w1), b1)
			a2 = tf.nn.sigmoid(z2)
			z3 = tf.add(tf.matmul(a2, w2),b2)
			y = tf.nn.softmax(z3)

		#Specify cost function
		with tf.name_scope("cross_entropy"):
			cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

		#Specify optimizer
		with tf.name_scope("train"):
			grad_op = tf.train.GradientDescentOptimizer(learning_rate)
			train_op = grad_op.minimize(cross_entropy, global_step=global_step)

		#Accurancy
		with tf.name_scope("accurancy"):
			correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
			accurancy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

		#Create a summary for our cost and accurancy
		tf.summary.scalar("Cost", cross_entropy)
		tf.summary.scalar("Accurancy", accurancy)

		#Merge all summaries into a singe "operation" which we can execute...
		summary_op = tf.summary.merge_all()
		#init_op = tf.initialize_all_variables()
		init_op = tf.global_variables_initializer()
		print("Variables initialized!")

	sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0), global_step=global_step, init_op=init_op)

	begin_time = time.time()
	frequence = 100

	with sv.prepare_or_wait_for_session(server.target) as sess:
		print("\n\n Prepare or wait for session\n")
		#Create a log object (this will log on every machine)
		#writer = tf.train.SummaryWriter(logs_path, graph=tf.get_default_graph())

		#Perform training cycles
		start_time = time.time()
		for epoch in range(training_epochs):
			#Perform the operations we defined earlier on batch
	    		batch_count = int(mnist.train.num_examples/batch_size)

	    		count = 0
	    		num_node = len(all_workers)
	    		for i in range(batch_count):
	    			batch_x, batch_y = mnist.train.next_batch(batch_size)

	    			#perform the operations we defines earlier on batch
	    			_, cost, summary, step = sess.run([train_op, cross_entropy, summary_op, global_step], feed_dict={x: batch_x, y_: batch_y})

	    			#writer.add_summary(summary, step)

	    			count += 1

	    			#if(FLAGS.task_index == 0):

	    			if count % frequence == 0 or i+1 == batch_count:
	    				elapsed_time = time.time() - start_time
	    				start_time = time.time()
	    				print("\n\n###########\nNumber of nodes are: %d" % (int(num_node)))

	    				"""print("Step: %d," % (step+1),
	    					"Epoch: %2d," % (epoch+1), 
	    					"Batch: %3d of %3d," % (i+1, batch_count),
	    					"Cost: %.4f," % (cost),
	    					"AvgTime: %3.2fms" % float(elapsed_time*1000/frequence))
	    				"""
	    				count= 0

	    	print("Test-Accurancy: %2.2f" % sess.run(accurancy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
	    	print("Total Time: %3.2fs" % float(time.time() - begin_time))
	    	print("Final Cost: %.4f" % cost)

	    	timezz = float(time.time() - begin_time)
	    	if(FLAGS.task_index == 0):
	    		print("\n ---------------------------\n WRITE TO FUCKING FILE\n ---------------------------\n")
	    		#Put time and # nodes in here
	    		result = open("result", 'a')
	    		result.write(str(num_node)) # # of nodes (2 ... 10) -> start on 2 - 12
	    		result.write(",")
	    		result.write(str(timezz)) #time each run used
	    		result.write("\n")
	    		result.close()
	    	else:
	    		print("\n ---------------------------\n Flags task index:")
	    		print(FLAGS.task_index)
	    		print("\n---------------------------\n")

	    	timezz = 0.0


	sv.stop()
	print("Done!")


print("\nAll done!\n")



