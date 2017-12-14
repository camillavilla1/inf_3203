
from BaseHTTPServer import HTTPServer, BaseHTTPRequestHandler
from SocketServer import ThreadingMixIn
import threading
import time
import re
import argparse
import httplib
import signal
import socket
import os
import subprocess
import sys
import collections


parameter_server_list = []
workers_list = []
free_nodes_list = []


ps_string = ""
w_string = ""


def launch(host, commandline, stdout=None, stderr=None, wait=False):
	#Runs a command either locally or on a remote host via SSH
	cwd = os.getcwd()
	if host == 'localhost':
		pass
	else:
		commandline = "cd %s; %s" % (cwd, commandline)
	
	print commandline
	process= subprocess.Popen(["ssh", host, commandline])
	if wait:
		process.wait()

	subp_list.append(process)


def kill(host):
	""" Kills a process by name, either locally or on a remote host via SSH """
	try:
		process = subprocess.Popen(["ssh", host, "pgrep -u cst042 python | xargs kill -s SIGTERM"])
		print process.wait()
	except Exception, e:
		print "Unable to kill on %s" % (str(host))


#Used if user want totype in number of nodes to run
def parse_args():
	parser = argparse.ArgumentParser(prog="Launch", description="Node in a cluster")
	parser.add_argument("num_nodes", type=int, help="Number of ps(1) + workers")

	return parser.parse_args()


def find_free_node(num_nodes, stdout=None, stderr=None, wait=False):
	i = 0
	for i in range(num_nodes):
		commandline = "sh find_nodes.sh 1"
		process = subprocess.Popen(commandline, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
		output = process.communicate()
		if wait:
			process.wait()

		ip = output[0]

		free_nodes_list.append(ip[:-2])

	return free_nodes_list


class HostPort(collections.namedtuple('HostPort', ['host','port'])):
    def __str__(self):
    	return "%s:%d" % (self.host, self.port)


def parse_host_port(hp_str, n_port):
	if ":" in hp_str:
		host, port = hp_str.split(":")
		port = int(port)
	else:
		host = hp_str
		port = n_port

	return HostPort(host,port)


def init_parameter_serv(nodes, k):
	global ps_string
	ps_string = str(nodes[0][0]) + ":" + str(nodes[0][1]+k)
	parameter_server_list = tuple(ps_string)


def init_workers(nodes, k):
	global w_string
	node_list = []
	workers_list = []
	n_nodes = nodes[1:]
	for node in n_nodes:
		node_list.append(str(node[0]) + ":" + str(node[1]+k))

	workers_list = node_list
	workers_list = tuple(workers_list)
	w_string = (",").join(workers_list)


if __name__ == '__main__':
	port = 8088
	#num_nodes = int(sys.argv[1])
	#args = parse_args()
	for i in range(2,12):
		num_nodes = i

		free_nodes_list = []
		find_free_node(num_nodes)

		for k in range(30):
			port = port + 1
			nodes = [parse_host_port(hp, port) for hp in free_nodes_list]
			init_parameter_serv(nodes, k)
			init_workers(nodes, k)

			task_index = 0
			subp_list = []

			try:
				for node in nodes:
					ps = nodes[0]
					if node == ps:
						launch(node.host, "python mnist.py --job_name=ps --task_index=0 --para_serv=%s --worker=%s" % (ps_string, w_string))
						time.sleep(3)
					else:
						launch(node.host, "python mnist.py --job_name=worker --task_index=%d --para_serv=%s --worker=%s " % (task_index, ps_string, w_string))
						task_index += 1

			finally:
				if len(subp_list) > 0:
					subp_list[1].wait()

				n_nodes = nodes[1:]
				for node in n_nodes:
					kill(node.host)
					time.sleep(5)

				kill(nodes[0].host)



	print("Exited")




