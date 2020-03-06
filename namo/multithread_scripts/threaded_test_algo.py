import numpy as np
import scipy.io as sio
import os
import sys
import threading

from Queue import Queue
from multiprocessing.pool import ThreadPool # dummy is nothing but multiprocessing but wrapper around threading
from multiprocessing import cpu_count

import pickle
import socket
import argparse
import csv
import time
import itertools
import sys

def worker_p(config):
  n_data  = config[0]
  n_trial = config[1]
  #pidx    = config[2]

  #command = 'python ./test_scripts/test_soap.py'+ ' -n_data '+ \
  #          str(n_data) + ' -n_trial '+str(n_trial) + ' -epoch '+str(pidx) 
  command = 'python ./test_scripts/test_algo.py'+ ' -pi trpo -n_data '+ \
            str(n_data) 
  
  print command
  os.system(command)

def worker_wrapper_multi_input(multi_args):
  return worker_p(multi_args)

def main():
  #n_datas   = [int(k) for k in sys.argv[1].split(',')]
  n_datas = [1000,3000,4000,6000,9000]
  trials   = [int(k) for k in sys.argv[1].split(',')]
  epochs    = range(300)
  
  configs=[]
  for n_data in n_datas:
    for g in epochs:
      configs.append([n_data,g])
      #  configs.append([n_data,t,g])
  #configs = itertools.product([n_data],trials,epochs)
  n_workers = int(cpu_count())
  print configs

  pool = ThreadPool(n_workers)
  results = pool.map(worker_wrapper_multi_input,configs)

if __name__ == '__main__':
  main()
