#!/bin/bash

python main_increase_neighbor.py cora 500 2
python main_increase_neighbor.py cora 500 5
python main_increase_neighbor.py cora 300 2
python main_increase_neighbor.py cora 300 5
python main_increase_neighbor.py citeseer 500 2
python main_increase_neighbor.py citeseer 500 5
python main_increase_neighbor.py citeseer 300 2
python main_increase_neighbor.py citeseer 300 5
python main_increase_neighbor_pub.py pubmed 2000 10
python main_increase_neighbor_pub.py pubmed 2000 15
python main_increase_neighbor_pub.py pubmed 1500 10
python main_increase_neighbor_pub.py pubmed 1500 15

