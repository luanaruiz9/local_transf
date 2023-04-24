#!/bin/bash

python main_increase_neigh.py cora 500 2
python main_increase_neigh.py cora 500 5
python main_increase_neigh.py cora 300 2
python main_increase_neigh.py cora 300 5
python main_increase_neigh.py citeseer 500 2
python main_increase_neigh.py citeseer 500 5
python main_increase_neigh.py citeseer 300 2
python main_increase_neigh.py citeseer 300 5
python main_increase_neigh_pub.py pubmed 2000 10
python main_increase_neigh_pub.py pubmed 2000 15
python main_increase_neigh_pub.py pubmed 1500 10
python main_increase_neigh_pub.py pubmed 1500 15

