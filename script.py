#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 16:16:20 2018

@author: inesarous
"""
import os
os.system("python run.py TVBPR 10 10 20 20 4099_actions")
os.system("python run.py TVBPR 10 20 20 20 4099_actions")
os.system("python run.py timeSVD_VC 100 5 20 20 4099_actions")
os.system("python run.py timeSVD_VC 100 10 20 20 4099_actions")
os.system("python run.py timeSVD_VC 100 20 20 20 4099_actions")
os.system("python run.py timeSVD_VC 100 40 20 20 4099_actions")
