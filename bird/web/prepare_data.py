# coding=utf-8
"""Prepare data.
"""
import os
for i in range(1, 1769):
    output_dir = 'test_set/{:04}'.format(i)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    file_path = 'test_set/{}.jpg'.format(i)
    if not os.path.exists(file_path):
        print(file_path)
        continue
    os.rename(file_path, 'test_set/{:04}/{}.jpg'.format(i, i))

