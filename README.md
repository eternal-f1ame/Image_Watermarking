# Computer Vision Course Project 1 (2023)

* This repository contains codes and demonstrations for the computer vision course project 1. Done by `Ayush Anand(B20CS082)` and `Aaditya Baranwal(B20EE001)`

## Instructions

- To test any of the watermarking techniques just run the relevant python file such as `python3 LSB.py`. Inside the <Algorithm>.py file replace the already existing image path with the path of the image you want to test on.

- Alternatively you can use it as a library. The demonstration for the same has been shown in the relevant test.ipynb file.

## Structure

- The folder "visual" consists of code relevant to the visual watermarking attack and dataset generation.
  - Modify the image directory path in the `prepare_dataset.py` as per your requirements.
  - first prepare the dataset by running the `prepare_dataset.py`. Make sure to create the folders `outputs` and `removal_results`.
  - Now run the `remove_watermark.py` file to see the results.
- The folder "util" contains helper facilities to run the encryption algorithms on different inputs and check their output.
- The file `metrics.py` consists of several metrics which we have implemented to evaluate the performance of the algorithms. To add another metric just create a function and add the mapping in the dictionary at the bottom of the file as has been illustrated.
