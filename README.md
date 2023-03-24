# Computer Vision Course Project 1 (2023)

___
[Report](/documents/report.pdf) -  [Demo-video](https://www.youtube.com/watch?v=jkqibgdkF3k) - [Presentation](/documents/presentation.pdf)

* This repository contains codes and demonstrations for the Computer Vision Course Project 1 on `Image Watermarking`.

___

## Demo video

 [![yt](/documents/thumbnail.png)](https://www.youtube.com/watch?v=jkqibgdkF3k)

___

## Installing requirements with conda

* Run the following commands in the master root to create a new virtual env to run the files in local:

```shell
conda create -n <ENV NAME> python=3.9
conda activate <ENV NAME>
conda install -r requirements.txt
```

___

## Instructions

* To test any of the watermarking techniques just run the relevant python file such as `python3 LSB.py`. Inside the <Algorithm>.py file replace the already existing image path with the path of the image you want to test on.

* Alternatively you can use it as a library. The demonstration for the same has been shown in the relevant test.ipynb file.

___

### Dataset description

* Dataset has two folders: `sub` and `super`
* `100_Image_Dataset` has 100 Custom Images
* `CODO_Dataset` has 128 Images from COCO Dataset 2017

___

## Structure

* The folder "visual" consists of code relevant to the visual watermarking attack and dataset generation.
  * Modify the image directory path in the `prepare_dataset.py` as per your requirements.
  * first prepare the dataset by running the `prepare_dataset.py`. Make sure to create the folders `outputs` and `removal_results`.
  * Now run the `remove_watermark.py` file to see the results.

* The folder "util" contains helper facilities to run the encryption algorithms on different inputs and check their output.
* The file `metrics.py` consists of several metrics which we have implemented to evaluate the performance of the algorithms. To add another metric just create a function and add the mapping in the dictionary at the bottom of the file as has been illustrated.
* For more comprehensive analysis and comparison, refer to [report](/documents/report.pdf).

___

### Contributors

> Aaditya Baranwal baranwal.1@iitj.ac.in ;  Github: [eternal-f1ame](https://github.com/aeternum) <br>
> Ayush Anand anand.5@iitj.ac.in ; Github: [iamayushanand](https://github.com/iamayushanand)
