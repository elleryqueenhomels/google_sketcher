# Google Sketcher

This repo is dedicated to build a simple yet effective <b>Convolutional Neural Network</b>, train it over the [Google Quick-Draw Dataset](https://github.com/googlecreativelab/quickdraw-dataset), and save it for further usage.

You can find a demo from [here](https://elleryqueenhomels.github.io/sketcher/).

## Description

The trained CNN can take any doodle image as input, and "guess" what the doodle describes within [345 categories](https://github.com/elleryqueenhomels/google_sketcher/blob/master/categories.txt).

- CNN Architecture:<br/>![cnn_architecture](https://user-images.githubusercontent.com/13844740/43365984-2d96416a-9368-11e8-972b-d8ca1e40ef3b.png)

- Manual:
1. run `python3 main.py` - The script will automatically download training dataset, prerpocess data, build the CNN, train the CNN, and save the model.
2. run `pip3 install tensorflowjs` - The command will automatically install the latest version of TensorFlowJS into your machine.
3. run `bash convert.sh` - The script will automatically convert the trained model into a TensorflowJS compatible model, so that we can use the trained model to do inference on a web application.

## Trained Model
- You can find my trained model in the `model` directory.
- You can also download the trained model and the training log file in [AWS S3](https://s3.console.aws.amazon.com/s3/buckets/wengaoye/sketcher-model-v1/?region=us-west-2&tab=overview).

## My Running Environment
<b>Hardware</b>
- AWS EC2 - r4.16xlarge
- CPU: Dual socket Intel® Xeon™ E5 Broadwell Processors (2.3 GHz) - vCPU x 64
- Memory: 488GB DDR4

<b>Operating System</b>
- Amazon Linux AMI 2018.03

<b>Software</b>
- Python 3.6.3
- NumPy 1.13.3
- TensorFlow 1.8.0
- Keras 2.1.6

## References
- [Google Quick-Draw Dataset](https://github.com/googlecreativelab/quickdraw-dataset)
- Zaid Alyafeai's [Tutorial](https://medium.com/tensorflow/train-on-google-colab-and-run-on-the-browser-a-case-study-8a45f9b1474e)

## Citation
```
  @misc{ye2018googlesketcher,
    author = {Wengao Ye},
    title = {Google Sketcher},
    year = {2018},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/elleryqueenhomels/google_sketcher}}
  }
```
