# AN EFFICIENT CNN FOR SPECTRAL RECONSTRUCTION FROM RGB IMAGES

See the relevant paper at (https://arxiv.org/abs/1804.04647)


### Prerequisites

Tensorflow - Written with version 1.4.1

### How to Use

After getting the files if you want to run the tests mentioned in the paper use tester.py. In tester.py modify the relevant folder paths to point to checkpoints and data.

There are models trained on ICVL, NUS and CAVE dataset. Apart from the NUS dataset, the results are obtained by dividing the set into 2 and training on one and testing on the other. The images in each set are given in the corresponding .txt files.

In order to train a model, use rgb_hs_main.py. 


## Authors

* **Yigit Baran CAN** 
* **Radu Timofte** 

## License

This project is licensed under the Apache License - see the [LICENSE.md](LICENSE.md) file for details




