# Prompt-Based Image Retrieval

This repository contains code for the prompt-based image retrieval project. The goal of this project is to retrieve images based on a given prompt using a pre-trained image classification model.

## IMPORTANT NOTE:

Please note that the pre-trained image classification model used in this project has been stored on Google Cloud Storage.
To use it, you need to download the model file from that platform and place it in the model folder before running predict.py and image_retrieval.py. You can download the file from the following link: [Google Cloud Storage](https://drive.google.com/file/d/1HHBBgmF-HIenATO4MG2NkQmckvbW8wkJ/view?usp=share_link).

## Requirements

- Python 3.6+
- PyTorch
- numpy
- pandas
- tqdm
- torchvision
- timm

## Usage

1. Clone the repository:

```bash
git clone https://github.com/your_username/prompt-based-image-retrieval.git
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Modify the configuration in config.py according to your needs.

4. Run the following command to perform image retrieval:
```bash
python image_retrieval.py 
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

The pre-trained image classification model is from the [timm](https://github.com/rwightman/pytorch-image-models) library.
The image dataset used in this project is [ImageNet](http://www.image-net.org/).