# Overview
This repository contains a project focused on Convolutional Neural Networks (CNNs). Experiments were conducted on the MNIST, CIFAR-10, and CIFAR-100 datasets.

After training the models from scratch, fine-tuning was performed on a pre-trained model using a different dataset. The models implemented include: Multilayer Perceptrons (MLPs); Convolutional Neural Networks, with a focus on ResNet architectures.

The code does not necessarily need to be executed. If you’re only interested in the results, you can simply refer to the report provided in the `relazione.pdf` file.

---

# Install
Clone the repository and install the requirements:

```bash
git clone https://github.com/edoardosarri24/CNN.git
cd CNN
conda env create -f environment.yml -n edoardosarri
conda activate edoardosarri
wandb login
```

# Run
To run the code:
- Open the notebook called main.ipynb`.
- If you wish to save the images, set the flag under the imports to True and set your desired path.
- Run the notebook.
