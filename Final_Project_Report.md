# **DATASCI223 Final Project - Faezeh Mirlohi**

# **1. Problem Description and Dataset**

## **1.1 Problem Description**

The core problem addressed in this project is image classification. Specifically, the goal is to accurately classify images into one of 100 distinct categories. This is a multi-class classification challenge where a given input image needs to be assigned the correct label from a predefined set of classes. In machine learning, especially with deep learning models, larger and more complex models often achieve higher accuracy but are computationally expensive, making them unsuitable for deployment on devices with limited resources (like mobile phones). This project aims to investigate Knowledge Distillation (KD) as a technique to mitigate this issue. Knowledge Distillation involves transferring knowledge from a large, complex "teacher" model (which achieves high accuracy) to a smaller, more efficient "student" model, enabling the student to perform better while being more resource friendly (with significantly fewer parameters and lower computational requirements).

## **1.2 Dataset: CIFAR-100**

The dataset used for this project is CIFAR-100. It is a widely used benchmark dataset in computer vision, particularly for image classification tasks. CIFAR-100 has 60,000 tiny color images with 100 different classes. The images in CIFAR-100 represent a wide variety of objects and animals, making it a challenging dataset for classification due to the subtle differences between some classes and the small image size.

**Table 1. CIFAR-100 Dataset Overview**

| Feature           | Description      |
|-------------------|------------------|
| Total Dataset     | 60000            |
| Train set         | 50000            |
| Test set          | 10000            |
| Data Type         | Images (PNG)     |
| Image Dimensions  | 32x32 pixels     |
| Image Channels    | 3 channels (RGB) |
| Number of Classes | 100              |

**Figure 1. CIFAR-100 Dataset Images**

![](Screenshot%202025-05-30%20at%2014.49.21.png)

## **1.3 Method:** Knowledge Distillation From ResNet32x4 to CNN (with 4 layers)

CIFAR-100 has 100 classes and small images, making it challenging. ResNet models are powerful enough to learn complex features from such data, however, it is a large model. The main goal is to get a smaller, faster model (student) that performs well. Knowledge distillation is perfect for this, as it transfers knowledge from a big, strong teacher to a smaller, more efficient student. Thus Knowledge Distillation is used in this project, applied to image classification using CNN model (with 4 convolutional layers) as the student and ResNet32x4 model as the teacher. The teacher model is usually very accurate but too big for real-world use on small devices. The student model is smaller and faster, but usually less accurate. Knowledge distillation helps the student model learn "soft targets" (class probabilities) from the teacher, not just the "hard targets" (the correct class labels). This extra information from the teacher helps the student model become much better than if it only learned from the hard labels.

## **1.4** Models

### **1.4.1 Teacher Model (ResNet32x4)**

For the teacher model, I specifically chose ResNet32x4 (With 7,433,860 parameters). ResNet is a type of convolutional neural network that is very good at image tasks. This model has 32 layers with channel width multiplicated by 4 compared to a standard ResNet. This increased width makes it a very powerful and accurate model, capable of learning complex patterns from the CIFAR-100 dataset, hence an excellent choice as a teacher that provides high-quality knowledge. It introduces residual connections that allow the network to learn more easily, even when it's very deep. This helps solve the problem of vanishing gradients. The specific teacher model used in this project was downloaded and utilized from the [SimKD GitHub repository](https://github.com/DefangChen/SimKD).

### **1.4.2 Student Model (Smaller CNN)**

My student model is a compact CNN (with 799,652 parameters) designed for image classification. It has four convolutional layers that progressively extract features, starting from a 3 channel input and expanding to 256 output channels. These layers primarily use ReLU activation functions and incorporate max pooling to reduce spatial dimensions. The extracted features are then flattened and fed into a final linear layer, which produces 100 output values corresponding to the different classes. For training the student model, Stochastic Gradient Descent (SGD) was used as the optimizer with a learning rate of 0.1 and weight decay set to 0.0005 to help prevent overfitting.

The loss function chosen for training without knowledge distillation was CrossEntropyLoss, which is well-suited for multi-class classification tasks. To improve learning stability, a learning rate scheduler (StepLR) was applied, which reduces the learning rate by a factor of 0.2 every 30 epochs. The model was trained for 100 epochs and the training process was repeated five times to ensure consistent and reliable results. All training was performed on the Mac GPU using Apple’s Metal Performance Shaders (MPS) to speed up computation.

For training knowledge distillation, two types of loss functions were used to guide the student model: cross-entropy loss and KL divergence loss. The cross-entropy loss compares the student model’s predictions to the true class labels and ensures it learns to classify correctly. The KL divergence loss, on the other hand, measures how close the student’s softened output probabilities are to those of the teacher model. This is done using a temperature scaling factor (`loss_temp`) that smooths the probability distributions, making it easier for the student to learn from the teacher’s confidence across all classes. The KL divergence loss is scaled by the square of the temperature to keep gradients in balance. By combining both losses, the student learns not only from the correct labels but also from the teacher’s richer knowledge about the relationships between classes, which helps improve its overall performance.

Fewer layers and fewer parameters make the student smaller in terms of model size (memory footprint) and computational cost (fewer operations during inference) and faster compared to teacher. Therefore, less memory and processing power are needed, making the model suitable to run on small devices. The student CNN, despite its smaller size, can still benefit greatly from the rich, distilled knowledge of the teacher. It aims to achieve better performance comparable to a model trained without knowledge distillation.

# 2. Code

## 2.1 Dependecies

Code dependencies included in requirements.txt are:

-   torch

-   torchvision

-   matplotlib

-   tqdm

-   numpy

-   scikit-learn

-   torchsummary

## 2.2 Issues

1.  I tried a 3 layered CNN as my student. However, it seemed that my student was not sufficiently large enough to learn from the teacher efficiently.

2.  I experimented with 5- and 6-layer CNNs as student models, but they were too computationally intensive to run efficiently on my laptop.

3.  Training times were quite long, especially when using knowledge distillation.

## 2.3 Decisions

1.  Student CNN were changed from 3 layers to 4 layers to improve learning form the teacher.
2.  I decided to use the 4-layer CNN as the student model because it was large enough to learn effectively from the teacher while keeping processing time reasonable.
3.  To reduce training time, all processing was done on the GPU using my Mac's built-in processing units.

## **2.4 Experimental Setup**

1.  The CIFAR-100 dataset is downloaded and split into two parts: a training set and a test set. Each image in the dataset is transformed before being fed into the model.

2.  For training data, random cropping and horizontal flipping are applied to introduce variety and help the model generalize better. Then, all images are converted to tensors and normalized using the CIFAR-100 dataset's mean and standard deviation values.

3.  The test images are only converted to tensors and normalized, without any random changes, to keep evaluation consistent. The images are grouped into batches of 128 and loaded using data loaders.

4.  The training loader shuffles the data for randomness, while the test loader does not. These preprocessed batches are then passed to the model during training and evaluation.

5.  The student model was trained without knowledge distillation (with standard cross-entropy loss).

6.  The student model was trained with knowledge distillation, using a combination of the standard cross-entropy loss (comparing student predictions to true labels) and the KL divergence loss (comparing student probabilities to teacher probabilities).

7.  For training with knowledge distillation, I manually tuned the values of KL divergence weight and temperature to find the best combination. I tested values of 0, 0.25, 0.5, and 0.75, which control the weight of the KL divergence loss in the total loss function, and temperature values of 1, 2, 4, and 8, which affect the softness of the predicted probabilities. The best improvement in accuracy due to knowledge distillation was achieved with KL divergence weight set to 0.25 and temperature set to 4.

**Figure 2. Training Loss Plot without Knowledge Distillation**

![](Screenshot%202025-06-10%20at%2020.01.34.png)

**Figure 3. Training Loss Plot with Knowledge Distillation**

![](Screenshot%202025-06-10%20at%2020.02.24.png)

8.  Then, all models, student without KD, student with KD and teacher model were tested on the test set and are reported below.

# **3. Results and Discussion**

## **3.1 Key Results**

**Table 2. Student and Teacher Models Accuracy (on test set)**

| Model                | Accuracy (Mean ± Standard Deviation\*) |
|----------------------|----------------------------------------|
| Teacher (ResNet32x4) | 78.33%                                 |
| Student (without KD) | 53.72% ± 0.50%                         |
| Student (with KD)    | 56.39% ± 0.46%                         |

\*Student models were trained for 5 times, and the results are reported as Mean ± SD of all five models.

## **3.2 Discussion of Results**

The student model was first trained without knowledge distillation, using only the standard cross-entropy loss. After 100 epochs and averaging over 5 runs, it achieved a test accuracy of 53.72% with a standard deviation of 0.50%, indicating stable baseline performance. Subsequently, knowledge distillation was applied using the same training setup. A manual grid search identified the optimal distillation parameters: temperature (T) = 4 and alpha (α) = 0.75. With these settings, the student model achieved an average test accuracy of 56.39% with a standard deviation of 0.46%, representing a 2.67% improvement. For reference, random guessing on CIFAR-100 yields only 1% accuracy, highlighting the effectiveness of the student model’s architecture. Although the student does not match the performance of the larger ResNet32x4 teacher model, it offers a strong trade-off between accuracy and efficiency. The observed improvement confirms that the student model successfully leveraged the teacher’s soft targets, demonstrating the value of knowledge distillation.

# 4. References

1.  Hinton, Geoffrey, Oriol Vinyals, and Jeff Dean. 2015. *Distilling the Knowledge in a Neural Network*. arXiv. <https://arxiv.org/abs/1503.02531>.
2.  Knowledge Distillation with the Reused Teacher Classifier (CVPR-2022) <https://arxiv.org/abs/2203.14001>.
