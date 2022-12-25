# K-means clustering

## Goal:
Try to diagnose breast cancer based solely on a Fine Needle Aspiration (FNA), which as the name suggests, takes a very small tissue sample using a syringe

# Wisconsin Diagnostic Breast Cancer dataset

Dataset available at https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(diagnostic)
### Attribute: Domain
Sample Code Number: id number 

Clump Thickness: 1-10 

Uniformity of Cell Size: 1-10 

Uniformity of Cell Shape: 1-10 

Marginal Adhesion: 1-10 

Single Epithelial Cell Size: 1-10

Bare Nuclei: 1-10

Bland Chromatin: 1-10

Normal Nucleoli: 1-10

Mitoses: 1-10

Class: 2 for benign, 4 for malignant

# Implementation

- Implement a function that performs K-means clustering from scratch
- Load the Wisconsin Diagnostic Breast Cancer dataset : Data matrix with D = 10 features and N = 699 samples
- First without the class label, run your algorithm. Then compare with the class labels and get the accuracy
- Run the algorithm several times, starting with different centers
- Try using a supervised method to check if we get a better result
