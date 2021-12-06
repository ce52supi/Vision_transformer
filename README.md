# Vision_transformer

In this project, Vision Transformer(ViT) model is implemented in keras and tensorflow for the prediction of invasive ductal carcinoma (type of breast cancer). The original dataset consisted of 162 whole mount slide images of Breast Cancer (BCa) specimens scanned at 40x. From that, 277,524 patches of size 50 x 50 were extracted (198,738 IDC negative and 78,786 IDC positive). There is a huge data imbalance between positive and negative samples. Performed random undersampling for the class imbalance.

After undersampling,achieved a classification accuracy of about 85%.

Dataset: https://www.kaggle.com/paultimothymooney/breast-histopathology-images
