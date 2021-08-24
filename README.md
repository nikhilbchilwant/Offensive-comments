# Offensive-comments

Immplementation of data selection with multi-task learning method to detect offensive comments. The ideas was taken from [Domain Adaptation with BERT-based Domain Classification and Data
Selection](https://assets.amazon.science/d6/7c/c2954857435a921ab0fb988e9caa/domain-adaptation-with-bert-based-domain-classification-and-data-selection.pdf).

The Scikit classification report:

                   precision    recall  f1-score   support

    non-offensive       0.81      0.95      0.88       212
        offensive       0.95      0.83      0.89       268

         accuracy                           0.88       480
        macro avg       0.88      0.89      0.88       480
     weighted avg       0.89      0.88      0.88       480
     
I cannot share the target dataset as it is owned by Eternio GmbH.

Branches:
- domain-classifier : Trains BERT to find the probability with which a comment in the dataset (say Germ Eval 2017) will belong to the Eternio dataset (target dataset).
- domain-adaptation-single-task : Supports fine tuning the BERT model for a single task.
- mtl : Supports fine tuning the BERT model for a multiple tasks. I referred [MT-DNN](https://github.com/microsoft/MT-DNN) for this.
