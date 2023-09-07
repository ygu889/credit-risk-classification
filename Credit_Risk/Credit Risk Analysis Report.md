# Module 20 Report 

## Overview of the Analysis

The purpose of this analysis is to develop a predictive model for assessing credit risk of the borrows and flag potential loan defaults. The goal is to analyze historical lending data and build a model that can classify borrowers into two categories: '0' for healthy loans (low credit risk) and '1' for high-risk loans (high credit risk). This analysis aims to assist company reducing financial risks associated with loan defaults.

The dataset  contains a variety of financial information related to borrowers, including the following features:
-Size of the loan
-Interest rate on the loan
-Borrower's income
-Debt_to_Income ratio
-The number of of accounts that the borrower uses regularly
-Derogatory marks on the borrower's credit report
-The borrower's total debt 
-The "loan_status" column serves as the target variable which indicates whether the borrower defaulted on the loan. A '1' indicates that the borrower defaulted, a '0' indicates a healthy loan.

In the dataset, there are 2,500 loans out of the 77,536 that are in default. This distribution suggests that there is a imbalance in the dataset, with a much larger number of healthy loans ('0') compared to high-risk loans ('1'). 


The original data was first split to remove the loan status column and save it as a separate data frame to be used as labels. The original data features were then split into training data and testing data to train and evaluate the machine learning model's performance.  The original training data was then fit to a Logistic Regression model to learn the relationship between features and loan status. The model was then used to predict the test data and indicators were generated for balanced accuracy score, confusion matrix, and a classification report. 

The original training data was then subjected to random oversampling of the loan default status. The purpose of this is to over-represent the default loans to balance the  distribution and potentially improve the model's performance, especially in handling high-risk/default loans. The resulting random data was composed of equal data points of healthy loans(0) and default loans (1). A new model was fit using Logistic Regression for the resampled data, and then this data was used to predict the resampled test data. Once again, a balanced accuracy score, confusion matrix, and classification report were generated from the results.


## Results

Using bulleted lists, describe the balanced accuracy scores and the precision and recall scores of all machine learning models.

* Machine Learning Model 1:
  * Description of Model 1 Accuracy, Precision, and Recall scores.
                  precision recall   f1-score   support

           0       1.00      1.00      1.00     18759
           1       0.87      0.89      0.88       625

    accuracy                           0.99     19384
   macro avg       0.94      0.94      0.94     19384
weighted avg       0.99      0.99      0.99     19384

-Precision: For label '0' (healthy loans), precision is perfect at 1.00, indicating that all the predicted '0's are indeed '0's. For label '1' (high-risk loans), precision is 0.87, which means that when the model predicts '1', it is correct about 87% of the time.
-Recall: For label '0', recall is perfect at 1.00, meaning the model identifies all '0's correctly. For label '1', recall is 0.89, which indicates that the model is capturing 89% of the actual '1's.
-F1-score: The F1-score is the harmonic mean of precision and recall. For label '0', the F1-score is 1.00, reflecting the perfect balance between precision and recall. For label '1', the F1-score is 0.88, showing a good balance between precision and recall but not perfect.
-The balanced accurary score was 94.426%. A balanced accuracy score of around 0.944 indicates that the logistic regression model is performing well in classifying both healthy and high-risk loans. It suggests that the model maintains a high level of accuracy while considering the class imbalance in the dataset.

* Machine Learning Model 2:
  * Description of Model 2 Accuracy, Precision, and Recall scores.
                precision    recall  f1-score   support

           0       0.99      0.99      0.99     56277
           1       0.99      0.99      0.99     56277

    accuracy                           0.99    112554
   macro avg       0.99      0.99      0.99    112554
weighted avg       0.99      0.99      0.99    112554

-Precision: For both label '0' and label '1', precision is 0.99, meaning that when the model predicts either '0' or '1', it is correct about 99% of the time. This high precision suggests that the model has a very low rate of false positives for both classes.
-Recall: For both label '0' and label '1', recall is also 0.99, indicating that the model is capturing 99% of the actual '0's and '1's. This high recall suggests that the model has a very low rate of false negatives for both classes.
-F1-score: The F1-score is the harmonic mean of precision and recall. For both label '0' and label '1', the F1-score is 0.99, showing an excellent balance between precision and recall for both classes.
-The balanced accuracy score of the resampled model is : 0.994180571103648. A balanced accuracy score of approximately 0.994180571103648 for the resampled model indicates that the model maintains a very high level of accuracy and generalization across both classes.
## Summary

Both machine learning models have demonstrated outstanding performance in distinguishing between healthy loans and high-risk loans, as indicated by their accuracy, precision, and recall scores. Howover, model 2 seems to perfomed better as since it has higher precision, recall and F-1 score for both 1 and 0. While both models are performing excellently, especially Model 2, continuous monitoring and periodic model updates may be necessary as the lending landscape and features impacting the target for both 1 and 0 evolve. 


