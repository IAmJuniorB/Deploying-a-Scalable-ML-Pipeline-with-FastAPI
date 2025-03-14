# Model Card

## Model Details
- **Model Type**: RandomForestClassifier
- **Number of Estimators**: 100

## Intended Use
This model is intended to predict whether an individual's income exceeds $50,000 based on demographic features.

## Training Data
- **Dataset**: Census data
- **Features**: age, workclass, education, marital-status, occupation, relationship, race, sex, native-country
- **Label**: salary (<=50K or >50K)

## Evaluation Data
- **Dataset**: Test set from the census data
- **Metrics**: Precision, Recall, F1 Score

## Metrics
- **Precision**: 0.7425
- **Recall**: 0.6334
- **F1 Score**: 0.6836

## Ethical Considerations
The model may have biases present in the training data, such as biases against certain demographic groups.

## Caveats and Recommendations
- **Data Quality**: The model's performance is dependent on the quality of the training data.
- **Future Improvements**: Consider using techniques like feature engineering or hyperparameter tuning to improve model performance.