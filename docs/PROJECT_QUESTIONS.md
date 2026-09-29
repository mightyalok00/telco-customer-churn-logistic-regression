# Telco Customer Churn — Project Question Bank

The project is organized into 18 analytical sections covering business analysis, statistics, Logistic Regression, model interpretation, risk segmentation, and business impact.

## 01. Business Problem & Objectives

1. What is the business problem of customer churn?
2. What is the overall customer churn rate?
3. What business factors may be associated with customer churn?
4. Which customer segments contribute most to overall churn?
5. What measurable customer characteristics can be used to understand churn risk?
6. How can predictive analytics support customer-retention decision making?

## 02. Data Understanding

7. What is the structure, size, and dimensionality of the dataset?
8. What does each feature represent?
9. Which variables are numerical, categorical, binary, or identifier variables?
10. What is the target variable?
11. What does each target class represent?
12. What is the distribution of the churn target?

## 03. Data Quality & Validation

13. Are there duplicate customer records?
14. Which columns contain missing values?
15. Are there blank or whitespace-only values?
16. Are there invalid, inconsistent, or unexpected values?
17. Are numerical variables stored with the correct data types?
18. Are categorical variables represented consistently?
19. Are customer identifiers unique?
20. Are there potential data-quality issues that could affect modeling?
21. Are there outliers or extreme numerical observations?
22. Is the target variable imbalanced?

## 04. Exploratory Data Analysis

23. What is the distribution of customer tenure?
24. What is the distribution of monthly charges?
25. What is the distribution of total charges?
26. How does churn vary across demographic characteristics?
27. How does churn vary across service characteristics?
28. How does churn vary across contract types?
29. How does churn vary across billing characteristics?
30. How does churn vary across payment methods?
31. Which features show visible differences between churned and retained customers?

## 05. Deep Business Analysis

32. How does customer tenure relate to churn?
33. How do month-to-month, one-year, and two-year contracts differ in observed churn?
34. How do monthly charges relate to churn?
35. How do total charges relate to churn?
36. How does churn vary across internet-service types?
37. How do payment methods differ in observed churn?
38. How does paperless billing relate to churn?
39. How do online security and technical-support services relate to churn?
40. How do online backup and device-protection services relate to churn?
41. How does phone-service usage relate to churn?
42. How does multiple-line usage relate to churn?
43. How does senior-citizen status relate to churn?
44. How do partner and dependent status relate to churn?
45. Do customers with multiple subscribed services show different churn behavior?
46. Which customer segments have the highest observed churn rates?
47. Which customer segments contain the largest number of churned customers?
48. Which customer segments combine high churn with high revenue exposure?
49. What observable characteristics define high-risk customer segments?

## 06. Statistical Analysis

50. Which categorical variables have the strongest association with churn?
51. Which numerical variables differ most between churned and retained customers?
52. Are the observed relationships statistically meaningful?
53. Which variables provide the strongest predictive signal?
54. Are there highly correlated numerical predictors?
55. Are there redundant or overlapping categorical predictors?
56. Could multicollinearity affect Logistic Regression coefficients?
57. Which variables remain informative after controlling for other predictors?

## 07. Feature Engineering & Preprocessing

58. Which columns should be excluded from modeling and why?
59. How should missing values be handled?
60. How should numerical variables be scaled?
61. How should categorical variables be encoded?
62. Does one-hot encoding improve model usability?
63. Should numerical and categorical preprocessing be performed through a pipeline?
64. Can preprocessing be performed without data leakage?
65. Which engineered features could provide additional business insight?
66. Does feature engineering improve predictive performance?

## 08. Logistic Regression

67. How accurately can Logistic Regression predict customer churn?
68. What is the appropriate baseline model?
69. How does feature scaling affect Logistic Regression?
70. How does one-hot encoding affect Logistic Regression?
71. How does regularization strength affect model performance?
72. How does L1 regularization compare with L2 regularization?
73. How does class weighting affect churn detection?
74. How does Logistic Regression perform under different hyperparameter settings?
75. How stable is Logistic Regression performance across cross-validation folds?
76. Does removing redundant features improve model stability?

## 09. Model Evaluation

77. What are the model's Accuracy, Precision, Recall, and F1-score?
78. What are the ROC-AUC and PR-AUC scores?
79. What is the model's Log Loss?
80. How well does the model identify actual churners?
81. How many churners are missed by the model?
82. How many retained customers are incorrectly classified as churners?
83. What does the confusion matrix reveal?
84. How does Logistic Regression compare with a baseline classifier?
85. How does Logistic Regression compare with Decision Tree?
86. How does Logistic Regression compare with Random Forest?
87. What are the trade-offs between predictive performance and interpretability?

## 10. Class Imbalance & Threshold Analysis

88. How does class imbalance affect churn prediction?
89. Does class weighting improve minority-class detection?
90. How do different probability thresholds affect Precision and Recall?
91. What threshold provides a useful trade-off between false positives and false negatives?
92. How does threshold selection affect F1-score?
93. How does threshold selection affect the number of customers flagged as high risk?
94. What are the potential business consequences of different classification thresholds?

## 11. Model Interpretation

95. Which features have the largest positive Logistic Regression coefficients?
96. Which features have the largest negative coefficients?
97. Which variables are associated with increased churn odds?
98. Which variables are associated with decreased churn odds?
99. How can Logistic Regression coefficients be converted into odds ratios?
100. What do the odds ratios imply about customer churn?
101. Which customer characteristics have the strongest independent association with churn probability?
102. Are the coefficient directions consistent with the exploratory analysis?
103. How stable are coefficients across cross-validation folds?

## 12. Probability & Calibration

104. Which customers have the highest predicted probability of churn?
105. Are predicted churn probabilities well calibrated?
106. How closely do predicted probabilities correspond to observed churn rates?
107. Does probability calibration improve the usefulness of the model?
108. Can predicted probabilities be converted into meaningful customer-risk groups?

## 13. Customer Risk Segmentation

109. Which customers fall into low-, medium-, and high-risk churn groups?
110. What characteristics are common among high-risk customers?
111. How does risk vary by contract type?
112. How does risk vary by tenure?
113. How does risk vary by monthly charges?
114. How does risk vary by service usage?
115. Which customer segments have the highest predicted churn probability?
116. Which high-risk segments contain the greatest number of customers?
117. Which high-risk segments represent the greatest potential business exposure?

## 14. Business Impact Analysis

118. What proportion of customers are classified as high churn risk?
119. How many customers could potentially require retention attention?
120. Which high-risk customer segments should be investigated first?
121. What customer characteristics should be considered when designing retention strategies?
122. How could churn probabilities support customer-retention prioritization?
123. How could the model be incorporated into a customer-risk monitoring process?
124. What additional business data would improve retention analysis?
125. What information is missing from the dataset that could affect business interpretation?

## 15. Model Robustness & Reliability

126. How consistent is model performance across cross-validation folds?
127. How sensitive is the model to different preprocessing choices?
128. How sensitive are coefficients to changes in the training data?
129. Does removing correlated or redundant variables improve stability?
130. Does the model generalize to unseen customers?
131. Are the model's predictions reliable across different customer segments?
132. What limitations should be considered before deploying the model?

## 16. Model Comparison

133. How does Logistic Regression compare with Decision Tree?
134. How does Logistic Regression compare with Random Forest?
135. How do the models differ in Precision, Recall, F1-score, ROC-AUC, and PR-AUC?
136. How do the models differ in interpretability?
137. How do the models differ in probability calibration?
138. How do the models differ in complexity and maintainability?
139. What additional complexity is introduced by more advanced models?

## 17. Final Business Analysis

140. What are the strongest observable factors associated with customer churn?
141. Which customer segments demonstrate the greatest observed churn?
142. Which customer segments have the greatest predicted churn risk?
143. Which factors are associated with increased or decreased churn odds?
144. How accurately can customer churn be predicted?
145. How reliable are the predicted churn probabilities?
146. What evidence supports using the model for customer-risk analysis?
147. What limitations prevent the model from being interpreted as causal evidence?
148. What additional data could improve future churn prediction?
149. What further experiments or analyses should be conducted?

## 18. Project Conclusion

150. What did the analysis reveal about customer churn?
151. What did Logistic Regression reveal that descriptive analysis alone could not?
152. Which features provide the strongest predictive signal?
153. How well does the final model generalize to unseen customers?
154. What are the main analytical limitations?
155. What are the main modeling limitations?
156. What business questions remain unanswered?
157. How could this project be extended into a production churn-monitoring system?
