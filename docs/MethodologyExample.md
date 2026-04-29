# 3 METHODOLOGY

## 3.1 Datasets

### 3.1.1 AMC Dataset

The investigated population is included under the scheme of the Follow Me Pediatric Cardiology program at hospital clinics (Emma Children Hospital, Academic Medical Center, Amsterdam, Netherlands). Follow Me Pediatric Cardiology is an ongoing multidisciplinary outpatient program that collects baseline, follow-up, and outcome data for patients who receive early surgical correction for CHD, and that redeveloped into multiple clinical studies. Every patient in this research underwent an index operation, defined as the first cardiac surgery on cardiopulmonary bypass. All sensitive data potentially traceable to individual identity, including names, patient IDs, dates of birth, and dates of examination, are anonymized with a random combination of characters and numbers by AMC beforehand. Echocardiographic measurements were done in a standardized way according to the Guidelines and Standards for Performance of a Pediatric Echocardiogram [38]. Offline analysis of images saved in raw data format was performed to assess cardiac morphology and function. The AMC datasets consist of two parts: the major dataset records the clinical variables, including sex, weight, BSA, height, all the echocardiographic measurements and their respective units in 264 patients. The second dataset documents patients' preliminary diagnosis, dates of cardiac surgery, and their respective outcomes, which is prespecified as time-to-reoperation. Each patient is observed for 80 months with 15 records on average.

### 3.1.2 External Dataset

To further validate the clustering performance of the proposed method, an external public dataset is introduced from PhysioNet Computing in Cardiology Challenge 2019 [39]. The original dataset comprises 20,336 patients and their monitoring data across different time steps at intensive care units (ICU). This dataset is comprised of time-dependent variables, measured on a hourly basis, while the outcome is defined as time-to-event (occurrence of sepsis). To make it compatible with our experimental setting, we selected 13 variables based on the volume of missing data and transformed it into 39 time-steps.

## 3.2 Data Preparation

In the stage of exploratory data analysis, we found that the given dataset was highly imbalanced in terms of sampled time distribution, as shown in Figure 1. Due to the nature of patients with congenital heart disease, the first few months were highly sampled, while intervals between two examinations gradually prolonged when they grew up. This unevenly spaced time-series data could pose some challenges in clustering and required additional modeling.

There were 413 unique echocardiographic parameters in total; however, in over 90% of patients, only 17 of them were broadly measured. Using Pearson correlation coefficient to examine the linearity between variables, the overall low scores between the months after cardiac surgery and other echocardiographic parameters imply that the cardiac structures and functions are likely to develop in a non-linearly way (Appendix, Figure 8). We could also notice some strong correlations between certain variables, such as peak velocity and peak gradient. Although it is arguable whether variables with high co-linearity could harm the clustering accuracy, we did not discard these variables since we would like to evaluate models' performance to this end. Finally, based on the number of missing values, 14 time-dependent features were chosen as inputs for clustering, including body surface area (BSA), weight, left ventricular fractional shortening (%FS), aortic valve peak velocity (AV Vmax), aortic valve peak gradient (AV maxPG), interventricular septum diastolic thickness (IVSd), interventricular septum systolic thickness (IVSs), left ventricular internal diastolic dimension (LVIDd), left ventricular internal systolic dimension (LVIDs), left ventricle posterior wall diastolic thickness (LVPWd), left ventricle posterior wall systolic thickness (LVPWs), pulmonary valve peak velocity (PV Vmax), and pulmonary valve peak gradient (PV maxPG). One static categorical variable (sex) was represented as binary vectors using one-hot encoding. After consulting with clinical experts, we decided to focus on the first six years (72 months) after the cardiac surgery.

After exploring all time steps, we found the number of missing values enormous, over 80%. As most of the patients had a more frequent follow-ups in the first year, the number of missing values increased drastically after that. While time-series models usually assume a fixed-length time interval, namely having an observation every month in our case, we propose an uneven aggregation method to model these time-dependent variables. This method aggregates time-dependent variables based on their measurement frequency. At the first half-year, when patients' data were sampled more often, the time steps were kept as original with a monthly interval, so that less information would be sacrificed. Starting from the seventh month after the index operation, intervals of 3-month, 6-month, and 12-month were applied to aggregate the original values to form the new time steps. Even though the time steps were shortened to 15, this aggregated method can effectively decrease the number of missing values. Figure 2 demonstrates the realization of this proposed time steps aggregations. Simple linear imputation and backward filling were utilized to impute the remaining missing data given that they more intuitive in clinical interpretation.

## 3.3 Model Implementation

### 3.3.1 Baseline Models

To better compare the performance of our proposed method, we chose K-means, DBA, and K-shape as our baseline models. These methods are both raw-data-based clustering techniques and require a prespecified hyperparameter k value. K-means serves as a naive baseline since it directly measures the Euclidean distances among observations. Both DBA and K-shape apply certain distortions to better align two time-series, and they are therefore deemed more robust than K-means in time-series clustering [27, 29].

### 3.3.2 Proposed Method

Motivated by the success in applying deep learning, we propose an autoencoder-based deep neural network architecture to better model this problem. The autoencoder acts as the backbone of our proposed method. Since the latent representation does not necessarily form clusters with well-separated outcomes, we added an additional predictor to facilitate the feature learning. This allows the encoder to play the most important role in generating a well-represented vector space; adding the predictor can force the encoder to form vectors embedded with outcome signals. This adapts the embedding to the problem of interest.

The proposed method reduced input dimensions to a feature space before applying a clustering algorithm. First, we trained the autoencoder-based model with two tasks: the decoder focuses on reconstructing the latent representation back into the original time series, and the predictor focuses on predicting patients' outcomes. The total loss is therefore contributed by two tasks, a reconstruction loss and a prediction loss. After training the model, the whole dataset was fed to the encoder to generate the latent representation from the input. Clusters for each patient were subsequently assigned by using a standard clustering algorithm K-means on the embedded space. Figure 3 illustrates the overall process of how clustering is formed using the proposed architecture.

Focusing on the model architecture in more detail: the encoder, and decoder are formed of Long Short-Term Memory (LSTM), a recurrent neural network (RNN) deemed robust for sequential data. Unlike standard feedforward neural networks, LSTM has feedback connections and it can not only process single data points, but also entire sequences of data. LSTM is known for its memory capacity, meaning that it can memorize previously seen data. The characteristic of storing information over a period of time is realized by three different gates: a forget gate, an input gate, and an output gate. Each gate decides whether and how strong the previous signal can be passed to the current state and the next state. The predictor is made up of several layers of Convolutional Neural Network (CNN). CNN models learn an internal representation of inputs from different dimensions in a process referred to as feature learning. The same process can be harnessed on one-dimensional time series data and leverage the filtered features to predict outcomes. The overall details of respective layers can be found in Figure 4.

## 3.4 Experimental Setup

Before clustering, all the input variables were normalized to improve the efficiency of clustering algorithms and accelerate the training. The final input was reshaped into a three-dimensional matrix (182, 15, 15), representing 182 patients, 15 time-steps, and 15 variables. We used an elbow method and an average Silhouette width to determine the optimal k value for clustering, which have been commonly utilized as a criterion for selecting the number of clusters. The k value was found to be three and used in our analysis (Appendix, Figure 9).

The autoencoder-based architecture is optimized through backpropagation. The input firstly passes forward through the network to yield reconstructed time-series and the predicted outcome. After comparing the predictions with the original time-series and ground truth, the architecture updates its parameters based on the computed loss over the whole network. The total loss is composed of the reconstruction loss and the prediction loss, multiplied by their respective weights: $Loss_{total} = loss_{reconstruction} * w_1 + loss_{prediction} * w_2$

Both the AMC dataset and the external dataset were randomly split into training, validation, and test sets. The total training was set to be 200 epochs, and the training terminated early if the validation loss did not improve for 20 epochs to prevent overfitting. At each epoch, the training data was fed to the model and the model updated the trainable parameters to minimize the loss. The batch size was set to 12 and Adam was selected to be the learning rate optimizer during training. Our models were implemented using Python 3.7, tslearn 0.5, TensorFlow 2.4, and Keras 2.4 software.

## 3.5 Evaluation Metrics

The clustering results are generally assessed by clustering validation indexes (CVI). The external indexes require the externally supplied class labels to measure the similarity of formed clusters, while internal indexes evaluate the clustering structure itself without external labels or information. Given that the external ground-truth labels are not attainable in our research, the clustering results are evaluated by the internal indexes: Silhouette score and COP index.

### 3.5.1 Silhouette Score

The Silhouette score is a widely-used internal metric that measures the similarity between each sample and its assigned cluster (the intra-cluster distance) compared to other clusters (the nearest-cluster distance) [40]. The silhouette ranges from −1 to +1, where a high value indicates that the object is well matched to its own cluster and poorly matched to neighboring clusters. Values near 0 indicate overlapping clusters. Negative values generally indicate that a sample has been assigned to the wrong cluster, as a different cluster is more similar.

For data point $i$ in the cluster $C_i$ ($i \in C_i$), the mean distance between $i$ and all other data points in the same cluster is denoted as $a(i)$, where $d(i, j)$ is the distance between data points $i$ and $j$ in the cluster.

$$a(i) = \frac{1}{|C_i| - 1} \sum_{j \in C_i, i \neq j} d(i, j)$$

The smallest mean distance of data point $i$ to all points in any other cluster, which the sample is not a part of, is denoted as $b(i)$. The cluster with the smallest mean dissimilarity could be interpreted as the nearest cluster since it is the next best fit cluster for sample $i$.

$$b(i) = \min_{k \neq i} \frac{1}{|C_k|} \sum_{j \in C_k} d(i, j)$$

The silhouette value for each data point $i$ is therefore defined as follows:

$$s(i) = \frac{b(i) - a(i)}{\max\{a(i), b(i)\}}, \quad if \; |C_i| > 1$$

### 3.5.2 COP Index

The COP-index is the ratio of the tightness within the cluster to the farthest adjacent distance [41]. It can assess the quality of clustering results with an estimate of the intra-cluster variance (cohesion) in the numerator and an estimate of the inter-cluster variance (separation) in the denominator.

Given a dataset, $X = \{x_1, x_2, ...x_N\}$, a cluster $C$ is a subset of data points in the dataset. The centroid of a cluster is denoted as $\overline{C}$, whereas $P^Y = \{C_1, C_2, ..C_k\}$ is a set of disjoint clusters of a subset of the dataset.

$$COP(P^Y, X) = \frac{1}{|Y|} \sum_{C \in P^Y} |C| \frac{intra_{COP}(C)}{inter_{COP}(C)}$$

where

$$intra_{COP}(C) = \frac{1}{|C|} \sum_{x \in C} d(x, \overline{C})$$

$$inter_{COP}(C) = \min_{x_i \notin C} \max_{x_j \in C} d(x_i, x_j)$$

## 3.6 Statistical Analysis

The association across clusters is examined by descriptive statistics. The mean fluctuations of continuous variables, namely the difference between the maximum and minimum observed values, are used to determine whether there are any statistically significant differences within clusters with one-way ANOVA test. Categorical variables are described as percentages and compared by the Pearson chi-square test. We investigate the outcomes of patients with Kaplan-Meier curves and the logrank test to assess the association between clusters and corresponding prognosis.
