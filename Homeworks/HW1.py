### Homework 1 Answers

# 1) Machine learning is simply the concept of instructing approches (algorithms) to machines for solving big and/or complex problems which cannot be solved otherwise. 
# Machine learning is the sub-branch of artificial intelligence.

# 2) The supervised and unsupervised learning differentiates based on a single main reason. While in supervised learning, the data set provided to the model contains labeled information. In contrast, the data set is not labled beforehand in unsupervised learning. 
# In other words, supervised learning algorithms require input which is pre classified so the data points are known by the model hence it is not required for model to understand the data. However, the data should be categorized initially then, the model (algorithm) can predict. 
# From the nature of the data set given to the model in supervised learning, the number of classes (labels) are known, the end product of model is classifying the future observations, and model compeltely depends on a training set. For the unsupervised learning, model does not have any knowledge about
# the inputs so that data labels not known as well, and it is utilized to understand the data as oppose to using for predicting new values and understanding existing relationships of the data.
# Mostly, regression (simple linear, multiple linear, polynomial) and classification (k-nearest neighbour, naive bayes, random forest) analysis can be provided as examples for the supervised; 
# clustering (k-means, density based, hierarchical) and dimension reduction (factor analysis, principal components and independent component analysis) are for unsupervised learning algorithms.

# 3) Any data set is priorily splitted into two or three parts based on what algoirthm to apply for. Sometimes the preference of selecting the train and test split number depends on the user.
# In most cases, data set is divided into two sections as train and test sets. As the name implies, train set is for training the machine learning model for building it for the prediction, test set is
# to test the applicability/performance of the built-model. Frequently, the whole data is fragmented by 70 - 30% for training and testing, respectively. In cases where the data sets divided into three, the one additional part is performed after the model is trained. The paramteres of the trained model
# is determined by the different set of data (validation set) so that the model neither over- nor under-fit as well as more optimized model could be utilized for testing the model with test data. This approach
# may enable model to provide us more likely to be appropriate and/or predictable results. In order to find the optimum parameters, we can use cross validation which divides whole data into small sub-parts randomly.
# Then, when a single small part is used for testing the models for testing the parameters whether they are best or not, the remaining small sub parts of the whole data use as trying parameters for building desirably optimum model.

# 4) Although data pre-processing steps can vary from source to source, I will use this course's slide materials as an answer. According to the "Data Preparation" slides. 
# There may be data duplications due to copying the same instances twice or so or during data generation process. These duplicate values, in most cases, are removed so as 
# to not give that particular data object an advantage or bias, when running machine learning algorithms. Besides, the data set we would like to use might be imbalanced.
# An Imbalanced dataset means the number of instances of a class(es) are significantly higher than another class(es), thus leading to an imbalance between the data itself 
# which could be tricky for model to asses and infer knowledge from the data set equally. The data may involves some missing values which must be taken care of by either removing
# or replacing with an sufficient values. To replace the missing values with the meaninigful ones without disturbing integrity of the whole data set, depending on the data type
# whether it is categorical, numerical, boolean or continous, discrete or data set distributions a mean, mode, median values could be selected to replace the missing value. In addition,
# KNN kind of approach could also be performed for identifying possible position of the unknown value inside the data and assigning it to the overal value based on the outcome.
# The data probably will ahve outliers which are the rare instances of data which does not fit any of the class. In most cases, they are just errors or statistacally ignorable observations
# Hence, it is important to detect the outliers. There are several ways to recognize and decide on whether the points are outliers or prominent sample points. These approches are
# standart deviation, box plots / IQR calculation, and isolation forest algorithm from sckit learning. The standart deviation and IQR are more robust approaches for outlier detection
# but isolation forest is an algoirthmic and complex approach for detection. The principle of IQR and standart deviation is if a given sample point(s) are above the (IQR/standart deviation)
# threshold then, the point(s) is outlier. The data set contents may not be in line with each other in terms of units, range, degrees of magnitute and etc. When this is the case,
# the data should be sacaled so that each sample of data should be same in terms of their diferences. For this purpose, different techniques are available. The gradianet and 
# distance based algorithms. The most common and basic ones are standardization and normalization. The feature scaling improves the performance of some machine learning algorithms
# and does not work at all for others. Normalization is a scaling technique in which values are shifted and rescaled so that they end up ranging between 0 and 1. Standardization 
# is another scaling technique where the values are centered around the mean with a unit standard deviation. This means that the mean of the attribute becomes zero and the resultant
# distribution has a unit standard deviation. The prefernce of these two technique depends on algorithm to be used and the problem itself. Normalization is good to use when you 
# know that the distribution of your data does not follow a Gaussian distribution as oppose to standardization. Standardization can be helpful in cases where the data follows a 
# Gaussian distribution and does not have a bounding range. So, even if there are outliers in the data, they will not be affected by the technique. The data values can be diverged
# and it makes it difficult for model to differentiate (categorize) them for training and testing or similarly the noise data may flactuate the data distribution so that some sample
# points are so intertwined that the points cause unordered distribution. These two situations can be solved by binning (bucketing) the data set. The binning is simply seperating
# the data into small but reasonable chunks and generalizes all the data points found at a selected data chunk according to the values of all points. Therefore, the bins are the
# representation of all the points found within the range of that bin and are replaced by a general value calculated for that bin.
# Sometimes data set is hugh and involves many features. The increase of features computational overwheelming for the model this is because there will be many classes to provide wide range
# of information which most of the time makes unfeasible for model to comprehend the data appropriately. The problem of selecting some subset of a learning algorithm’s input variables
# upon which it should focus attention, while ignoring the rest is called feature extraction (selection). Especially when dealing with a large number of variables there is a need for 
# dimensionality reduction a.k.a feature selection. Performing feature selection can significantly improve a learning algorithm’s performance. The main idea of dimensionality reduction
# or feature extraction is the elemination of unnecessary or removal of group of dependent features with each other from the data set but leave only one. There are various approaches
# to perform feature selection suchg as principle components analysis, independent component analysis, linear discriminant analysis, and factor analysis are some of them.
# Similar to feature selection, features can also be tranformed into a different data type or represented differently when the existing data set is not plausible to be given as input
# to model due the possible apprehension problems of computer to interpret the data. After the feature encoding, model can be easily accepted as input for machine learning algorithms 
# while still retaining its original meaning. Depending on the type of data, nominal or ordinal, different apporpaches should be applied. For nominal data, such as marriage status, colors,
# car brands, ethnicity and etc, the string values can be represented as binary like One-Hot Encoding or multiplary encodings. Moreover, ordinal values are An order-preserving change 
# of values, such as football league rankings or pain scale, the string values can be represented as sigle integers by preserving the order information between the values.
# Lastly, the data set is required to be divided into two or three irrespective of the algorithm. These parts are crucial for model to be built appropriately. Machine Learning algorithms 
# has to be first trained on the data available and then validated and tested. The train and test sets which are obtained from the data set by dividing the whole data set randomly to 70% - 30%.
# Besides, we want to find find the optimum parameters for the model, in that case we can use cross validation method which divides whole data into small sub-parts randomly.
# Then, when a single small part is used for testing the models for testing the parameters whether they are best or not, the remaining small sub parts of the whole data use as trying parameters
# for building desirably optimum model.

# In data pre-processing, user have a chance to visualize, understand and catch the relationships within the data set and ready the data accordingly for the model development. 
# A good data preprocessing is prominent for attaining better results after the model predictions. The results are most of the time proportional to the quality of data.
# Therefore, as giving more ordered, interpretable, and knowledgeful data to the algorithm, it is easier for model to process and learn the data more efficiently, and may 
# improve its ability of predicting more reliable results. To sum up, it is extremely important that we preprocess our data before feeding it into our model.

# 5) First of all, in order to understand whether the data set is continues or discrete, I would try to plot a histogram. If the boxes are sperate and 
# there are spaces between them then, this indicates the presenc of discrete variables. Unlike discrete data, the continues data histogram should plot clustered bars
# where there is no space between the boxes at all, it means we are dealing with continues variables. The better way of representing this is ploting a trend line like line
# which clearly dictates whether the data includes discrete or continues variables. This line will be calculated based on the average value of the points from left to right
# on the histogram. If the line has cuts, this exhibits the values are discrete because values has a particular range between the sample points. Furthermore, a simplest way
# to describe the difference between the two is to visualize a simple plot. If plot becomes like a scatter graph it is discrete, if plot turnes into a line graph then it is 
# continous variable.

# 6) First of all, since there are two peaks, this is a bimodal distribution. The trend of data distribution can be understood easier by looking at the blue line. I believe
# that there are two hints about the variable type of the data. First, the x and y axis indicate that the values are numeric since they are float meaning each individual sample 
# takes a range of values. Secondly, the blue line as well as the boxes reveal that there are not dotted lines or deperate boxes so the data are not discrete but it is continous.
# Since, the EDA step has been conduted already, we can directly skip to the pre processing step. After understanding the data, identfying variable types, and distribution types,
# I would search for duplicates, missing values and outliers. Especially, the first two are the simplest to overcome using DF[DF.duplicated(keep=False)] and DF.isnull().sum(). 
# The missing values can be replaced by the median values since data distribution is not standard. For the outlier detection, we may try to limit the max x-y values via xlim, ylim
# and extract those data points from the main data set to identify the outlier candidates. We can compare them with the remaining data points standart deviations and try to find out
# which one to remove and which one to keep. This part can be carried out by just removing some part by scientific guess and train and test the model to see the presence or 
# absence of the discarded sample points affect the model performance. Or we could utilize the isolation forest alogirthm from scikit learn and ensemble class. I think feature
# scaling may not be extremely necessary for this case but normalization can be applied since data distribution is not gaussian so that we may have better structed data. I think
# it is not a must to apply data binning because it looks like it's already binned so I think there is not critical number of noisy data that capable of affecting our model.
# Similarly, I would not perform feature encoding because values are already float so do not need to convert them into another data representation. Besides, if I performed normalization,
# it means our values would be assinged into new values between 0 and 1. However, feature extraction might be good to go so that we can eleminate unnecesarry features.
# After that I would split the data set into two for training and test set but I would try different percentage of random splits such as 70-30, 75-30, 80-20 to see which model 
# predicts better. I probably does not divide data set into three for validation this is because data is not complex and does not require to many tunning. The trial and error for
# parameter search would be less time consuming.
