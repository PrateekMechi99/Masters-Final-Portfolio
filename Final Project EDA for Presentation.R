
wine <- read.csv("C:\\Users\\Prateek\\Documents\\Masters Homework Folder\\IST 707\\Final Project\\winequality-red.csv")
whitewine <-read.csv("C:\\Users\\Prateek\\Documents\\Masters Homework Folder\\IST 707\\Final Project\\winequality-white.csv")
summary(wine)
install.packages("GGally")
library("knitr")
library("corrgram")
library('ggplot')
install.packages("FactoMineR")
install.packages("factoextra")
install.packages("class")
library(factoextra)
library (clValid)
library (caret)
library(dplyr)
library(ggplot2)
library(MASS)
library(plotly)
library(corrplot)
library(gridExtra)
library(FactoMineR)
library(factoextra)
library(corrgram)
library(tidyr)
library(nnet)
library(party)
library(rpart)
library(rpart.plot)
library(randomForest)
library(e1071)
library(class)
##Corrgram
corrgram(wine, order = TRUE , lower.panel=panel.conf)

set.seed(12)
library(dplyr)
library(tidyr)

#wine factor correlation
Redwinecorr <- cor(wine) %>%  as.data.frame() %>%  mutate(var1 = rownames(.)) %>%  gather(var2, value, -var1) %>%  arrange(desc(value)) %>%  group_by(value) %>%  filter(row_number()==1)

Redwinecorr

ggpairs(wine, title="Wine Features Correlation with ggpairs") 

#factoringquality
wine$quality=as.factor(wine$quality)
#distribution of red wine quality
ggplot(wine, aes(x=quality, fill = quality)) +
  geom_bar(stat="count") +
  geom_text(position = "stack", stat='count',aes(label=..count..), vjust = -0.5)+
  labs(y="Num of Observations", x="Wine Quality") +
  labs(title= "Distribution of Red Wine Quality Ratings")



wine%>%
  gather(-quality, key = "var", value = "value") %>% 
  ggplot(aes(x = quality, y = value, color = quality)) +
  geom_boxplot() +
  facet_wrap(~ var, scales = "free", ncol = 3)+
  theme(legend.position="none")



#Decision Tree1 , test and train 1
set.seed(1)
index <- createDataPartition(wine$quality, p=0.8, list=FALSE) 
train <- wine[index,]
test <- wine[-index,]
model <- rpart(quality~., data=train)
prp(model, type=2, extra=1)
model$cptable

prediction <- predict(model, test, type="class")
confusionMatrix(prediction, test$quality)


#Decision Tree Part 2, scaled data, split into 3 groups
wine2 = scale(wine [,1:11])
wine_scaled= cbind.data.frame(wine2 , wine$quality)
colnames(wine_scaled) = colnames(wine)
head(wine_scaled)

wine_scaled$quality=as.integer(as.character( wine_scaled$quality))

wine_scaled$quality = lapply(wine_scaled[,12], function (x)
{
  if(x >6)  { "GOOD"}
  else if(x >4)  {"Meduim"}
  else { "Bad"}   
})
wine_scaled$quality = unlist(wine_scaled$quality)
wine_scaled$quality = as.factor(wine_scaled$quality)
table(wine_scaled$quality)


#test and train part2
set.seed(45)
x<-sample(c(1:nrow(wine_scaled)) , 0.8*nrow(wine_scaled))

train_data=wine_scaled[x,]
test_data=wine_scaled[-x,]

table(train_data$quality)
table(test_data$quality)

rpTree <- rpart(quality ~., data=train_data )
rpTree$cptable


rpTree <- rpart(quality~.,data=train_data , cp=0.01 , control = rpart.control((minsplit = 11)))
rpart.plot(rpTree, box.palette = "RdBu", shadow.col = "gray" ,nn=TRUE  )

prediction2 <- predict(rpTree, test_data, type = "class")
confusionMatrix(prediction2, test_data$quality)



##Random Forest 

rfModel <- randomForest(quality ~., data=train_data,
                        ntree=500,  # Use 500 trees in the forest
                        mtry=2,     # Consider 3 variables at each split
                        minsplit=10, # Require at least 10 samples at a node before splitting
                        minbucket=5  # Require at least 5 samples at a leaf node
)

predictions3 <- predict(rfModel, test_data)

confusionMatrix(predictions3, test_data$quality)

varImpPlot(rfModel)



##SVM

svmModel <- svm(quality ~., data=train_data,
                kernel="radial",  # Use a radial kernel
                degree=3,             # Use a degree of 2 for the radial
                cost=10,              # Use a cost of 10 for the SVM
                epsilon=0.1           # Use a tolerance of 0.1 for the SVM
)

svmModel2 <- svm(quality ~., data=train_data,
                kernel="polynomial",  # Use a polynomial kernel
                degree=3,             # Use a degree of 3 for the polynomial
                cost=10,              # Use a cost of 10 for the SVM
                epsilon=0.1           # Use a tolerance of 0.1 for the SVM
)

svmModel3 <- svm(quality ~., data=train_data,
                kernel="linear",  # Use a linear kernel
                degree=3,             # Use a degree of 3 for linear
                cost=10,              # Use a cost of 10 for the SVM
                epsilon=0.1           # Use a tolerance of 0.1 for the SVM
)



# Use the trained model to make predictions on the test data
predictionsradial <- predict(svmModel, test_data)
predictionspoly <- predict(svmModel2, test_data)
predictionslinear <- predict(svmModel3, test_data)


# Evaluate the model's performance on the test data
confusionMatrix(predictionsradial, test_data$quality)
confusionMatrix(predictionspoly, test_data$quality)
confusionMatrix(predictionslinear, test_data$quality)






##Naive Bayes
# Train the Naive Bayes classifier with modified parameters
nbModel <- naiveBayes(quality ~., data=train_data,
                      laplace=1,          # Use Laplace smoothing with a value of 2
                      usekernel=FALSE,    # Use a kernel density estimate for continuous variables
                      adjust=3          # Use a bandwidth adjustment factor of 1
)

nbModel2 <- naiveBayes(quality ~., data=train_data)



# Use the trained model to make predictions on the test data
predictionsnb1 <- predict(nbModel, test_data)
predictionsnb2 <- predict(nbModel2, test_data)

# Evaluate the model's performance on the test data
confusionMatrix(predictionsnb1, test_data$quality)

confusionMatrix(predictionsnb2, test_data$quality)

##KNN-not needed at the moment

# Train the KNN classifier with modified parameters
k_values <- data.frame(k=c(3, 5, 7))
ctrl <- trainControl(method="repeatedcv", number=3, verboseIter=TRUE)
knnfit <- train(quality~., data = train_data, method = "knn", trControl = ctrl, tuneGrid = k_values)
# Use the trained model to make predictions on the test data
knnpred <- predict(knnfit, newdata = test_data)

# Evaluate the model's performance on the test data
confusionMatrix(data = as.factor(knnpred), as.factor(test_data$quality))

plot(knnfit)
