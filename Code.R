#####DATABASE_PACKAGES__IMPORTATION#####
remove(list = ls())
Train <- read.csv("Train.csv", header=TRUE, dec=".", sep=",")
library(visdat)
library(dplyr)
library(corrplot)
library(FactoMineR)
library(factoextra)
library(ggplot2) 
library("nnet")
library("caret") 
library(e1071) 
library(naivebayes)
library(klaR)
library(rpart)
library(rpart.plot) 
library("rcompanion")
library(MASS)
library(RcmdrMisc)
library(ipred)
library("randomForest")
library("missMDA")
library(caTools)
library("DMwR") 
library("missForest")
library("mltools")
library("data.table")
library("plyr")
library("devtools")
library(kernlab)


#####EXPLORATORY_UNIVARIATE_ANALYSIS#####
#Train_Initial : 7131 observations and 14 variables 
class(Train)
summary(Train)
str(Train)
vis_miss(Train)

#Plot and analysis of each variable
table(Train$Work_Experience, useNA = "ifany")
plot(Train$Work_Experience)
hist(Train$Work_Experience)
boxplot(Train$Work_Experience)
mean(Train$Work_Experience, na.rm = TRUE) #2.617757
median(Train$Work_Experience, na.rm = TRUE) #1

table(Train$Family_Size, useNA = "ifany")
plot(Train$Family_Size)
mean(Train$Family_Size, na.rm = TRUE) #2.868864
median(Train$Family_Size, na.rm = TRUE) #3

table(Train$Child, useNA = "ifany")
plot(Train$Child)
mean(Train$Child, na.rm = TRUE) #1.208129
median(Train$Child, na.rm = TRUE) #1

table(Train$Graduated, useNA="ifany")

table(Train$Ever_Married, useNA='ifany')

table(Train$Profession, useNA='ifany')

table(Train$Var_1, useNA='ifany')

table(Train$Credit_Owner)

table(Train$Spending_Score)

#Conclusion of univariate exploration:
Train$Licence_Plate <- NULL
Train$Child <- NULL
Train$Credit_Owner <- factor(Train$Credit_Owner, labels = c("no","yes"))
Train$Spending_Score<- ordered(Train$Spending_Score, levels=c("Low", "Average", "High")) 


#### HANDLING NA ####
#MCA-PCA imputation
nbcomp <- estim_ncpFAMD(Train, method.cv = "Kfold", nbsim=10) #estimation du nombre de composantes principales à utiliser dans le PCA, -> 5
imputation <- imputeFAMD(Train, ncp=5, scale=TRUE) #remplacer les variables NA
Train <- imputation$completeObs


####EXPLORATORY_BIVARIATE_ANALYSIS####
###NUMERICAL VARIABLES
#Correlation
Train_corr <- Train[, c(3,6,8,9)] 
matrice_cor <- cor(Train_corr) 
matrice_cor 
corrplot(matrice_cor,method="number",type="upper") 
corrplot(matrice_cor,method = "circle", type="upper") 

#PCA 
res <- PCA(Train, graph = FALSE, quali.sup = c(1,2,4,5,7,11,12,10)) 
summary(res)
corrplot(res$var$cos2, method="circle")
plot.PCA(res, choix = "var", axes=c(1,2)) 

#ANOVA test
aov1 <- aov(Car~Segmentation, data=Train)
summary(aov1)
TukeyHSD(aov1) #Pas de diff significative entre C et B
aov2 <- aov(Age~Segmentation, data=Train)
summary(aov2)
TukeyHSD(aov2) 
aov3 <- aov(Work_Experience~Segmentation, data=Train)
summary(aov3)
TukeyHSD(aov3) # Pas de diff significative (p-valeur enorme) entre D et A et C et B
aov4 <- aov(Family_Size~Segmentation, data=Train)
summary(aov4)
TukeyHSD(aov4) #Pas de diff significative entre B et A

###CATEGORICAL VARIABLES
#Visual representation of relations between variables
plot(Train$Segmentation, Train$Age, main="Impact of Age on the Segmentation", xlab="Segmentation", ylab="Age")
plot(Train$Segmentation, Train$Car, main="Impact of Car on the Segmentation", xlab="Segmentation", ylab="Car")
plot(Train$Segmentation, Train$Gender, main="Impact of Gender on the Segmentation", xlab="Segmentation", ylab="Gender")
plot(Train$Segmentation, Train$Ever_Married, main="Impact of Ever_Married on the Segmentaion", xlab="Segmentation", ylab="Ever_Married")
plot(Train$Segmentation, Train$Graduated, main="Impact of Graduated on the Segmentaion", xlab="Segmentation", ylab="Graduated")
plot(Train$Segmentation, Train$Profession, main="Impact of Profession on the Segmentaion", xlab="Segmentation", ylab="Profession")
plot(Train$Segmentation, Train$Work_Experience, main="Impact of Work_Experience on the Segmentaion", xlab="Segmentation", ylab="Work_Experience")
plot(Train$Segmentation, Train$Spending_Score, main="Impact of Spending_Score on the Segmentaion", xlab="Segmentation", ylab="Spending_Score")
plot(Train$Segmentation, Train$Credit_Owner, main="Impact of Credit_Owner on the Segmentaion", xlab="Segmentation", ylab="Credit_Owner")
plot(Train$Segmentation, Train$Var_1, main="Impact of Var_1 on the Segmentaion", xlab="Segmentation", ylab="Var_1")

#Contigency tables with the dependent variable
table(Train$Gender, Train$Segmentation) 
table(Train$Ever_Married, Train$Segmentation) 
table(Train$Age, Train$Segmentation)
table(Train$Graduated, Train$Segmentation)
table(Train$Profession, Train$Segmentation)
table(Train$Work_Experience, Train$Segmentation)
table(Train$Spending_Score, Train$Segmentation)
table(Train$Family_Size, Train$Segmentation)
table(Train$Car, Train$Segmentation)
table(Train$Credit_Owner, Train$Segmentation)
table(Train$Var_1, Train$Segmentation)

#CRAMER'S V between features
cramerV(Train$Ever_Married, Train$Graduated) #0.1972
cramerV(Train$Ever_Married, Train$Gender) #0.1143
cramerV(Train$Ever_Married, Train$Profession) #0.5214
cramerV(Train$Ever_Married, Train$Spending_Score) #0.6763
cramerV(Train$Ever_Married, Train$Credit_Owner) #0.2215
cramerV(Train$Ever_Married, Train$Var_1) #0.1244
cramerV(Train$Graduated, Train$Gender) #0.02621
cramerV(Train$Graduated, Train$Profession) #0.4229
cramerV(Train$Graduated, Train$Spending_Score) #0.1414
cramerV(Train$Graduated, Train$Credit_Owner) #0.07286
cramerV(Train$Graduated, Train$Var_1) #0.2376
cramerV(Train$Gender, Train$Profession) #0.3518
cramerV(Train$Gender, Train$Spending_Score) #0.05525
cramerV(Train$Gender, Train$Credit_Owner) #0.08355
cramerV(Train$Gender, Train$Var_1) #0.05567
cramerV(Train$Profession, Train$Spending_Score) #0.4458
cramerV(Train$Profession, Train$Credit_Owner) #0.2682
cramerV(Train$Profession, Train$Var_1) #0.1198
cramerV(Train$Spending_Score, Train$Credit_Owner) #0.446
cramerV(Train$Spending_Score, Train$Var_1) #0.08251
cramerV(Train$Credit_Owner, Train$Var_1) #0.05904

#Pearson's Chi-square test
chisq.test(Train$Segmentation, Train$Gender) #Indépendant
chisq<-chisq.test(Train$Segmentation, Train$Credit_Owner) #Dépendant
chisq.test(Train$Segmentation, Train$Var_1) #Dépendant
chisq.test(Train$Segmentation, Train$Ever_Married) #dépendant
chisq.test(Train$Segmentation, Train$Graduated) #dépendant
chisq.test(Train$Segmentation, Train$Profession) #dépendant
chisq.test(Train$Segmentation, Train$Spending_Score) #Dépendant

#### FEATURE_SELECTION ####
Train2<- Train[-9] #without car
Train3<- Train[-c(1,9)] #without car and gender


##### FEATURE_EXTRACTION ####
###PCA 
##Creating database with only numerical variables and perform PCA
Train_num <- Train[,c(3,6,8,9)]
pc <- princomp(Train_num, cor=TRUE, score=TRUE)
summary(pc) 
pc2 <-princomp(Train_num[,-4], cor=TRUE, score=TRUE)

##Creating database with PCA components
#Two dimensions kept
PCA_2dim <- pc$scores[,c(1,2)]
PCA_2dim_Data <- cbind(PCA_2dim, Train[, c(1,2,4,5,7,10,11,12)])
#Three dimensions kept
PCA_3dim <- pc$scores[,c(1,2,3)]
PCA_3dim_Data <- cbind(PCA_3dim, Train[, c(1,2,4,5,7,10,11,12)])

###MCA 
##Creating database with only categorical variables and perform MCA
Train_factor <- Train[,-c(3,6,8,9,12)]
res_MCA <- MCA(Train_factor, ncp=20)
#Elbow criterion
get_eigenvalue(res_MCA)
mean(res_MCA$eig[,1])
barplot(res_MCA$eig[, 2], main= "Histogramme des valeurs propres", names.arg=rownames(res_MCA$eig), 
        xlab= "Axes", ylab= "Pourcentage d'inertie", cex.axis=0.8, font.lab=3, col= "orange")
#Kaiser criterion
fviz_screeplot(res_MCA, addlabels = TRUE, ylim = c(0, 45))

##Creating database with MCA dimensions
#13dimensions kept according to 80% criterion
MCA_13dim <- res_MCA$ind$coord[,c(1:13)] 
MCA_13dim_Data <- cbind(MCA_13dim, Train[,c(3,6,8,9,12)])
names(MCA_13dim_Data) <- c("dim1","dim2","dim3","dim4","dim5","dim6","dim7","dim8","dim9","dim10","dim11","dim12","dim13","Age","Work_Experience","Family_Size","Car","Segmentation")
#6dimensions kept according to the elbow criterion
MCA_6dim <- res_MCA$ind$coord[,c(1:6)] 
MCA_6dim_Data <- cbind(MCA_6dim, Train[,c(3,6,8,9,12)])
names(MCA_6dim_Data) <- c("dim1","dim2","dim3","dim4","dim5","dim6","Age","Work_Experience","Family_Size","Car","Segmentation")
#9dimensions kept according to the Kaiser criterion
MCA_9dim<- res_MCA$ind$coord[,c(1:9)] 
MCA_9dim_Data <- cbind(MCA_9dim, Train[,c(3,6,8,9,12)])
names(MCA_9dim_Data) <- c("dim1","dim2","dim3","dim4","dim5","dim6","dim7","dim8","dim9","Age","Work_Experience","Family_Size","Car","Segmentation")

##Creating combined databases of PCA and MCA
#Combination of PCA 2dimensions and MCA 6dimensions
COMBI_2_6 <- cbind(PCA_2dim, MCA_6dim, Train$Segmentation)
COMBI_2_6 <- as.data.frame(COMBI_2_6)
names(COMBI_2_6) <- c("Comp.1","Comp.2","dim1","dim2","dim3","dim4","dim5","dim6","Segmentation")
COMBI_2_6$Segmentation <- as.factor(COMBI_2_6$Segmentation)
COMBI_2_6$Segmentation <- revalue(COMBI_2_6$Segmentation, c("1"="A", "2"="B", "3"="C", "4"="D"))
#Combination of PCA 2dimensions and MCA 13dimensions
COMBI_2_13 <- cbind(PCA_2dim, MCA_13dim, Train$Segmentation)
COMBI_2_13 <- as.data.frame(COMBI_2_13)
names(COMBI_2_13) <- c("Comp.1","Comp.2","dim1","dim2","dim3","dim4","dim5","dim6","dim7","dim8","dim9","dim10","dim11","dim12","dim13","Segmentation")
COMBI_2_13$Segmentation <- as.factor(COMBI_2_13$Segmentation)
COMBI_2_13$Segmentation <- revalue(COMBI_2_13$Segmentation, c("1"="A", "2"="B", "3"="C", "4"="D"))

###Conclusion of feature extraction
TrainPCA <- PCA_2dim_Data


#### CROSS_VALIDATION_K_FOLD ####
#Randomly shuffle the data with same seed
set.seed(123)
Train_cross<- Train[sample(nrow(Train2)),]
set.seed(123)
Train_cross2<-Train2[sample(nrow(Train2)),]
set.seed(123)
Train_cross3<-Train3[sample(nrow(Train3)),]
set.seed(123)
Train_crossPCA<- TrainPCA[sample(nrow(TrainPCA)),]
#Create 10 equally sized folds
folds <- cut(seq(1,nrow(Train_cross)),breaks=10,labels=FALSE)
folds2 <- cut(seq(1,nrow(Train_cross2)),breaks=10,labels=FALSE)
folds3 <- cut(seq(1,nrow(Train_cross3)),breaks=10,labels=FALSE)
foldsPCA <- cut(seq(1,nrow(Train_crossPCA)),breaks=10,labels=FALSE)


#### MODEL 1 - LOGISTIC REGRESSION ####
#Normal logistic regression on Train
accuracy_reg.m1 <- c()
for(k in 1:10){
  testIndexes <- which(folds==k, arr.ind=TRUE)
  testData <- Train_cross[testIndexes, ]
  trainData <- Train_cross[-testIndexes,]
  reg.m1 <- multinom(Segmentation ~ ., data=trainData)
  predict_reg.m1 <- predict(reg.m1, testData, type = 'class') 
  confusion_reg.m1 <- table(testData$Segmentation, predict_reg.m1) 
  accuracy_reg.m1[k] <- sum(diag(confusion_reg.m1)) / sum(confusion_reg.m1) }
accuracy_reg.m1
mean(accuracy_reg.m1)
sd(accuracy_reg.m1)

#Stepwise logistic regression on Train
accuracy_reg.m2 <- c(0)
for(k in 1:10){
  testIndexes <- which(folds==k, arr.ind=TRUE)
  testData <- Train_cross[testIndexes, ]
  trainData <- Train_cross[-testIndexes,]
  reg.m1 <- multinom(Segmentation ~ ., data=trainData)
  reg.m2 <- stepwise(reg.m1, direction = "backward/forward", criterion = "AIC")
  predict_reg.m2 <- predict(reg.m2, testData, type = 'class') 
  confusion_reg.m2 <- table(testData$Segmentation, predict_reg.m2) 
  accuracy_reg.m2[k] <- sum(diag(confusion_reg.m2)) / sum(confusion_reg.m2)}
accuracy_reg.m2
mean(accuracy_reg.m2)
sd(accuracy_reg.m2)

#Normal logistic regression on Train2
accuracy_reg.m3 <- c(0)
for(k in 1:10){
  testIndexes <- which(folds2==k, arr.ind=TRUE)
  testData <- Train_cross2[testIndexes, ]
  trainData <- Train_cross2[-testIndexes,]
  reg.m3 <- multinom(Segmentation ~ ., data=trainData)
  predict_reg.m3 <- predict(reg.m3, testData, type = 'class') 
  confusion_reg.m3 <- table(testData$Segmentation, predict_reg.m3) 
  accuracy_reg.m3[k] <- sum(diag(confusion_reg.m3)) / sum(confusion_reg.m3) 
}
accuracy_reg.m3
mean(accuracy_reg.m3)
sd(accuracy_reg.m3)

#Stepwise logistic regression on Train2
accuracy_reg.m4 <- c(0)
for(k in 1:10){
  testIndexes <- which(folds2==k, arr.ind=TRUE)
  testData <- Train_cross2[testIndexes, ]
  trainData <- Train_cross2[-testIndexes,]
  reg.m3 <- multinom(Segmentation ~ ., data=trainData)
  reg.m4 <- stepwise(reg.m3, direction = "backward/forward", criterion = "AIC")
  predict_reg.m4 <- predict(reg.m4, testData, type = 'class') 
  confusion_reg.m4 <- table(testData$Segmentation, predict_reg.m4) 
  accuracy_reg.m4[k] <- sum(diag(confusion_reg.m4)) / sum(confusion_reg.m4) 
}
accuracy_reg.m4
mean(accuracy_reg.m4)
sd(accuracy_reg.m4)

#Normal logistic regression on Train3
accuracy_reg.m5 <- c(0)
for(k in 1:10){
  testIndexes <- which(folds3==k, arr.ind=TRUE)
  testData <- Train_cross3[testIndexes, ]
  trainData <- Train_cross3[-testIndexes,]
  reg.m5 <- multinom(Segmentation ~ ., data=trainData)
  predict_reg.m5 <- predict(reg.m5, testData, type = 'class') 
  confusion_reg.m5 <- table(testData$Segmentation, predict_reg.m5) 
  accuracy_reg.m5[k] <- sum(diag(confusion_reg.m5)) / sum(confusion_reg.m5) 
}
accuracy_reg.m5
mean(accuracy_reg.m5)
sd(accuracy_reg.m5)

#Stepwise logistic regression on Train3
accuracy_reg.m6 <- c(0)
for(k in 1:10){
  testIndexes <- which(folds3==k, arr.ind=TRUE)
  testData <- Train_cross3[testIndexes, ]
  trainData <- Train_cross3[-testIndexes,]
  reg.m5 <- multinom(Segmentation ~ ., data=trainData)
  reg.m6 <- stepwise(reg.m5, direction = "backward/forward", criterion = "AIC")
  predict_reg.m6 <- predict(reg.m6, testData, type = 'class') 
  confusion_reg.m6 <- table(testData$Segmentation, predict_reg.m6) 
  accuracy_reg.m6[k] <- sum(diag(confusion_reg.m6)) / sum(confusion_reg.m6) }
accuracy_reg.m6
mean(accuracy_reg.m6)
sd(accuracy_reg.m6)

#Normal logistic regression on TrainPCA
accuracy_reg.m7 <- c(0)
for(k in 1:10){
  testIndexes <- which(foldsPCA==k, arr.ind=TRUE)
  testData <- Train_crossPCA[testIndexes, ]
  trainData <- Train_crossPCA[-testIndexes,]
  reg.m7 <- multinom(Segmentation ~ ., data=trainData)
  predict_reg.m7 <- predict(reg.m7, testData, type = 'class') 
  confusion_reg.m7 <- table(testData$Segmentation, predict_reg.m7) 
  accuracy_reg.m7[k] <- sum(diag(confusion_reg.m7)) / sum(confusion_reg.m7) }
accuracy_reg.m7
mean(accuracy_reg.m7)
sd(accuracy_reg.m7)

#Stepwise logistic regression on Train3
accuracy_reg.m8 <- c(0)
for(k in 1:10){
  testIndexes <- which(foldsPCA==k, arr.ind=TRUE)
  testData <- Train_crossPCA[testIndexes, ]
  trainData <- Train_crossPCA[-testIndexes,]
  reg.m7 <- multinom(Segmentation ~ ., data=trainData)
  reg.m8 <- stepwise(reg.m7, direction = "backward/forward", criterion = "AIC")
  predict_reg.m8 <- predict(reg.m8, testData, type = 'class') 
  confusion_reg.m8 <- table(testData$Segmentation, predict_reg.m8) 
  accuracy_reg.m8[k] <- sum(diag(confusion_reg.m8)) / sum(confusion_reg.m8) }
accuracy_reg.m8
mean(accuracy_reg.m8)
sd(accuracy_reg.m8)


#### MODEL 2 - NAIVE BAYES ####
#Train
accuracy_nb.m1 <- c(0)
for(k in 1:10){
  testIndexes <- which(folds==k, arr.ind=TRUE)
  testData <- Train_cross[testIndexes, ]
  trainData <- Train_cross[-testIndexes,]
  nb.m1 <- naive_bayes(formula =Segmentation ~ ., data=trainData, usekernel = TRUE)
  predict_nb.m1 <- predict(object=nb.m1, newdata=testData, type = "class") 
  confusion_nb.m1 <- table(testData$Segmentation, predict_nb.m1) 
  accuracy_nb.m1[k] <- sum(diag(confusion_nb.m1)) / sum(confusion_nb.m1) }
accuracy_nb.m1
mean(accuracy_nb.m1)
sd(accuracy_nb.m1)

#Train2
accuracy_nb.m2 <- c(0)
for(k in 1:10){
  testIndexes <- which(folds2==k, arr.ind=TRUE)
  testData <- Train_cross2[testIndexes, ]
  trainData <- Train_cross2[-testIndexes,]
  nb.m2 <- naive_bayes(formula =Segmentation ~ ., data=trainData)
  predict_nb.m2 <- predict(nb.m2, testData[,-9], type = 'class') 
  confusion_nb.m2 <- table(testData$Segmentation, predict_nb.m2) 
  accuracy_nb.m2[k] <- sum(diag(confusion_nb.m2)) / sum(confusion_nb.m2) }
accuracy_nb.m2
mean(accuracy_nb.m2)
sd(accuracy_nb.m2)

#Train3
accuracy_nb.m3 <- c(0)
for(k in 1:10){
  testIndexes <- which(folds3==k, arr.ind=TRUE)
  testData <- Train_cross3[testIndexes, ]
  trainData <- Train_cross3[-testIndexes,]
  nb.m3 <- naive_bayes(formula =Segmentation ~ ., data=trainData)
  predict_nb.m3 <- predict(nb.m3, testData[,-9], type = 'class') 
  confusion_nb.m3 <- table(testData$Segmentation, predict_nb.m3) 
  accuracy_nb.m3[k] <- sum(diag(confusion_nb.m3)) / sum(confusion_nb.m3) }
accuracy_nb.m3
mean(accuracy_nb.m3)
sd(accuracy_nb.m3)

#TrainPCA
accuracy_nb.m4 <- c(0)
for(k in 1:10){
  testIndexes <- which(foldsPCA==k, arr.ind=TRUE)
  testData <- Train_crossPCA[testIndexes, ]
  trainData <- Train_crossPCA[-testIndexes,]
  nb.m4 <- naive_bayes(formula =Segmentation ~ ., data=trainData)
  predict_nb.m4 <- predict(nb.m4, testData[,-9], type = 'class') 
  confusion_nb.m4 <- table(testData$Segmentation, predict_nb.m4) 
  accuracy_nb.m4[k] <- sum(diag(confusion_nb.m4)) / sum(confusion_nb.m4) }
accuracy_nb.m4
mean(accuracy_nb.m4)
sd(accuracy_nb.m4)


#### MODEL 3 - RANDOM FOREST ####
#Train
set.seed(123)
accuracy_rf.m1 <- c(0)
for(k in 1:10){
  testIndexes <- which(folds==k, arr.ind=TRUE)
  testData <- Train_cross[testIndexes, ]
  trainData <- Train_cross[-testIndexes,]
  rf.m1<-randomForest(Segmentation ~ ., data = trainData)
  predict_rf.m1 <- predict(rf.m1, testData, type = 'class') 
  confusion_rf.m1 <- table(testData$Segmentation, predict_rf.m1) 
  confusion_total_rfOther <- confusion_total_rfOther + confusion_rf.m1
  accuracy_rf.m1[k] <- sum(diag(confusion_rf.m1)) / sum(confusion_rf.m1) }
accuracy_rf.m1
mean(accuracy_rf.m1)
sd(accuracy_rf.m1)

#Train2 
set.seed(123)
accuracy_rf.m2 <- c(0)
for(k in 1:10){
  testIndexes <- which(folds2==k, arr.ind=TRUE)
  testData <- Train_cross2[testIndexes, ]
  trainData <- Train_cross2[-testIndexes,]
  rf.m2<-randomForest(Segmentation ~ ., data = trainData)
  predict_rf.m2 <- predict(rf.m2, testData, type = 'class') 
  confusion_rf.m2 <- table(testData$Segmentation, predict_rf.m2) 
  accuracy_rf.m2[k] <- sum(diag(confusion_rf.m2)) / sum(confusion_rf.m2) }
accuracy_rf.m2
mean(accuracy_rf.m2)
sd(accuracy_rf.m2)

#Train3
set.seed(123)
accuracy_rf.m3 <- c(0)
for(k in 1:10){
  testIndexes <- which(folds3==k, arr.ind=TRUE)
  testData <- Train_cross3[testIndexes, ]
  trainData <- Train_cross3[-testIndexes,]
  rf.m3<-randomForest(Segmentation ~ ., data = trainData)
  predict_rf.m3 <- predict(rf.m3, testData, type = 'class') 
  confusion_rf.m3 <- table(testData$Segmentation, predict_rf.m3) 
  accuracy_rf.m3[k] <- sum(diag(confusion_rf.m3)) / sum(confusion_rf.m3) }
accuracy_rf.m3
mean(accuracy_rf.m3)
sd(accuracy_rf.m3)

#TrainPCA
set.seed(123)
accuracy_rf.m4 <- c(0)
for(k in 1:10){
  testIndexes <- which(foldsPCA==k, arr.ind=TRUE)
  testData <- Train_crossPCA[testIndexes, ]
  trainData <- Train_crossPCA[-testIndexes,]
  rf.m4<-randomForest(Segmentation ~ ., data = trainData)
  predict_rf.m4 <- predict(rf.m4, testData, type = 'class') 
  confusion_rf.m4 <- table(testData$Segmentation, predict_rf.m4) 
  accuracy_rf.m4[k] <- sum(diag(confusion_rf.m4)) / sum(confusion_rf.m4) }
accuracy_rf.m4
mean(accuracy_rf.m4)
sd(accuracy_rf.m4)


#### MODEL 4 - SVM ####
###KERNEL RADIAL
#Train
accuracy_svm.m1 <- c(0)
for(k in 1:10){
  testIndexes <- which(folds==k, arr.ind=TRUE)
  testData <- Train_cross[testIndexes, ]
  trainData <- Train_cross[-testIndexes,]
  svm.m1<- svm(Segmentation ~., data= trainData, type = "C-classification", probability=TRUE, kernel="radial", cost=10)
  predict_svm.m1 <- predict(svm.m1, testData, type = 'class') 
  confusion_svm.m1 <- table(testData$Segmentation, predict_svm.m1) 
  accuracy_svm.m1[k] <- sum(diag(confusion_svm.m1)) / sum(confusion_svm.m1) }
accuracy_svm.m1
mean(accuracy_svm.m1)
sd(accuracy_svm.m1)

#Train2
accuracy_svm.m2 <- c(0)
for(k in 1:10){
  testIndexes <- which(folds2==k, arr.ind=TRUE)
  testData <- Train_cross2[testIndexes, ]
  trainData <- Train_cross2[-testIndexes,]
  svm.m2<- svm(Segmentation ~., data= trainData, type = "C-classification", probability=TRUE, kernel="radial", cost=10)
  predict_svm.m2 <- predict(svm.m2, testData, type = 'class') 
  confusion_svm.m2 <- table(testData$Segmentation, predict_svm.m2) 
  accuracy_svm.m2[k] <- sum(diag(confusion_svm.m2)) / sum(confusion_svm.m2) }
accuracy_svm.m2
mean(accuracy_svm.m2)
sd(accuracy_svm.m2)

#Train3
accuracy_svm.m3 <- c(0)
for(k in 1:10){
  testIndexes <- which(folds3==k, arr.ind=TRUE)
  testData <- Train_cross3[testIndexes, ]
  trainData <- Train_cross3[-testIndexes,]
  svm.m3<- svm(Segmentation ~., data= trainData, type = "C-classification", probability=TRUE, kernel="radial", cost=10)
  predict_svm.m3 <- predict(svm.m3, testData, type = 'class') 
  confusion_svm.m3 <- table(testData$Segmentation, predict_svm.m3) 
  accuracy_svm.m3[k] <- sum(diag(confusion_svm.m3)) / sum(confusion_svm.m3) }
accuracy_svm.m3
mean(accuracy_svm.m3)
sd(accuracy_svm.m3)

#TrainPCA
accuracy_svm.m4 <- c(0)
for(k in 1:10){
  testIndexes <- which(foldsPCA==k, arr.ind=TRUE)
  testData <- Train_crossPCA[testIndexes, ]
  trainData <- Train_crossPCA[-testIndexes,]
  svm.m4<- svm(Segmentation ~., data= trainData, type = "C-classification", probability=TRUE, kernel="radial", cost=10)
  predict_svm.m4 <- predict(svm.m4, testData, type = 'class') 
  confusion_svm.m4 <- table(testData$Segmentation, predict_svm.4) 
  accuracy_svm.m4[k] <- sum(diag(confusion_svm.m4)) / sum(confusion_svm.m4) }
accuracy_svm.m4
mean(accuracy_svm.m4)
sd(accuracy_svm.m4)

###KERNEL POLYNOMIAL DEGREE = 2
#Train
accuracy_svm.m5 <- c(0)
for(k in 1:10){
  testIndexes <- which(folds==k, arr.ind=TRUE)
  testData <- Train_cross[testIndexes, ]
  trainData <- Train_cross[-testIndexes,]
  svm.m5<- svm(Segmentation ~., data= trainData, type = "C-classification", probability=TRUE, kernel="polynomial", degree=2, cost=100)
  predict_svm.m5 <- predict(svm.m5, testData, type = 'class') 
  confusion_svm.m5 <- table(testData$Segmentation, predict_svm.m5) 
  accuracy_svm.m5[k] <- sum(diag(confusion_svm.m5)) / sum(confusion_svm.m5) }
accuracy_svm.m5
mean(accuracy_svm.m5)
sd(accuracy_svm.m5)

#Train2
accuracy_svm.m6 <- c(0)
for(k in 1:10){
  testIndexes <- which(folds2==k, arr.ind=TRUE)
  testData <- Train_cross2[testIndexes, ]
  trainData <- Train_cross2[-testIndexes,]
  svm.m6<- svm(Segmentation ~., data= trainData, type = "C-classification", probability=TRUE, kernel="polynomial", degree=2, cost=100)
  predict_svm.m6 <- predict(svm.m6, testData, type = 'class') 
  confusion_svm.m6 <- table(testData$Segmentation, predict_svm.m6) 
  accuracy_svm.m6[k] <- sum(diag(confusion_svm.m6)) / sum(confusion_svm.m6) }
accuracy_svm.m6
mean(accuracy_svm.m6)
sd(accuracy_svm.m6)

#Train3
accuracy_svm.m7 <- c(0)
for(k in 1:10){
  testIndexes <- which(folds3==k, arr.ind=TRUE)
  testData <- Train_cross3[testIndexes, ]
  trainData <- Train_cross3[-testIndexes,]
  svm.m7<- svm(Segmentation ~., data= trainData, type = "C-classification", probability=TRUE, kernel="polynomial", degree=2, cost=100)
  predict_svm.m7 <- predict(svm.m7, testData, type = 'class') 
  confusion_svm.m7 <- table(testData$Segmentation, predict_svm.m7) 
  accuracy_svm.m7[k] <- sum(diag(confusion_svm.m7)) / sum(confusion_svm.m7) }
accuracy_svm.m7
mean(accuracy_svm.m7)
sd(accuracy_svm.m7)

#TrainPCA
accuracy_svm.m8 <- c(0)
for(k in 1:10){
  testIndexes <- which(foldsPCA==k, arr.ind=TRUE)
  testData <- Train_crossPCA[testIndexes, ]
  trainData <- Train_crossPCA[-testIndexes,]
  svm.m8<- svm(Segmentation ~., data= trainData, type = "C-classification", probability=TRUE, kernel="polynomial", degree=2, cost=100)
  predict_svm.m8 <- predict(svm.m8, testData, type = 'class') 
  confusion_svm.m8 <- table(testData$Segmentation, predict_svm.8) 
  accuracy_svm.m8[k] <- sum(diag(confusion_svm.m8)) / sum(confusion_svm.m8) }
accuracy_svm.m8
mean(accuracy_svm.m8)
sd(accuracy_svm.m8)


#### FOCUS_ON_CLASS_B #### 
#This is one of our multiple tests. This algorithm takes 1h to run.

#Cross validation in Fictive Test Set
set.seed(123)
Train_cross<-Train[sample(nrow(Train)),]
#Create 10 equally size folds
accuracy_Bmethod <- c()
for(k in 1:10){
  folds <- cut(seq(1,nrow(Train_cross)),breaks=10,labels=FALSE)
  testIndexes <- which(folds==k, arr.ind=TRUE)
  train_holdout =Train_cross[-testIndexes,] #creates the training dataset with row numbers stored in train_ind
  test_holdout =Train_cross[testIndexes,] 
  
  #Create database B vs other
  Segmentation <- combineLevels(Train_cross$Segmentation, c(1,3,4), "other")
  Train_B_h <- cbind(Train_cross[,-10], Segmentation)
  #Oversampling
  Train_B_over_h<- ovun.sample(Segmentation ~., data=Train_B_h, method="over", p=0.45, seed=123)$data
  Train_B_over_h$Segmentation <- relevel(Train_B_over_h$Segmentation, "B")
  #Model on B vs other database
  train_control1 <- trainControl(method="cv",number=10)
  svm.m1 <- train(Segmentation ~., data= Train_B_h, method="svmRadial", trControl=train_control1)
  predict_svm.m1 <- predict(svm.m1, test_holdout, type = 'raw')
  predictions.m1 <- data.frame(predict_svm.m1)
  predictions.m1 <- cbind(test_holdout,predictions.m1)
  #test set of the second model
  predictions.m1_other <- predictions.m1
  is.na(predictions.m1_other$predict_svm.m1) <- predictions.m1_other$predict_svm.m1 == "B"
  predictions.m1_other <- na.omit(predictions.m1_other)
  
  #Create database A C D
  Train_Other_h <- train_holdout
  is.na(Train_Other_h$Segmentation) <- Train_Other_h$Segmentation == "B"
  Train_Other_h <- na.omit(Train_Other_h)
  Train_Other_h$Segmentation <- factor(Train_Other_h$Segmentation) 
  #Model 2 on database A C D
  train_control <- trainControl(method="cv",number=10)
  svm.m2 <- train(Segmentation ~., data= Train_Other_h, method="svmRadial", trControl=train_control)
  predict_svm.m2 <- predict(svm.m2, predictions.m1_other[,-c(12,13)], type="raw")
  predictions.m2 <- data.frame(predict_svm.m2)
  
  #Get all predictions together
  predictions_all <-predictions.m1
  is.na(predictions_all$predict_svm.m1) <- predictions_all$predict_svm.m1 == "other"
  predictions_all$predict_svm.m1 <- factor(predictions_all$predict_svm.m1, levels = c("A","B","C","D"))
  predictions_all$predict_svm.m1[is.na(predictions_all$predict_svm.m1)] <- predictions.m2$predict_svm.m2
  confusion_total <- table(predictions_all$predict_svm.m1, predictions_all$Segmentation, dnn = c("observed", "predicted"))
  accuracy_Bmethod[k] <- sum(diag(confusion_total)) / sum(confusion_total)} 

#Accuracy of the total model (model B-other + model A-C-D)
accuracy_Bmethod
mean(accuracy_Bmethod)


#### ANALYSE_RECALL_PRECISION_F1_TOP3MODELS ####
#Model SVM Kernel Radial on Train 3 = svm.m3 
confusion_total <- confusion_svm.m3
confusion_total[] <- 0
accuracy_svm.m3 <- c(0)
for(k in 1:10){
  testIndexes <- which(folds3==k, arr.ind=TRUE)
  testData <- Train_cross3[testIndexes, ]
  trainData <- Train_cross3[-testIndexes,]
  svm.m3<- svm(Segmentation ~., data= trainData, type = "C-classification", probability=TRUE, kernel="radial", cost=10)
  predict_svm.m3 <- predict(svm.m3, testData, type = 'class') 
  confusion_svm.m3 <- table(testData$Segmentation, predict_svm.m3) 
  confusion_total <- confusion_total + confusion_svm.m3
  accuracy_svm.m3[k] <- sum(diag(confusion_svm.m3)) / sum(confusion_svm.m3) }
accuracy_svm.m3
mean(accuracy_svm.m3)
sd(accuracy_svm.m3)
confusion_total
precision <- diag(confusion_total)/apply(confusion_total,2,sum)
recall <- diag(confusion_total)/apply(confusion_total,1,sum)
f1 <- 2*precision*recall/(precision+recall)
data.frame(precision, recall, f1)
macroPrecision = mean(precision)
macroRecall = mean(recall)
macroF1 = mean(f1)
data.frame(mean(macroPrecision), mean(macroRecall), mean(macroF1))

#Model Logistic Regression on Train 2 = reg.m3
confusion_total <- confusion_reg.m3
confusion_total[] <- 0
accuracy_reg.m3 <- c(0)
for(k in 1:10){
  testIndexes <- which(folds2==k, arr.ind=TRUE)
  testData <- Train_cross2[testIndexes, ]
  trainData <- Train_cross2[-testIndexes,]
  reg.m3 <- multinom(Segmentation ~ ., data=trainData)
  predict_reg.m3 <- predict(reg.m3, testData, type = 'class') 
  confusion_reg.m3 <- table(testData$Segmentation, predict_reg.m3) 
  confusion_total <- confusion_total + confusion_reg.m3
  accuracy_reg.m3[k] <- sum(diag(confusion_reg.m3)) / sum(confusion_reg.m3) }
accuracy_reg.m3
mean(accuracy_reg.m3)
sd(accuracy_reg.m3)
confusion_total
precision <- diag(confusion_total)/apply(confusion_total,2,sum)
recall <- diag(confusion_total)/apply(confusion_total,1,sum)
f1 <- 2*precision*recall/(precision+recall)
data.frame(precision, recall, f1)
macroPrecision = mean(precision)
macroRecall = mean(recall)
macroF1 = mean(f1)
data.frame(mean(macroPrecision), mean(macroRecall), mean(macroF1))

#Model Random Forest with Train2 =  rf.m2
set.seed(123)
confusion_total <- confusion_rf.m2
confusion_total[] <- 0
accuracy_rf.m2 <- c(0)
for(k in 1:10){
  testIndexes <- which(folds2==k, arr.ind=TRUE)
  testData <- Train_cross2[testIndexes, ]
  trainData <- Train_cross2[-testIndexes,]
  rf.m2<-randomForest(Segmentation ~ ., data = trainData, ntree=300)
  predict_rf.m2 <- predict(rf.m2, testData, type = 'class') 
  confusion_rf.m2 <- table(testData$Segmentation, predict_rf.m2) 
  confusion_total <- confusion_total + confusion_rf.m2
  accuracy_rf.m2[k] <- sum(diag(confusion_rf.m2)) / sum(confusion_rf.m2) }
accuracy_rf.m2
mean(accuracy_rf.m2)
sd(accuracy_rf.m2)
confusion_total
precision <- diag(confusion_total)/apply(confusion_total,2,sum)
recall <- diag(confusion_total)/apply(confusion_total,1,sum)
f1 <- 2*precision*recall/(precision+recall)
data.frame(precision, recall, f1)
macroPrecision = mean(precision)
macroRecall = mean(recall)
macroF1 = mean(f1)
data.frame(mean(macroPrecision), mean(macroRecall), mean(macroF1))


#### GRAPH_FOR_FINAL_REPORT ####
corrplot(matrice_cor,method="number",type="upper") 

pca1<-plot.PCA(res, choix = "var", axes=c(1,2))
pca2<-plot.PCA(res, choix = "var", axes=c(2,3))
plot_grid(pca1, pca2, ncol = 2, nrow = 1)

ggplot(Train, aes(x=Segmentation, y=Age , color=Segmentation)) +
  geom_boxplot(outlier.size = 0.5) +
  ggtitle("Boxplot")+
  theme(plot.title = element_text(hjust = 0.5),  panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(), panel.background = element_blank()) +
  scale_color_brewer(palette="Dark2") 

barplot(res_MCA$eig[, 2], main= "Histogramme des valeurs propres", names.arg=rownames(res_MCA$eig), 
        xlab= "Axes", ylab= "Pourcentage d'inertie", cex.axis=0.8, font.lab=3, col= "orange")

fviz_screeplot(res_MCA, addlabels = TRUE, ylim = c(0, 45))


#### TESTSTUDENT ####
#Importation
TestStudent <- read.csv("TestStudent.csv", header=TRUE, dec=".", sep=",")

#Cleaning of database and removing Car and Gender to correspond to Train3
TestStudent$Licence_Plate <- NULL
TestStudent$Child <- NULL
TestStudent$Credit_Owner <- factor(TestStudent$Credit_Owner, labels = c("no","yes"))
TestStudent$Spending_Score<- ordered(TestStudent$Spending_Score, levels=c("Low", "Average", "High")) 
TestStudent$Car <- NULL
TestStudent$Gender <- NULL 

#Our final model
svm.final<- svm(Segmentation ~., data= Train3, type = "C-classification", probability=TRUE, kernel="radial", cost=10)
predict_TestStudent <- predict(svm.final, TestStudent, type = 'class')
predict_TestStudent <- as.matrix(predict_TestStudent)
summary(predict_TestStudent)

#Export predictions
write.table(x = predict_TestStudent, file="LLSMF2014-09-predictions.csv",  col.names=FALSE, row.names = FALSE)





