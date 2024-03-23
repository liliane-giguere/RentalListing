train = read.csv("W23P2_train.csv") ## interest level is col 1
test = read.csv("W23P2_test.csv") ## ID is col 1
train$interest_level = as.factor(train$interest_level)
### Load libraries
library(caret)
library(class)
library(MASS)
library(tidyverse)
library(ggplot2)
library(ranger)
library(xgboost)
library(tictoc)
library(doParallel)
registerDoParallel(cores=4)
library(randomForest)
library(kernlab)
library(xgboost)
library(e1071)
library(dplyr)
library(nnet)
library(purrr)
library(Ckmeans.1d.dp)


################## early data viz ####################
train %>%
  ggplot(aes(interest_level)) +
  geom_bar(fill = "blue") 

## there is one data point that is 1,150,000 for price -- skews chart so let's
## exclude it for now
which.max(train$price)
trainviz = train[-8081,]
trainviz = trainviz[!rowSums(trainviz[6]>10000),] ## exclude prices more than 10000 to see majority of datapoints
trainviz %>%
  ggplot(aes(x = price, y = bedrooms, color = interest_level))  +
  geom_point(size = 1)

## let's look at lat vs long
## datapoints with 0 lat and long
trainlat=train
trainlat = trainlat[trainlat$latitude!=0,]

ggplot(trainlat, aes(x=longitude, y=latitude, color = interest_level )) +
  geom_point() +
  scale_x_continuous(limits = c(-74.05, -73.8)) +
  scale_y_continuous(limits = c(40.6, 40.9)) 

ggplot(trainfinalless, aes(x=bedrooms, color = interest_level)) +
  geom_bar()

ggplot(trainfinalless, aes(x=bathrooms, color = interest_level)) +
  geom_bar()

## indicator variables -- are there any that are majority 0 or 1?
sort(colMeans(train[8:170]))

#### Data cleaning ##############################
#remove eat.in.kitchen
trainfinal = train
testfinal = test
trainfinal=dplyr::select(train, -Eat.In.Kitchen)
testfinal= dplyr::select(test, -Eat.In.Kitchen)

## remove datapoint where lat and long are 0, datapoint where price is million
trainfinal = trainfinal[c(-6648, -8081)]
## remove columns that have small means 
trainfinal = dplyr::select(trainfinal, -Cable.Satellite.TV, -Washer.Dryer.in.Unit,
                    -Guarantors.Accepted, -WiFi, -Concierge)
testfinal = dplyr::select(testfinal, -Cable.Satellite.TV, -Washer.Dryer.in.Unit,
                   -Guarantors.Accepted, -WiFi, -Concierge)

## are any indicator variables highly correlated with each other?
corr = cor(trainfinal[sapply(trainfinal, is.numeric)])
b = as.data.frame(apply(corr, 2, function(x) ifelse (abs(x) >=0.80, round(x,3), "-")))

trainfinal = dplyr::select(trainfinal, -private.balcony, -assigned.parking.space)
testfinal = dplyr::select(testfinal, -private.balcony, -assigned.parking.space)

############################### final data ##############
### export trainfinal -- without messy data so don't have to rerun everything
write.csv(trainfinal, "C:\\Users\\cream\\OneDrive - Queen's University\\Desktop\\STAT457\\Project 2\\train1.csv",
          row.names=TRUE)
write.csv(testfinal, "C:\\Users\\cream\\OneDrive - Queen's University\\Desktop\\STAT457\\Project 2\\test1.csv",
          row.names=TRUE)

trainfinal = read.csv("train1.csv") ## interest level is col 1
trainfinal$interest_level = as.factor(trainfinal$interest_level)
testfinal = read.csv("test1.csv") ## ID is col 1

newtrain <- trainfinal %>%
  mutate(street_name = gsub("^\\d+\\s", "", street_address)) %>%
  mutate(street_name = gsub("(?i)\\bst\\b", "Street", street_name)) %>%
  mutate(street_name = gsub("(?i)\\bave\\b", "Avenue", street_name)) %>%
  mutate(street_name = gsub("\\.$", "", street_name)) %>%
  mutate(street_number = ifelse(grepl("^\\d+", street_address), 
                                gsub("\\s.*", "", street_address), 0)) %>%
  mutate(is_west = ifelse(grepl("(?i)\\b(West|W)\\b", street_address), 1, 0)) %>%
  mutate(is_east = ifelse(grepl("(?i)\\b(East|E)\\b", street_address), 1, 0))

newtest <- testfinal %>%
  mutate(street_name = gsub("^\\d+\\s", "", street_address)) %>%
  mutate(street_name = gsub("(?i)\\bst\\b", "Street", street_name)) %>%
  mutate(street_name = gsub("(?i)\\bave\\b", "Avenue", street_name)) %>%
  mutate(street_name = gsub("\\.$", "", street_name))  %>%
  mutate(street_number = ifelse(grepl("^\\d+", street_address), 
                                gsub("\\s.*", "", street_address), 0))  %>%
mutate(is_west = ifelse(grepl("(?i)\\b(West|W)\\b", street_address), 1, 0)) %>%
  mutate(is_east = ifelse(grepl("(?i)\\b(East|E)\\b", street_address), 1, 0))

newtrain = dplyr::select(newtrain, -street_address)
newtest = dplyr::select(newtest, -street_address)

## Clearly need to use address field in a better way
street_ad = trainfinal$street_address 
street_string = as.data.frame(str_split_fixed(street_ad, " ", 2))
street_num = street_string[,1]
street_name = street_string[,2]

length(unique(street_ad)) ## less options
length(unique(street_num)) ## less options
length(unique(street_name)) ## less options

# do it for test
street_ad1 = testfinal$street_address 
street_string1 = as.data.frame(str_split_fixed(street_ad1, " ", 2))
street_num1 = street_string1[,1]
street_name1 = street_string1[,2]
length(unique(street_ad1)) ## less options
length(unique(street_num1)) ## less options
length(unique(street_name1)) ## less options

## add it back
trainfinal$street_num = street_num
trainfinal$street_name = street_name
testfinal$street_num = street_num1
testfinal$street_name = street_name1

trainfinnos = dplyr::select(trainfinal, -street_num, -street_name)
testfinalnos = dplyr::select(testfinal, -street_num, -street_name)

data <- trainfinal %>%
  mutate(street_name = gsub("^\\d+\\s", "", street_address)) %>%
  mutate(street_name = gsub("st$", "street", street_name, ignore.case = TRUE)) %>%
  mutate(street_name = gsub("ave$", "avenue", street_name, ignore.case = TRUE))

############# a lot of laundry variables -- lets see if we can cut down ######
laundry = as.data.frame(cbind(trainfinal$Laundry.in.Unit, trainfinal$Laundry, trainfinal$Laundry.in.Building, 
                trainfinal$Laundry.In.Building, trainfinal$Laundry.In.Unit,
                trainfinal$On.site.laundry, trainfinal$On.site.Laundry))

########################## let's delete all dupe variables ##############  -- still has street_address (var7)
trainfinalless = dplyr::select(trainfinal, -Childrens.Playroom, -Children.s.Playroom,  
-Concierge.Service, -Full.time.doorman, -FT.Doorman, -Hardwood.floors, -Hardwood, -High.ceilings,
-HIGH.CEILINGS, -gym, -Gym.Fitness, -Gym, -Gym.In.Building, -Marble.Bath, -Marble.Bathroom, -Prewar,
-prewar, -Common.roof.deck, -ROOFDECK, -Roof.deck, -Live.In.Superintendent, -Live.in.superintendent,
-Live.in.Super, -LIVE.IN.SUPER, -Laundry, -On.site.laundry, -On.site.Laundry)

testfinalless = dplyr::select(testfinal, -Childrens.Playroom, -Children.s.Playroom,  
                              -Concierge.Service, -Full.time.doorman, -FT.Doorman, -Hardwood.floors, -Hardwood, -High.ceilings,
                              -HIGH.CEILINGS, -gym, -Gym.Fitness, -Gym, -Gym.In.Building, -Marble.Bath, -Marble.Bathroom, -Prewar,
                              -prewar, -Common.roof.deck, -ROOFDECK, -Roof.deck, -Live.In.Superintendent, -Live.in.superintendent,
                              -Live.in.Super, -LIVE.IN.SUPER, -Laundry, -On.site.laundry, -On.site.Laundry)

## create price/bed, price/bath
trainfinalless$pperbed = trainfinalless$bedrooms/trainfinalless$price 
trainfinalless$pperbath = trainfinalless$bathrooms/trainfinalless$price
testfinalless$pperbed = testfinalless$bedrooms/testfinalless$price 
testfinalless$pperbath = testfinalless$bathrooms/testfinalless$price 

## how many "features" do they have?
trainfinalless$total = rowSums(trainfinalless[8:135])
testfinalless$total = rowSums(testfinalless[8:135])


######### section off neighbourhoods of long and lat ########

# Define latitude and longitude ranges for each neighborhood
uptown_lat_range <- c(40.7967, 40.8744)
midtown_lat_range <- c(40.7549, 40.7966)
downtown_lat_range <- c(40.7000, 40.7527)
brooklyn_lat_range <- c(40.5707, 40.7394)
queens_lat_range <- c(40.5417, 40.8006)
bronx_lat_range <- c(40.7855, 40.9176)

uptown_long_range <- c(-73.9884, -73.9194)
midtown_long_range <- c(-74.0088, -73.9782)
downtown_long_range <- c(-74.0155, -73.9865)
brooklyn_long_range <- c(-74.0472, -73.8572)
queens_long_range <- c(-73.9663, -73.7001)
bronx_long_range <- c(-73.9339, -73.7654)

trainfinalless <- trainfinalless %>%
  mutate(neighborhood = map2_chr(latitude, longitude, assign_neighborhood))

testfinalless <- testfinalless %>%
  mutate(neighborhood = map2_chr(latitude, longitude, assign_neighborhood))

trainfinalless <- trainfinalless %>%
  mutate(uptown = ifelse(neighborhood == "Uptown", 1, 0),
         midtown = ifelse(neighborhood == "Midtown", 1, 0),
         downtown = ifelse(neighborhood == "Downtown", 1, 0),
         brooklyn = ifelse(neighborhood == "Brooklyn", 1, 0),
         queens = ifelse(neighborhood == "Queens", 1, 0),
         bronx = ifelse(neighborhood == "Bronx", 1, 0),
         other = ifelse(!neighborhood %in% c("Uptown", "Midtown", "Downtown", "Brooklyn", "Queens", "Bronx"), 1, 0))

testfinalless <- testfinalless %>%
  mutate(uptown = ifelse(neighborhood == "Uptown", 1, 0),
         midtown = ifelse(neighborhood == "Midtown", 1, 0),
         downtown = ifelse(neighborhood == "Downtown", 1, 0),
         brooklyn = ifelse(neighborhood == "Brooklyn", 1, 0),
         queens = ifelse(neighborhood == "Queens", 1, 0),
         bronx = ifelse(neighborhood == "Bronx", 1, 0),
         other = ifelse(!neighborhood %in% c("Uptown", "Midtown", "Downtown", "Brooklyn", "Queens", "Bronx"), 1, 0))

# Function to assign neighborhood based on latitude and longitude
assign_neighborhood <- function(lat, long) {
  if (lat >= uptown_lat_range[1] && lat <= uptown_lat_range[2] &&
      long >= uptown_long_range[1] && long <= uptown_long_range[2]) {
    return("Uptown")
  } else if (lat >= midtown_lat_range[1] && lat <= midtown_lat_range[2] &&
             long >= midtown_long_range[1] && long <= midtown_long_range[2]) {
    return("Midtown")
  } else if (lat >= downtown_lat_range[1] && lat <= downtown_lat_range[2] &&
             long >= downtown_long_range[1] && long <= downtown_long_range[2]) {
    return("Downtown")
  } else if (lat >= brooklyn_lat_range[1] && lat <= brooklyn_lat_range[2] &&
             long >= brooklyn_long_range[1] && long <= brooklyn_long_range[2]) {
    return("Brooklyn")
  } else if (lat >= queens_lat_range[1] && lat <= queens_lat_range[2] &&
             long >= queens_long_range[1] && long <= queens_long_range[2]) {
    return("Queens")
  } else if (lat >= bronx_lat_range[1] && lat <= bronx_lat_range[2] &&
             long >= bronx_long_range[1] && long <= bronx_long_range[2]) {
    return("The Bronx")
  } else {
    return("Other")
  }
}

#################### REading in final dataset ############################
write.csv(trainfinalless, "C:\\Users\\cream\\OneDrive - Queen's University\\Desktop\\STAT457\\Project 2\\trainfinalless.csv",
          row.names=TRUE)
write.csv(testfinalless, "C:\\Users\\cream\\OneDrive - Queen's University\\Desktop\\STAT457\\Project 2\\testfinalless.csv",
          row.names=TRUE)

trainfinalless = read.csv("trainfinalless.csv") ## interest level is col 1
trainfinalless$interest_level = as.factor(trainfinalless$interest_level)
testfinalless = read.csv("testfinalless.csv") ## ID is col 1

######################### find vars that correlate most with interest_level ###
cortrain = trainfinalless
cortrain$interest_level = as.numeric(cortrain$interest_level)
cortrain = dplyr::select(cortrain, -street_address, -neighborhood)
corr_matrix = cor(cortrain)
corr_target <- sort(abs(corr_matrix[,1]), decreasing = TRUE)

print(names(corr_target)[1:6])

corrtrain = dplyr::select(newtrain, c(names(corr_target)[1:150]))
corrtest = dplyr::select(newtest, c(names(corr_target)[2:150]))
# Print the top correlated variables
print(names(corr_target)[1:10])

############# high dimension dataset -- let's do PCA for dimension reduction ####
train_x = select(trainfinal, -street_name, -street_num, -interest_level) 
test_x = select(testfinal, -street_name, -street_num, -ID)

## we need to exclude street address because it is not eligible for pca
## we need to normalize lat and longitude -- sigh

sort(colMeans(test_x))
## test set has 3 columns that are all 0 -- they need to be removed for PCA

train_x = select(train_x, -On.site.Parking.Lot, -Hi.Rise, -In.Unit.Washer.Dryer)
test_x = select(test_x, -On.site.Parking.Lot,- Hi.Rise, -In.Unit.Washer.Dryer)

pca = prcomp(train_x, scale = TRUE)
pcatest=prcomp(test_x, scale = TRUE)

pcaac = pca$x

par(mfrow=c(1,2))
ex_var = pca$sdev^2/sum(pca$sdev^2)
plot(ex_var, ylab = 'Explained percentage')
plot(cumsum(ex_var), ylab = 'Cumlative')
abline(h=0.90)
abline(v=100)

pcaac1 = pcatest$x

ggplot(final, aes(x = PCA1, y = PCA2, color = Y)) + 
  geom_point(size=1) +
  scale_x_continuous(limits = c(-25, 0)) 

trainpca = cbind.data.frame(pcaac[,1:100])
trainpca$interest_level = train$interest_level
testpca = cbind.data.frame(pcaac1[,1:100])

##### try classification trees ############
set.seed(3)
treedata = trainfinalless
treedata$interest_level = as.factor(trainfinalless$interest_level)
trainInd = sample(1:nrow(treedata), floor(0.7*nrow(treedata)))
treedata = select(treedata, -street_address)
tree_train = treedata[trainInd,]
tree_test = treedata[-trainInd,]
tree_test1 = select(testfinalless, -street_address)

ctrl <- trainControl(method = "cv",
                     number = 5,
                     allowParallel = TRUE)

rf.Grid = expand.grid(mtry = 2*(1:12),
                      splitrule ='gini', 
                      min.node.size = 1)

tic()
rf.cv.model <- train(interest_level ~ ., data = tree_train,
                     method = "ranger",
                     trControl = ctrl,
                     tuneGrid = rf.Grid)
toc()

yhat.cv.test = predict(rf.cv.model, tree_test)
table(yhat.cv.test, tree_test$interest_level)
mean(yhat.cv.test != tree_test$interest_level)

yhat.test = predict(rf.cv.model, tree_test1, type="prob")
preds = cbind(test$ID, yhat.test)

write.csv(preds, "C:\\Users\\cream\\OneDrive - Queen's University\\Desktop\\STAT457\\Project 2\\rf1.csv",
          row.names=TRUE)

###################### try logisitic regression ################
trainfinallessnos = select(trainfinalless, -street_address)
testfinallessnos = select(testfinalless, -street_address)

lnn = multinom(interest_level ~., data = trainfinallessnos)
phat = predict(lnn, newdata = testfinallessnos, type="probs")
write.csv(phat, "C:\\Users\\cream\\OneDrive - Queen's University\\Desktop\\STAT457\\Project 2\\logreg1.csv",
          row.names=TRUE)

### try random forest ####

## try more trees
rfmodel3 = randomForest(interest_level~., data=trainfinal, importance = T, mtry = 20
                        , ntree=500)
yhat.test = predict(rfmodel3, testfinal, type = "prob")
preds = cbind(test$ID, yhat.test)

write.csv(preds, "C:\\Users\\cream\\OneDrive - Queen's University\\Desktop\\STAT457\\Project 2\\rf5.csv",
          row.names=TRUE)

######## try with new total features ######## best result ########
rfmodel24 = randomForest(interest_level~. -street_address, data=trainfinalless, importance = T, mtry=24, ntree=750)
toc()
yhat.test = predict(rfmodel24, testfinalless, type = "prob")

write.csv(yhat.test, "C:\\Users\\cream\\OneDrive - Queen's University\\Desktop\\STAT457\\Project 2\\r24.csv",
          row.names=TRUE)

##################### try svm ####################
rfSVM.Grid = expand.grid(C = (1:6)*10, 
                         sigma = (1:8)*0.001, 
                         Weight = 1)
ctrl <- trainControl(method = "cv",
                     number = 5,
                     allowParallel = TRUE)

tic()
rfSVM.cv.model <- train(interest_level ~ ., data = svmtrain,
                        method = "svmRadialWeights",
                        trControl = ctrl,
                        tuneGrid = rfSVM.Grid)
toc()
 rfSVM.cv.model
 rfSVM.cv.pred = predict(rfSVM.cv.model, newdata =svmtest, type="prob" )
## best result @ sigma = 0.005, C=50, weight = 1 took 1 hour to train -- don't try again
 svmtest = as.data.frame(svmtest)
 
 X = svmtest[,2:134]
 Y = svmtest[,1]
 
yhat.svm = extractProb(models =list(rfSVM.cv.model), testX = X, testY = Y)
 
 ################ try just using a base svm #############
 svmdata = as.data.frame(scale(trainfinallessnos[,2:5]))
 svmdata1 = as.data.frame(scale(testfinallessnos[,2:5]))
colnames(svmtrain)[1]= "interest_level"

svmtrain = cbind(trainfinallessnos[,1], svmdata, trainfinallessnos[6:134])
svmtest = cbind(testfinallessnos$ID, svmdata1, testfinallessnos[6:134])

svm2 = svm(interest_level~., data = svmtrain, probability=TRUE)
pred_prob <- predict(svm2, svmtest[,-1], probability = TRUE)
getanswer=attr(pred_prob, "probabilities")

write.csv(getanswer, "C:\\Users\\cream\\OneDrive - Queen's University\\Desktop\\STAT457\\Project 2\\svm2.csv",
          row.names=TRUE)

####### try xgboost ####################
set.seed(7)

 ## xgboost only seems to support num
 
 ## get rid of street add for xgboost
 trainxg = dplyr::select(trainfinalless, -neighborhood, -street_address)
 testxg = dplyr::select(testfinalless, -neighborhood, -street_address)
 
 ## convert to numeric
 trainxg = sapply(trainxg, as.numeric)
 testxg = sapply(testxg, as.numeric)
 trainxg[,1] = trainxg[,1]-1 #### need predictions to be 0,1,2 instead of 1,2,3
 
 set.seed(7)
 split_index = createDataPartition(trainfinalless$interest_level, p = 0.7, list = FALSE, times = 1)
 xgb.trainy = trainxg[split_index,]
 xgb.testy = trainxg[-split_index,]
   
xgb.train=xgb.DMatrix(data=xgb.trainy[,-1], label=xgb.trainy[,1])
xgb.test = xgb.DMatrix(data=xgb.testy[,-1], label=xgb.testy[,1])

xgb.finaltest =xgb.DMatrix(data=testxg[,-1])

params = list(
  booster="gbtree",
  eta=0.3,
  max_depth = 6,
  gamma = 1,
  objective="multi:softprob",
  nthread = 4,
  num_class=3,
  min_child_weight = 1
)

nround    <- 1000 # number of XGBoost rounds
cv.nfold  <- 5
cv_model <- xgb.cv(params = params,
                   data = xgb.train, 
                   nrounds = nround,
                   nfold = cv.nfold,
                   verbose = FALSE,
                   prediction = TRUE)

xgb.fit=xgb.train(
  params=params,
  data=xgb.train,
  nrounds=1000,
  nthreads=1,
  early_stopping_rounds=10,
  watchlist = list(val1=xgb.train, val2=xgb.test),
  verbose=1
)

xgb.preds = predict(xgb.fit, testxg[,-1], reshape=T)
write.csv(xgb.preds, "C:\\Users\\cream\\OneDrive - Queen's University\\Desktop\\STAT457\\Project 2\\xgb17.csv",
          row.names=TRUE)


################# tune xgb ##########
# Customsing the tuning grid
gbmGrid <-  expand.grid(max_depth = c(3, 5, 7), 
                        nrounds = c(100,500,1000,10000),    # number of trees
                        eta = c(0.1,0,0.3),
                        gamma = 0,
                        subsample = 1,
                        min_child_weight = 1,
                        colsample_bytree = 0.6)

train_control = trainControl(method="cv", number=5, search="grid", allowParallel = TRUE)

xgbtune = train(interest_level~., data = xgb.trainy, method = "xgbTree", trControl = train_control, tuneGrid = gbmGrid)
## predict function won't give probabilities -- need to run xgb.train

params = list(
  booster="gbtree",
  eta=0.1,
  max_depth = 3,
  gamma = 0,
  objective="multi:softprob",
  nthread = 4,
  num_class=3,
  min_child_weight = 1
)

xgb.fit2=xgb.train(
  params=params,
  data=xgb.train,
  nrounds=100,
  colsample_bytree=0.6,
  subsample=1,
  nthreads=1,
  early_stopping_rounds=10,
  watchlist = list(val1=xgb.train, val2=xgb.test),
  verbose=1
)
xgb.preds = predict(xgb.fit2, testxg[,-1], reshape=T)
write.csv(xgb.preds, "C:\\Users\\cream\\OneDrive - Queen's University\\Desktop\\STAT457\\Project 2\\xgb18.csv",
          row.names=TRUE)

############ feature importance
names = colnames(testxg[,-1])
importance_matrix = xgb.importance(feature_names = names, model = xgb.fit)
head(importance_matrix)
gp = xgb.ggplot.importance(importance_matrix)
print(gp) 


################## try knn ###################### -- really bad 
knnFit <- train(interest_level ~ ., data = svmtrain, method = "knn", 
                trControl = trainControl(method = "cv"))

knn_res = predict(knnFit, svmtest[-1], type = "prob")
write.csv(knn_res, "C:\\Users\\cream\\OneDrive - Queen's University\\Desktop\\STAT457\\Project 2\\knn.csv",
          row.names=TRUE)
