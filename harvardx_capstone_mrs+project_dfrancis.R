
### HarvardX Capstone Project: Movie Recommendation Systems ###
##                      Francis Dzakpasu


## Installing and loading required packages and libraries
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(Metrics)) install.packages("Metrics", repos = "http://cran.us.r-project.org")
if(!require(kableExtra)) install.packages("kableExtra", repos = "http://cran.us.r-project.org")
if(!require(recosystem)) install.packages("recosystem", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(broom)
library(lubridate)
library(gam)
library(randomForest)
library(Rborist)
library(matrixStats)
library(rafalib)
library(naivebayes)
library(ggplot2)
library(ggthemes)
library(Metrics)
library(scales)
library(dslabs)
ds_theme_set()
library(knitr)
library(kableExtra)
library(recosystem)

### Downloading and cleaning the MovieLens datasets ###
# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

options(timeout = 120)

dl <- "ml-10M100K.zip"
if(!file.exists(dl))
  download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings_file <- "ml-10M100K/ratings.dat"
if(!file.exists(ratings_file))
  unzip(dl, ratings_file)

movies_file <- "ml-10M100K/movies.dat"
if(!file.exists(movies_file))
  unzip(dl, movies_file)

ratings <- as.data.frame(str_split(read_lines(ratings_file), fixed("::"), 
                                   simplify = TRUE),
                                   stringsAsFactors = FALSE)
colnames(ratings) <- c("userId", "movieId", "rating", "timestamp")
ratings <- ratings %>%
  mutate(userId = as.integer(userId),
         movieId = as.integer(movieId),
         rating = as.numeric(rating),
         timestamp = as.integer(timestamp))

movies <- as.data.frame(str_split(read_lines(movies_file), fixed("::"), 
                                  simplify = TRUE),
                                  stringsAsFactors = FALSE)
colnames(movies) <- c("movieId", "title", "genres")
movies <- movies %>%
  mutate(movieId = as.integer(movieId))

# Merging the two data frames into one
movielens <- left_join(ratings, movies, by = "movieId")

# Final hold-out test set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.6 or later
# set.seed(1) # if using R 3.5 or earlier
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in final hold-out test set are also in edx set
final_holdout_test <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from final hold-out test set back into edx set
removed <- anti_join(temp, final_holdout_test)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

## Save the cleaned data - final hold-out test and edx sets

# final_holdout_test - ONLY be used for evaluating the RMSE of your final algorithm
save(final_holdout_test, file = "final_holdout_test.RData")

# edx data - training and testing sets and/or use cross-validation to design and test algorithm
save(edx, file = "edx.RData")

## Load the saved the datasets 
load("final_holdout_test.RData")
load("edx.RData")


## edx dataset exploration to familarise with the structure

# The class of the dataset
class(edx)

# The structure of the dataset
str(edx, vec.len = 2)

head(edx)

summary(edx)

# The number of unique users ratings and the unique movies rated
n_users_movies <- edx %>% 
  summarize(tibble(unique_users = n_distinct(userId),
                   unique_movies = n_distinct(movieId)))

## The most common genre and average ratings
edx %>%
  separate_rows(genres,
                sep = "\\|") %>%
  group_by(genres) %>%
  summarize(n = n(),
            avg_ratings = mean(rating)) %>%
  arrange(desc(n))

## Knowing all the genres associated with the movies in the MovieLens dataset,
## the number of ratings for the individual genres can be determined as follow:

genres <- c("Action", "Adventure", "Animation", "Children", "Comedy", "Crime", 
            "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "IMAX", 
            "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western")
n_ratings <- function(x) {
  sum(str_detect(edx$genres, x))
}
tibble(Genres = genres,
       Count = unlist(lapply(genres, n_ratings))) 


## The common rating score
edx %>%
  group_by(rating) %>%
  summarise(n = n()) %>%
  arrange(desc(n))


## A plot of total number of movies released per year
edx  %>%  
  mutate(year = as.numeric(str_extract(str_extract(title, "[/(]\\d{4}[/)]$"), regex("\\d{4}")))) %>% 
  group_by(year) %>% 
  mutate(n=n()) %>%
  ggplot(aes(year)) +
  geom_bar(fill = "grey", color = "black") + 
  ggtitle("Movies' released per year") +
  xlab("Year") +
  ylab("Number of movies") +
  scale_y_continuous(n.breaks = 10, labels = scales::label_comma()) + 
  scale_x_continuous(n.breaks = 15) +
  theme_bw() +
  theme(axis.title.x = element_text(vjust = 0), 
        axis.title.y = element_text(vjust = 2),
        plot.title = element_text(hjust=0.5)) 


## Ratings per year plot
edx %>% 
  mutate(rating_year = year(as_datetime(timestamp, origin = "1970-01-01"))) %>%
  ggplot(aes(x = rating_year)) +
  geom_bar(fill = "grey", color = "black") + 
  ggtitle("Movies' ratings per year") +
  xlab("Year movie rated") +
  ylab("Number of ratings") +
  scale_y_continuous(n.breaks = 8) + 
  scale_x_continuous(n.breaks = 10) +
  theme_bw() +
  theme(axis.title.x = element_text(vjust = 0), 
        axis.title.y = element_text(vjust = 2),
        plot.title = element_text(hjust=0.5))


# Visualiasation of users and movies ratings
# Users' rating histogram 
hist_users <- edx %>%
  group_by(userId) %>%
  summarize(count = n()) %>%
  ggplot(aes(x = count)) +
  geom_histogram(fill = "grey", color = "black") +
  ggtitle("Users' rating") +
  xlab("Rating count") +
  ylab("Number of users") +
  scale_y_continuous(n.breaks = 10) +
  scale_x_log10(n.breaks = 5) +
  theme_bw() +
  theme(axis.title.x = element_text(vjust = 0), 
        axis.title.y = element_text(vjust = 2),
        plot.title = element_text(hjust=0.5))

## Movies' ratings plot
hist_movies <-edx %>%
  group_by(movieId) %>%
  summarize(count = n()) %>%
  ggplot(aes(x = count)) +
  geom_histogram(width="40", fill = "grey", color = "black") +
  ggtitle("Movies' ratings") +
  xlab("Rating count") +
  ylab("Number of movies") +
  scale_y_continuous(n.breaks = 10) +
  scale_x_log10(n.breaks = 5) +
  theme_bw() +
  theme(axis.title.x = element_text(vjust = 0), 
        axis.title.y = element_text(vjust = 2),
        plot.title = element_text(hjust=0.5))

gridExtra::grid.arrange(hist_users, hist_movies, ncol = 2)

## Matrix construction for randomly selected users and movies 
N <- 200
matrix_const <- edx %>% 
  filter(userId %in% sample(unique(edx$userId), N)) %>%
  select(userId, movieId, rating) %>%
  mutate(rating = 1) %>%
  spread(movieId, rating) %>% 
  select(sample(ncol(.), N)) %>% 
  as.matrix() %>% 
  t(.) # to transposes the matrix

# Matrix plot
matrix_const %>% 
  image(1:N, 1:N,. , xlab="Movies", ylab="Users")
abline(h=0:N+0.5, v=0:N+0.5, col = "grey")
title(main = list("Matrix", cex = 2, font = 4)) 


### Building the Recommendation Systems ###
##  Spliting edx dataset into train_edx (training) and test_edx (testing) sets: 
## also, 10% of the edx set will be used to validate 
##  the trained algorithms. 
# Seed set to 1
set.seed(1)
test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.1, list = FALSE)
train_edx <- edx[-test_index,]
test_data <- edx[test_index,]

# Ensuring userId and movieId in the training set are also in testing set 
test_edx <- test_data %>% 
  semi_join(train_edx, by = "movieId") %>%
  semi_join(train_edx, by = "userId")

# Adding rows removed from the testing set back into the training set
removed <- anti_join(test_data, test_edx)
train_edx <- rbind(train_edx, removed) 

## Number of observations in the partitioned data sets 
nrow(train_edx)

nrow(test_edx)


## Model Evaluation function
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

### Building the prediction algorithms ###

### Using Linear regression Modelling approach
## initial model: the base model with only the mean rating without considering the users and the movies 

mu <- mean(train_edx$rating)
y_hat_base <- rep(mu, nrow(test_edx))

base_model_rmse <- RMSE(y_hat_base, test_edx$rating)


rmse_evaluations <- tibble(Method = "Linear regression: base model", RMSE = base_model_rmse) 

rmse_evaluations


### Accounting for movies and users effect in the linear regression model

## Calculate the movies' mean rating and the adjusted effect 

movie_bias <- train_edx %>%
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu),   # b_i = adjusted movie effect
            b_i0 = mean(rating))       # b_i0 = average movies' ratings

## Calculate the users' mean rating and the adjusted effect 
user_bias <- train_edx %>%
  left_join(movie_bias, by = 'movieId') %>%
  group_by(userId) %>% 
  summarize(b_u = mean(rating - (mu+b_i)), # b_u = adjusted user effect
            b_u0 = mean(rating))           # b_u0 = average users' ratings


## Plot the movies and users mean ratings and adjusted effects
# Average movies ratings
p1_avg <-  ggplot(movie_bias, aes(x = b_i0)) + 
  geom_histogram(aes(y = ..density..),
                 colour = "chocolate3", 
                 fill = "peachpuff") +
  geom_density(col = "chocolate3", size=0.7) +
  ggtitle("Average movies ratings") +
  xlab("Rating") +
  ylab("Density") +
  scale_y_continuous(n.breaks = 8) +
  scale_x_continuous(n.breaks = 10) +
  theme_bw() +
  theme(axis.title.x = element_text(vjust = 0), 
        axis.title.y = element_text(vjust = 2),
        plot.title = element_text(hjust=0.5))

# Adjusted movies effects
p2_adj <- ggplot(movie_bias, aes(x = b_i)) + 
  geom_histogram(aes(y = ..density..),
                 colour = "black", 
                 fill = "grey") +
  geom_density(col = "black", size=0.7) +
  ggtitle("Adjusted movies' effect") +
  xlab("Effect size") +
  ylab("Density") +
  scale_y_continuous(n.breaks = 8) +
  scale_x_continuous(n.breaks = 10) +
  theme_bw() +
  theme(axis.title.x = element_text(vjust = 0), 
        axis.title.y = element_text(vjust = 2),
        plot.title = element_text(hjust=0.5)) 

# Average users ratings
p3_avg <- ggplot(user_bias, aes(x = b_u0)) + 
  geom_histogram(aes(y = ..density..),
                 colour = "chocolate3", 
                 fill = "peachpuff") +
  geom_density(col = "chocolate3", size=0.7) + 
  ggtitle("Average users ratings") +
  xlab("Rating") +
  ylab("Density") +
  scale_y_continuous(n.breaks = 8) +
  scale_x_continuous(n.breaks = 10) +
  theme_bw() +
  theme(axis.title.x = element_text(vjust = 0), 
        axis.title.y = element_text(vjust = 2),
        plot.title = element_text(hjust=0.5))

# Adjusted users effects
p4_adj <- ggplot(user_bias, aes(x = b_u)) + 
  geom_histogram(aes(y = ..density..),
                 colour = "black", 
                 fill = "grey") +
  geom_density(col = "black", size=0.7) + 
  ggtitle("Adjusted users' effect") +
  xlab("Effect size") +
  ylab("Density") +
  scale_y_continuous(n.breaks = 8) +
  scale_x_continuous(n.breaks = 10) +
  theme_bw() +
  theme(axis.title.x = element_text(vjust = 0), 
        axis.title.y = element_text(vjust = 2),
        plot.title = element_text(hjust=0.5)) 

# Combining the plots
gridExtra::grid.arrange(p1_avg, p2_adj, p3_avg, p4_adj, ncol = 2)


### Adjusted linear regression models
## Model with only movie effect
y_hat_movie <- mu + test_edx %>%
  left_join(movie_bias, by = "movieId") %>%
  pull(b_i)

movie_only_rmse <- RMSE(y_hat_movie, test_edx$rating)
knitr::kable(rmse_evaluations <- bind_rows(rmse_evaluations,
                                           tibble(Method="Linear regression: with movies effect",
                                                  RMSE = movie_only_rmse )))

## Model with adjusted movies' and users' effect
y_hat_adjusted <- test_edx %>%
  left_join(movie_bias, by='movieId') %>%
  left_join(user_bias, by='userId') %>%
  mutate(y_hat_adjusted = mu+b_i+b_u) %>%
  pull(y_hat_adjusted)

adjusted_effects_rmse <- RMSE(y_hat_adjusted, test_edx$rating)
knitr::kable(rmse_evaluations <- bind_rows(rmse_evaluations,
                                           tibble(Method="Linear regression: adjusted movies' & users' effect",
                                                  RMSE = adjusted_effects_rmse)))


### Testing the trained linear regression recommendation system on the 'final holdout test' dataset
movie_recomd <- final_holdout_test %>%
  left_join(movie_bias, by = "movieId") %>%
  left_join(user_bias, by = "userId") %>%
  mutate(y_hat_movie_recomd = mu+b_i+b_u) %>%
  pull(y_hat_movie_recomd)

movie_recomd_rmse <- RMSE(movie_recomd, final_holdout_test$rating)
knitr::kable(rmse_evaluations <- bind_rows(rmse_evaluations,
                                           tibble(Method="Linear regression: trained model on final holdout test",
                                                  RMSE = movie_recomd_rmse)))

## Best movies recommended by the linear model on the final_holdout_test set
best20_movies <- final_holdout_test %>%
  left_join(movie_bias, by = "movieId") %>%
  left_join(user_bias, by = "userId") %>%
  mutate(y_hat_best_preds = mu+b_i+b_u) %>%
  arrange(desc(y_hat_best_preds)) %>%
  select(title) %>% 
  unique() %>% slice(1:20)
best20_movies

## Best predictions with number of ratings and average rating
bp_lr <- tibble(No. = 1:nrow(best20_movies),
                Movie_Title = character(length = nrow(best20_movies)), 
                Count = integer(length = nrow(best20_movies)), 
                Average_Rating = numeric(length = nrow(best20_movies)))
# for looping
for (i in seq_along(best20_movies$title)) {
  ind <- which(final_holdout_test$title == as.character(best20_movies$title[i]))
  avg_rating <- mean(final_holdout_test$rating[ind])
  count <- sum(final_holdout_test$title == as.character(best20_movies$title[i]))
  bp_lr$Movie_Title[i] <- best20_movies$title[i]
  bp_lr$Count[i] <- count
  bp_lr$Average_Rating[i] <- round(avg_rating, 2)
}
knitr::kable(bp_lr, caption = "Best linear regression system predicted movies")

## Worst movies recommended by the linear model on the final_holdout_test set
worst20_movies <- final_holdout_test %>%
  left_join(movie_bias, by = "movieId") %>%
  left_join(user_bias, by = "userId") %>%
  mutate(y_hat_best_preds = mu+b_i+b_u) %>%
  arrange(y_hat_best_preds) %>%
  select(title) %>% 
  unique() %>% slice(1:20)
worst20_movies

## Worst predictions with number of ratings and average rating
wp_lr <- tibble(No. = 1:nrow(worst20_movies),
                Movie_Title = character(length = nrow(worst20_movies)), 
                Count = integer(length = nrow(worst20_movies)), 
                Average_Rating = numeric(length = nrow(worst20_movies)))
# for looping
for (i in seq_along(worst20_movies$title)) {
  ind <- which(final_holdout_test$title == as.character(worst20_movies$title[i]))
  avg_rating <- mean(final_holdout_test$rating[ind])
  count <- sum(final_holdout_test$title == as.character(worst20_movies$title[i]))
  wp_lr$Movie_Title[i] <- worst20_movies$title[i]
  wp_lr$Count[i] <- count
  wp_lr$Average_Rating[i] <- round(avg_rating, 2)
}
knitr::kable(wp_lr, caption = "Worst linear regression system predicted movies")


### Regularization technique to improve the linear regression prediction further ###

## Generating the regulisation function
regular_fun <- function(lambda, train, test){
  mu <- mean(train_edx$rating)
  
  movie_bias <- train_edx %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+lambda))
  
  user_bias <- train_edx %>% 
    left_join(movie_bias, by="movieId") %>%
    filter(!is.na(b_i)) %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - (mu+b_i))/(n()+lambda))
  
  predicted_ratings <- test_edx %>% 
    left_join(movie_bias, by = "movieId") %>%
    left_join(user_bias, by = "userId") %>%
    filter(!is.na(b_i), !is.na(b_u)) %>%
    mutate(pred = mu+b_i+b_u) %>%
    pull(pred)
  
  return(RMSE(predicted_ratings, test_edx$rating))
}

### Applying the regularisation function
lambda_rmse <- seq(0, 10, 0.25)
tune_rmse <- sapply(lambda_rmse,
                    regular_fun, 
                    train = train_edx, 
                    test = test_edx)

### Plotting the lambda effects on RMSE
qplot(lambda_rmse, tune_rmse) +
  ggtitle("Lambda tuning cross validation curve") +
  xlab("Lambda") +
  ylab("RMSE") +
  scale_y_continuous(n.breaks = 8) +
  scale_x_continuous(n.breaks = 10) +
  theme_bw() +
  theme(axis.title.x = element_text(vjust = 0), 
        axis.title.y = element_text(vjust = 2),
        plot.title = element_text(hjust=0.5)) 

###Linear regression model using tuned regularisation parameters
# Evaluating trained regularisation system on the test_edx data set 
lambda <- lambda_rmse[which.min(tune_rmse)]
mu <- mean(train_edx$rating)

movie_bias_regularised <- train_edx %>% 
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu)/(n()+lambda))

user_bias_regularised <- train_edx %>% 
  left_join(movie_bias_regularised, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - (mu+b_i))/(n()+lambda))

y_hat_regularised <- test_edx %>% 
  left_join(movie_bias_regularised, by = "movieId") %>%
  left_join(user_bias_regularised, by = "userId") %>%
  mutate(pred_regularised = mu+b_i+b_u) %>%
  pull(pred_regularised)

regularised_rmse <- RMSE(y_hat_regularised, test_edx$rating)
knitr::kable(rmse_evaluations <- bind_rows(rmse_evaluations,
                                    tibble(Method="Linear regression: regularisation on test_edx",
                                           RMSE = regularised_rmse)))



## Evaluating trained regularised model with the final_holdout_test 
mu <- mean(train_edx$rating)

lambda <- lambda_rmse[which.min(tune_rmse)]
lambda

movie_bias_regularised <- train_edx %>% 
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu)/(n()+lambda))

user_bias_regularised <- train_edx %>% 
  left_join(movie_bias_regularised, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - (mu+b_i))/(n()+lambda))

y_hat_regularised <- final_holdout_test %>% 
  left_join(movie_bias_regularised, by = "movieId") %>%
  left_join(user_bias_regularised, by = "userId") %>%
  mutate(pred_regularised = mu+b_i+b_u) %>%
  pull(pred_regularised)

regularised_rmse <- RMSE(y_hat_regularised, final_holdout_test$rating)
knitr::kable(rmse_evaluations <- bind_rows(rmse_evaluations,
                                           tibble(Method="Linear regression: regularisation on final hold out test",
                                                  RMSE = regularised_rmse)))

### Best movie recommendations by regularisation model 
best20_movies_regularised <- final_holdout_test %>%
  left_join(movie_bias_regularised, by = "movieId") %>%
  left_join(user_bias_regularised, by = "userId") %>%
  mutate(y_hat_regularised = mu + b_i + b_u) %>%
  arrange(desc(y_hat_regularised)) %>%
  select(title) %>% 
  unique() %>%
  slice(1:20)
best20_movies_regularised

## Best regularisation predictions with number of ratings and average rating
bp_lr_regularised <- tibble(No. = 1:nrow(best20_movies_regularised),
                            Movie_Title = character(length = nrow(best20_movies_regularised)), 
                            Count = integer(length = nrow(best20_movies_regularised)), 
                            Average_Rating = numeric(length = nrow(best20_movies_regularised)))
# for looping
for (i in seq_along(best20_movies_regularised$title)) {
  ind <- which(final_holdout_test$title == as.character(best20_movies_regularised$title[i]))
  avg_rating <- mean(final_holdout_test$rating[ind])
  count <- sum(final_holdout_test$title == as.character(best20_movies_regularised$title[i]))
  bp_lr_regularised$Movie_Title[i] <- best20_movies_regularised$title[i]
  bp_lr_regularised$Count[i] <- count
  bp_lr_regularised$Average_Rating[i] <- round(avg_rating, 2)
}
knitr::kable(bp_lr_regularised, caption = "Best regularisation system predicted movies")

### Worst movie recommendations by regularisation model 
worst20_movies_regularised <- final_holdout_test %>%
  left_join(movie_bias_regularised, by = "movieId") %>%
  left_join(user_bias_regularised, by = "userId") %>%
  mutate(y_hat_regularised = mu + b_i + b_u) %>%
  arrange(y_hat_regularised) %>%
  select(title) %>% 
  unique() %>%
  slice(1:20)
worst20_movies_regularised

## Worst regularisation predictions with number of ratings and average rating
wp_lr_regularised <- tibble(No. = 1:nrow(worst20_movies_regularised),
                            Movie_Title = character(length = nrow(worst20_movies_regularised)), 
                            Count = integer(length = nrow(worst20_movies_regularised)), 
                            Average_Rating = numeric(length = nrow(worst20_movies_regularised)))
# for looping
for (i in seq_along(worst20_movies_regularised$title)) {
  ind <- which(final_holdout_test$title == as.character(worst20_movies_regularised$title[i]))
  avg_rating <- mean(final_holdout_test$rating[ind])
  count <- sum(final_holdout_test$title == as.character(worst20_movies_regularised$title[i]))
  wp_lr_regularised$Movie_Title[i] <- worst20_movies_regularised$title[i]
  wp_lr_regularised$Count[i] <- count
  wp_lr_regularised$Average_Rating[i] <- round(avg_rating, 2)
}
knitr::kable(wp_lr_regularised, caption = "Worst regularisation system predicted movies")



### Matrix factorization recommendation system ###
## First, the input format edx dataset (both the train_edx and test_edx) is transformed into recosystem format
set.seed(1)
recosys_train <- with(train_edx, data_memory(user_index = userId, item_index = movieId, rating = rating))
recosys_test <- with(test_edx, data_memory(user_index = userId, item_index = movieId, rating = rating))

## Creating the model object
recosys <- Reco()

## Model tuning
model_tune <- recosys$tune(recosys_train, opts = list(dim = c(10, 20, 30), lrate = c(0.1, 0.2), nthread  = 4, niter = 10)) 

## training the model
recosys$train(recosys_train, opts = c(model_tune$min, nthread = 4, niter = 20))



## Making prediction on the recosys_test (test_edx) set
y_hat_mf <-  recosys$predict(recosys_test, out_memory()) ## Return an R vector

mf_rmse_testedx <- RMSE(y_hat_mf, test_edx$rating)

knitr::kable(rmse_evaluations <- bind_rows(rmse_evaluations,
                              tibble(Method="Matrix factorisation system on test_edx",
                                     RMSE = mf_rmse_testedx )))

## Movie recommendations by the matrix factorization model
recosys_finaltest <- with(final_holdout_test, data_memory(user_index = userId, item_index = movieId, rating = rating))

## prediction on final holdout test
y_hat_mf_finaltest <-  recosys$predict(recosys_finaltest, out_memory())

mf_rmse_finaltest <- RMSE(y_hat_mf_finaltest, final_holdout_test$rating)
knitr::kable(rmse_evaluations <- bind_rows(rmse_evaluations,
                                    tibble(Method="Matrix factorisation system on final hold-out test",
                                           RMSE = mf_rmse_finaltest )))

#  Best recommendations - MF
best20_movies_mf <- data.frame(title=final_holdout_test$title, pred_rating=y_hat_mf_finaltest) %>%
  arrange(desc(pred_rating)) %>%
  select(title) %>% 
  unique() %>%
  slice(1:20)
best20_movies_mf

## Best MF predictions with number of ratings and average rating
bp_mf <- tibble(No. = 1:nrow(best20_movies_mf),
                Movie_Title = character(length = nrow(best20_movies_mf)), 
                Count = integer(length = nrow(best20_movies_mf)), 
                Average_Rating = numeric(length = nrow(best20_movies_mf)))
# for looping
for (i in seq_along(best20_movies_mf$title)) {
  ind <- which(final_holdout_test$title == as.character(best20_movies_mf$title[i]))
  avg_rating <- mean(final_holdout_test$rating[ind])
  count <- sum(final_holdout_test$title == as.character(best20_movies_mf$title[i]))
  bp_mf$Movie_Title[i] <- best20_movies_mf$title[i]
  bp_mf$Count[i] <- count
  bp_mf$Average_Rating[i] <- round(avg_rating, 2)
}
knitr::kable(bp_mf, caption = "Best matrix factorisation system predicted movies")


# Worst recommendations - MF
worst20_movies_mf <- data.frame(title=final_holdout_test$title, pred_rating=y_hat_mf_finaltest) %>%
  arrange(pred_rating) %>%
  select(title) %>% 
  unique() %>%
  slice(1:20)
worst20_movies_mf

## Worst MF predictions with number of ratings and average rating
wp_mf <- tibble(No. = 1:nrow(worst20_movies_mf),
                Movie_Title = character(length = nrow(worst20_movies_mf)), 
                Count = integer(length = nrow(worst20_movies_mf)), 
                Average_Rating = numeric(length = nrow(worst20_movies_mf)))
# for looping
for (i in seq_along(worst20_movies_mf$title)) {
  ind <- which(final_holdout_test$title == as.character(worst20_movies_mf$title[i]))
  avg_rating <- mean(final_holdout_test$rating[ind])
  count <- sum(final_holdout_test$title == as.character(worst20_movies_mf$title[i]))
  wp_mf$Movie_Title[i] <- worst20_movies_mf$title[i]
  wp_mf$Count[i] <- count
  wp_mf$Average_Rating[i] <- round(avg_rating, 2)
}
knitr::kable(wp_mf, caption = "Worst matrix factorisation system predicted movies")


### RMSE table
knitr::kable(rmse_evaluations, caption = "Recommendation systems evaluation") %>%
  kable_classic(full_width = F)




