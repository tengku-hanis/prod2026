##=============================================================================##
## Title:  Basic workflow for tidymodels 
## Author: Tengku Muhd Hanis Bin Tengku Mokhtar, PhD
## Date: June 16, 2026
##=============================================================================##

# Packages ---------------------------------------------------------------

# Install
install.packages(c("tidymodels", "tidyverse", "mlbench", "medicaldata", "skimr", "vip", "glmnet"))

# Load
library(tidymodels)
library(tidyverse)
library(mlbench)


# Data -------------------------------------------------------------------

# Load data
data(PimaIndiansDiabetes)

# Data description
?PimaIndiansDiabetes


# Descriptive ------------------------------------------------------------

# Original data
skimr::skim(PimaIndiansDiabetes)

# Remove a few data to make balance the data
set.seed(123)
data_pima <- 
  PimaIndiansDiabetes %>% 
  group_by(diabetes) %>% 
  slice_sample(n = 268) %>% 
  ungroup() %>% 
  # Set positive as our main prediction
  mutate(diabetes = relevel(diabetes, ref = "pos"))
  

# A reduced but balanced data
skimr::skim(data_pima)

# Split data -------------------------------------------------------------

# Set seed for reproducibility
set.seed(123)

# Split
ind <- initial_split(data_pima, prop = 0.8)

# Training and testing data
data_train <- training(ind)
data_test <- testing(ind)

# Create 10-fold CV
data_cv <- vfold_cv(data_train, v = 10)


# Specify recipe ---------------------------------------------------------

rec_logr <- 
  recipe(diabetes ~., data = data_train)

# Specify the models -----------------------------------------------------

# Logistic regression
spec_logr <- 
  logistic_reg(
    penalty = tune(),
    mixture = tune()
  ) %>% 
  set_mode("classification") %>% 
  set_engine("glmnet")


# Specify workflow -------------------------------------------------------

wf_logr <- 
  workflow() %>%
  add_model(spec = spec_logr) %>% 
  add_recipe(recipe = rec_logr)


# Tune the model ---------------------------------------------------------

# Specify performance metrics
perf_metrics <- metric_set(accuracy, sensitivity, specificity)

# Tune
set.seed(123)
tuned_res <- 
  wf_logr %>% 
  tune_grid(
    grid = 50,
    metrics = perf_metrics,
    resamples = data_cv
  )


# Evaluate tuning result -------------------------------------------------

# General results
autoplot(tuned_res) + theme_bw()
tuned_res %>% collect_metrics() 

# Result for specific metric
tuned_res %>% show_best(metric = "accuracy")
tuned_res %>% show_best(metric = "sensitivity")
tuned_res %>% show_best(metric = "specificity")

# Select the best model
best_tune <- 
  tuned_res %>% 
  select_best(metric = "accuracy")


# Finalise workflow ------------------------------------------------------

wf_logr_final <- 
  wf_logr %>% 
  finalize_workflow(best_tune)


# Re-fit on training data -------------------------------------------------

logr_trained <- 
  wf_logr_final %>% 
  fit(data = data_train)


# Visualise variable importance ------------------------------------------

# Extract raw model
logr_raw_model <- 
  logr_trained %>% 
  extract_fit_parsnip()

# Visualise
vip::vip(logr_raw_model)


# Assess on testing data --------------------------------------------------

# Predict on the new data
pima_pred <- 
  data_test %>% 
  bind_cols(predict(logr_trained, new_data = data_test)) %>% 
  bind_cols(predict(logr_trained, new_data = data_test, type = "prob"))

# Performance metrics 
# 1) custom metric set to evaluate performance
test_performance <- metric_set(accuracy, sensitivity, specificity) 
test_performance(pima_pred, truth = diabetes, estimate = .pred_class)

# 2) Specific metrics
## Accuracy
pima_pred %>% 
  accuracy(truth = diabetes, estimate = .pred_class)

## Plot ROC
pima_pred %>% 
  roc_curve(diabetes, .pred_pos) %>% 
  autoplot()

## ROC-AUC
pima_pred %>% 
  roc_auc(diabetes, .pred_pos)

# 3) Confusion matrix
conf_mat(pima_pred, truth = diabetes, estimate = .pred_class) %>% 
  autoplot("heatmap")

# 4) All available metrics
conf_mat(pima_pred, truth = diabetes, estimate = .pred_class) %>% 
  summary()
