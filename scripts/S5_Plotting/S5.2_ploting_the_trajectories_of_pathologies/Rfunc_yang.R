#library(caret)
#library(glmnet)
#library(ROCR);
#library(plotROC)
#library(pROC)
#library(e1071)
#library(party)
#library(gbm)
#library(abind)

###### Functions to be used

### Assess prediction accuracy
pred_func <- function(ADD_test = ADD_test, risk_ADD = risk_ADD, spec_target = 0.8){
    pred_by_amyloid <- data.frame(Y3 = rep(NA, nrow(risk_ADD)), 
                                  Y5= rep(NA, nrow(risk_ADD)))
    for (i in 1:2){
        rocproj <- roc(ADD_test[, i], risk_ADD[, i])
        print(rocproj)
        idx = first(which(rocproj$specificities >= spec_target))
        print(c("risk score threshold=", rocproj$thresholds[idx]))
        pred_by_amyloid[, i] = as.numeric(risk_ADD[, i] > rocproj$thresholds[idx])
        conf_mat = confusionMatrix(factor(pred_by_amyloid[, i]), 
                                   factor(ADD_test[, i]), positive = "1")
        print(conf_mat)
    }
    return(pred_by_amyloid)
}


## Plot GLM-EN coefficient
# Plot EN coefficients
plot_GLMEN_coef <- function(fit = lm_tangles_map, title = NULL, 
                            pdf_file = "./Figures/beta_en_tangles.pdf", 
                            beta_abs_cut = 0.01, var_list_name,
                            width = 5, height = 6){
    
    coef <-  fit %>%  coefficients() %>% as.matrix()
    coef <-  coef[-c(1), ]
    coef <-  coef %>% as_tibble() %>%
        dplyr::mutate(Var = var_list_name[names(coef)]) %>%
        dplyr::filter(value != 0) %>% dplyr::arrange(value)  %>% 
        dplyr::filter(abs(value) > beta_abs_cut)%>% as.data.frame()
    colnames(coef) <- c("beta", "Var")
    
    coef %>% 
        ggplot(aes(x = Var, y = beta, fill=beta>0)) +
        geom_bar(stat = "identity", width = 0.5, show.legend = FALSE) +
        coord_flip() +
        scale_x_discrete(limits = coef$Var ) +
        labs(x = NULL, y = "beta", title = NULL) +
        theme(text = element_text(size = 20, face = "bold"),
              axis.text.x = element_text(angle = -90, vjust = -0.1),
              axis.text.y = element_text(size = 30, face = "bold"))
    
   # ggsave(pdf_file, width = width, height = height)
}

## Scatter plot of imputed path vs. measured 
plot_imputed_path <- function(path_measured, path_imputed, NIA, 
                              figpath = "./Figures/path_atdeath.pdf",
                              width = 8, height = 8){
    qplot(path_measured, path_imputed, colour = NIA, size = I(3)) +
        geom_abline(slope = 1, intercept = 0) +
        labs(x = "Measured", y = "Estimated", 
             title = NULL, color = "NIA Diagnosis") +
        theme(text = element_text(size = 30, face = "bold"),
              legend.position = "bottom", 
              axis.title = element_text(size = 30, face = "bold"),
              legend.title = element_text(size = 30, face="bold")) 
    ggsave(figpath, width = width, height = height)
}

plot_imputed_nia <- function(NIA, path_imputed,  
                             figpath = "./Figures/path_atdeath.pdf",
                             width = 8, height = 8){
    rocobj <- roc(NIA, path_imputed)
    pROC::ggroc(rocobj, colour = "red", size = 2) +
        # ggtitle("Predict NIA by imputed niareagansc") +
        geom_abline(slope = 1, intercept = 1, linetype = 2, size = 2) +
        annotate("text", x = .4, y = .4,
                 label = paste("AUC =", round(rocobj$auc, 3)), hjust = 0,
                 size = 10) +
        theme(text = element_text(size = 30, face = "bold"),
              legend.position = "bottom", 
              axis.title = element_text(size = 30, face = "bold"),
              legend.title = element_text(size = 30, face="bold"),
              plot.title = element_text(size = 30, face="bold")) 
    
        ggsave(figpath, width = width, height = height)
}

boxplot_imputed <- function(NIA, path_imputed, 
                            xlab = "Pathologic AD \n NIA Reagan at Death",
                            ylab = "Estimated Prob(NIA=1)",
                            figpath = "./Figures/boxplot_atdeath.pdf",
                            width = 8, height = 8){
    ggplot(data.frame(NIA = NIA, path_imputed = path_imputed), 
           aes(x = NIA, y = path_imputed, fill = NIA)) + geom_boxplot() +
        labs(x = xlab, y = ylab, title = NULL) +
        theme(text = element_text(size = 30, face = "bold"),
              axis.title = element_text(size = 30, face = "bold"),
              legend.title = element_text(size = 30, face="bold"),
              legend.position = c(0.5, 0.8),
              legend.direction = "horizontal",
              legend.background = element_rect(fill = "transparent")) +
        scale_fill_discrete(name = "NIA",
                            guide = guide_legend(label.theme = element_text(size = 30))) +
        guides(color=guide_legend(title="NIA", override.aes = list(size = 2)),size = FALSE)
    
    ggsave(figpath, width = width, height = height)
}

# Plot Cox coefficients
plot_cox_coef <- function(coef_mat, pdf_file = "Figures/beta_cox_ADD_nocog.pdf", 
                          add_legend = FALSE, width = 5, height = 6)
{
  
  coef <-  data.frame(Var = rownames(coef_mat), beta = coef_mat[, 1]) 
  #View(coef)
  coef <- coef[order(coef$beta), ]
  
  coef %>% ggplot(aes(x = Var, y = beta, fill=beta>0)) +
    geom_bar(stat = "identity", width = 0.5, show.legend = add_legend) +
    coord_flip() +
    scale_x_discrete(limits = (coef$Var) ) +
    labs(x = NULL, y = "beta", title = NULL) +
    theme(text = element_text(size = 20, face = "bold"),
          axis.text.x = element_text(angle = -90, vjust = -0.1, 
                                     face = "bold"))
  
  #ggsave(pdf_file, width = width, height = height)
}

## Fit Cox model
myCoxfit <- function(var_list_x, dt_surv_train = dt_surv_map) {
  var_remove = NULL
  cscfit_ADD <- CSC(as.formula(paste("Hist(time, ADD) ~ ", paste(var_list_x, collapse = "+"))), 
                    data=dt_surv_train)
  C_stat = summary(cscfit_ADD$models[[1]])$concordance
  coef_mat = summary(cscfit_ADD$models[[1]])$coefficients
  coef_mat = coef_mat[order(coef_mat[, 5], decreasing = TRUE), ]
  # coef_mat
  var_list_temp = rownames(coef_mat)[-1]
  #var_list_temp
  
  cscfit_ADD_temp <- CSC(as.formula(paste("Hist(time, ADD) ~ ", paste(var_list_temp, collapse = "+"))), data=dt_surv_train)
  C_stat_temp = summary(cscfit_ADD_temp$models[[1]])$concordance
  mod_comp = anova(cscfit_ADD$models[[1]], cscfit_ADD_temp$models[[1]])
  while( ( (mod_comp[2, 4] > 0.05) | (C_stat_temp[1] > C_stat[1]) ) & (length(var_list_temp) > 1)  ){
    var_list_x = var_list_temp
    cscfit_ADD <- cscfit_ADD_temp
    C_stat = C_stat_temp
    var_remove = c(var_remove, rownames(coef_mat)[1])
    coef_mat = summary(cscfit_ADD$models[[1]])$coefficients
    coef_mat = coef_mat[order(coef_mat[, 5], decreasing = TRUE), ]
    #coef_mat
    var_list_temp = rownames(coef_mat)[-1]
    #var_list_temp
    cscfit_ADD_temp <- CSC(as.formula(paste("Hist(time, ADD) ~ ", 
                                            paste(var_list_temp, collapse = "+"))), data=dt_surv_train)
    C_stat_temp = summary(cscfit_ADD_temp$models[[1]])$concordance
    # print(rbind(C_stat, C_stat_temp))
    mod_comp = anova(cscfit_ADD$models[[1]], cscfit_ADD_temp$models[[1]])
  }
  var_list_select = var_list_x
  ### Fit cox model with selected variables
  cscfit_select <- CSC(as.formula(paste("Hist(time, ADD) ~ ", paste(var_list_select, collapse = "+"))), 
                       data=dt_surv_train)
  print(summary(cscfit_select$models[[1]])$concordance)
  print(summary(cscfit_select$models[[1]])$coefficients)
  return(cscfit_select)
}

my_roc_test <- function(true_outcome = ADD_test,  pred_outcome_1, pred_outcome_2){
  # test risk_ADD_1 AUC > risk_ADD_2
  roc_1_Y1 = roc(true_outcome[, 1], pred_outcome_1[, 1]);  
  roc_1_Y3 = roc(true_outcome[, 2], pred_outcome_1[, 2]); 

  roc_2_Y1 = roc(true_outcome[, 1], pred_outcome_2[, 1]);  
  roc_2_Y3 = roc(true_outcome[, 2], pred_outcome_2[, 2]); 

  return(c(roc.test(roc_1_Y1, roc_2_Y1, alternative = "greater")$p.value,
           roc.test(roc_1_Y3, roc_2_Y3, alternative = "greater")$p.value))
}

##### Test with ROS samples
test_auc_ros <- function(true_outcome = ADD_test, pred_outcome = risk_ADD_full,
                         sens_target = 0.8,
                         file_name = "Figures/ROC_coxreg_ros_ADD_full.pdf"){
  
  rocproj_list <- conf_mat_list <- vector(mode = "list", 2)
  
  true_event = NULL
  risk_pred = NULL
  accuracy_vec <- rep(NA, 2)
  for(k in 1:2){
    true_event = c(true_event, true_outcome[, k])
    risk_pred = c(risk_pred, pred_outcome[, k])
    rocproj_list[[k]] <- roc(true_outcome[, k], pred_outcome[, k])
    
    idx = last(which(rocproj_list[[k]]$sensitivities>= sens_target))
    print(c("risk score threshold=", rocproj_list[[k]]$thresholds[idx]))
    conf_mat_list[[k]] = confusionMatrix(factor( as.numeric(pred_outcome[, k] > 
                                                              rocproj_list[[k]]$thresholds[idx]) ), 
                                         factor(true_outcome[, k]), positive = "1")
    accuracy_vec[k] = conf_mat_list[[k]]$overall["Accuracy"]
  }
  coxpred_data = data.frame(D = true_event, 
                            M = risk_pred, 
                            t = c(rep(3, nrow(dt_surv_ros)), 
                                  rep(5, nrow(dt_surv_ros)) )
  )
  ## Plot ROC curves
  basicplot = ggplot(coxpred_data, 
                     aes(x = D, y = M, color = as.factor(t), size = 1)) + 
    geom_roc(labels = FALSE) + style_roc()  +
    labs(x = "1 - Specificity", y = "Sensitivity", title = NULL) 
  View(calc_auc(basicplot))
    
  basicplot2 = basicplot+annotate("text", x = .35, y = c(0.45, 0.3), 
             label = paste(c("AUC Year 3 =", "AUC Year 5 ="), 
                           round(plotROC::calc_auc(basicplot)$AUC, 3) ), hjust = 0, 
             size = 9)
  print(basicplot2)
  #+
  #   guides(color=guide_legend(title="Year", override.aes = list(size = 2)), 
  #          size = FALSE) +
  #   theme(text = element_text(size = 30, face = "bold"),
  #         axis.text.x = element_text(angle = 30, hjust = 1, face = "bold"),
  #         legend.position=c(0.54, 0.2), legend.direction = "horizontal", 
  #         legend.background = element_rect(fill = "transparent"))
  # ggsave(file_name, width = 8, height = 7)

  
  return((list(rocproj_list = rocproj_list, 
               accuracy_vec = accuracy_vec, 
               conf_mat_list = conf_mat_list)))
}

#######3
##### Test with ROS samples
test_auc_ros_Y3_Y5 <- function(true_outcome = ADD_test, pred_outcome = risk_ADD_full,
                         sens_target = 0.8,
                         file_name = "Figures/ROC_coxreg_ros_ADD_full.pdf"){
  
  rocproj_list <- conf_mat_list <- vector(mode = "list", 3)
  
  true_event = NULL
  risk_pred = NULL
  accuracy_vec <- rep(NA, 2)
  for(k in 1:2){
    true_event = c(true_event, true_outcome[, k])
    risk_pred = c(risk_pred, pred_outcome[, k])
    rocproj_list[[k]] <- roc(true_outcome[, k], pred_outcome[, k])
    
    idx = last(which(rocproj_list[[k]]$sensitivities>= sens_target))
    conf_mat_list[[k]] = confusionMatrix(factor( as.numeric(pred_outcome[, k] > 
                                                              rocproj_list[[k]]$thresholds[idx]) ), 
                                         factor(true_outcome[, k]), positive = "1")
    accuracy_vec[k] = conf_mat_list[[k]]$overall["Accuracy"]
  }
  coxpred_data = data.frame(D = true_event, 
                            M = risk_pred, 
                            t = c( 
                                  rep(3, nrow(dt_surv_ros)), 
                                  rep(5, nrow(dt_surv_ros)) )
  )
  ## Plot ROC curves
  basicplot = ggplot(coxpred_data, 
                     aes(x = D, y = M, color = as.factor(t), size = 1)) + 
    geom_roc(labels = FALSE) #+ style_roc() 
  basicplot +
    annotate("text", x = .35, y = c(0.45, 0.3), 
             label = paste(c(
                             "AUC Year 3 =", "AUC Year 5 ="), 
                           round(calc_auc(basicplot)$AUC, 3) ), hjust = 0, 
             size = 9) +
    guides(color=guide_legend(title="Year", override.aes = list(size = 2)), 
           size = FALSE) +
    theme(text = element_text(size = 30, face = "bold"),
          axis.text.x = element_text(angle = 30, hjust = 1),
          legend.position=c(0.54, 0.2), legend.direction = "horizontal", 
          legend.background = element_rect(fill = "transparent"))
  ggsave(file_name, width = 8, height = 7)
  
  return((list(rocproj_list = rocproj_list, 
               accuracy_vec = accuracy_vec, 
               conf_mat_list = conf_mat_list)))
}



## My Cox_regression function with variable selection by anova
myCOX <- function(surv_data_train, var_list_x){

  set.seed(2021)
  trainIndex_folds = groupKFold(surv_data_train$projid, k = cv_fold)
  auc_vec <- rep(NA, length(pval_vec))

  cscfit_full <- CSC(as.formula(paste("Hist(time, ADD) ~ ",
                      paste(var_list_x, collapse = "+"))), data=surv_data_train)
  coef_mat <- summary(coxfit)$coefficients
  var_pval <- coef_mat[, 5]

  for(i in 1:tune_length) {
    Test_AD = NULL
    Test_pred = NULL
    for(k in 1:cv_fold){
      # i= k = 1
      temp_train = data_train[trainIndex_folds[[k]], ]
      temp_test = data_train[-trainIndex_folds[[k]], ]
      Test_AD = c(Test_AD, temp_test[, eval(var_y)])

      var_list_x_select = row.names(coef_mat)[var_pval < pval_vec[i]]
      # print(var_list_x_select)
      fm1 = paste("Surv(time, event!=0) ~ ", paste(var_list_x_select, collapse = "+"))
      fit_temp = coxph(as.formula(fm1), data=temp_train, x = TRUE)
      pred_temp <- 100*predictRisk(coxfit_select, newdata=temp_test, times=t_vec[i], cause=1)
      Test_pred = rbind(Test_pred, pred_temp)
    }
      auc_vec[i] = roc(test_AD, as.vector(Test_pred))$auc
  }

  best_pval_threshold = pval_vec[which.max(auc_vec)]
  print(c(pval = best_pval_threshold, best_auc = max(auc_vec)))

  var_list_x_select = row.names(coef_mat)[var_pval < best_pval_threshold]
  # print(var_list_x_select)
      fm1 = paste("Surv(time, event!=0) ~ ", paste(var_list_x_select, collapse = "+"))
      fit_select = coxph(as.formula(fm1), data=temp_train, x = TRUE)

  return(list(best_pval_threshold = best_pval_threshold,
              best_auc = max(auc_vec), best_fit = fit_select) )
}


### Calculate missing rate per column
getMissingRate <- function(dt){
  return(apply(dt, 2, function(x){sum(is.na(x)) / nrow(dt)} ))
}

### get AUC functions
getAUC <- function(D, pred){
  return( roc(D, pred, quiet = TRUE)$auc )
}

getMulticlassAUC <- function(D, pred){
  return( multiclass.roc(D, pred, quiet = TRUE)$auc )
}

## My EN_CV function
# var_list = colnames(dataset_fit_lm_noMCI_sub4)
myENcv <- function(data_train, var_list_x, var_y, family = "binomial", cv_fold = 10,
                   tune_length = 10){
  data_train0 = data_train %>% dplyr::select(c("projid", eval(var_y), eval(var_list_x)))
  data_train0[, eval(var_list_x)] <- scale(data_train0[, eval(var_list_x)])

  set.seed(2019)
  trainIndex_folds = groupKFold(data_train$projid, k = cv_fold)
  alpha_vec = seq(0, 1, length.out = tune_length)
  lambda_vec = exp( seq(-3, 6, length.out = 100) )
  best_lambda_vec <- best_auc <- rep(NA, tune_length)

  for(i in 1:tune_length) {
    Test_AD = NULL
    Test_pred = NULL
    for(k in 1:cv_fold){
      # i= k = 1
      temp_train = data_train0[trainIndex_folds[[k]], ]
      temp_test = data_train0[-trainIndex_folds[[k]], ]
      Test_AD = c(Test_AD, temp_test[, eval(var_y)])

      fit_temp = glmnet(x = as.matrix(temp_train[, eval(var_list_x)]), y = temp_train[, eval(var_y)],
                        family = family, alpha = alpha_vec[i], lambda = lambda_vec)
      pred_temp = predict(fit_temp, newx = as.matrix(temp_test[, eval(var_list_x)]) )
      if(family == "binomial"){
        Test_pred = rbind(Test_pred, pred_temp)
      }else{
        Test_pred = abind(Test_pred, pred_temp, along = 1)
      }
    }
    if(family == "binomial"){
      AUC_temp = apply(Test_pred, 2, getAUC, D = Test_AD)
    }else{
      # multiclass.roc( D, Test_pred[, , 1])
      D = factor(Test_AD); levels(D) = c("NCI", "MCI", "ADD")
      AUC_temp = apply(Test_pred, 3, getMulticlassAUC, D = D )
    }
    best_lambda_vec[i] = fit_temp$lambda[which.max(AUC_temp)]
    best_auc[i] = max(AUC_temp)
  }
  best_alpha = alpha_vec[which.max(best_auc)]
  best_lambda = best_lambda_vec[which.max(best_auc)]
  # print(c(best_alpha, best_lambda, max(best_auc)))

  best_fit = glmnet(x = as.matrix(data_train0[, eval(var_list_x)]),
                    y = data_train0[, eval(var_y)],
                    family = family, alpha = best_alpha, lambda = best_lambda)
  # print(best_fit$beta)
  return(list(best_alpha = best_alpha, best_lambda = best_lambda,
              best_auc = max(best_auc), best_fit = best_fit) )
}

# Plot EN coefficients
plot_ENcoef <- function(fit = fit1$best_fit, title = "Model 4", family = "multinomial",
                        pdf_file = "Figures/beta_en_model4.pdf", add_legend = TRUE,
                        beta_abs_cut = 0.01, var_list_name,
                        width = 5, height = 6){
  if(family == "binomial"){
    coef <-  fit %>%  coefficients() %>% as.matrix()
    coef <-  coef[-1, ]
    coef <- coef %>% as_tibble() %>%
      dplyr::mutate(Var = var_list_name[names(coef)], beta = coef) %>%
      dplyr::filter(beta != 0) %>% dplyr::arrange(beta) %>% as.data.frame()
    coef %>% ggplot(aes(x = Var, y = beta, fill=beta>0)) +
      geom_bar(stat = "identity", width = 0.5, show.legend = add_legend) +
      coord_flip() +
      scale_x_discrete(limits = (coef$Var) ) +
      labs(x = NULL, y = "beta", title = title) +
      theme(text = element_text(size = 20),
            axis.text.x = element_text(angle = -90, vjust = -0.1))
  }else{
    coef <-  fit %>%  coefficients() %>% lapply(as.matrix)
    coef <- data.frame(Var = c("Intercept", var_list_name[row.names(coef[[1]])[-1]]),
                       beta2 = coef[[2]], beta3 = coef[[3]])
    colnames(coef) = c("Var", "Incident MCI", "Incident ADD")
    coef = coef[-1, ]
      # %>% arrange(beta) %>% as.data.frame()
    coef_plot = melt(coef) %>% filter(abs(value) > beta_abs_cut)
    colnames(coef_plot) = c("Var", "variable", "beta")
    coef_plot%>%
      ggplot(aes(x = Var, y = beta, fill=beta>0)) +
      geom_bar(stat = "identity", width = 0.5, show.legend = add_legend) +
      coord_flip() +
      facet_grid(cols = vars(variable)) +
      scale_x_discrete(limits = unique(coef_plot$Var) ) +
      labs(x = NULL, y = NULL, title = title) +
      theme(text = element_text(size = 20),
            axis.text.x = element_text(angle = -90, vjust = -0.1))
  }
  ggsave(pdf_file, width = width, height = height)
}



## My GBM_CV function
myGBMcv <- function(data_train, var_list, family = "bernoulli", cv_fold = 10,
                   tune_length = 3){
  #var_list <- colnames(dataset_fit_lm_noMCI_sub3)
  data_train0 = data_train %>% dplyr::select(eval(var_list)) %>% dplyr::select(-study)
  if(family == "bernoulli"){
    data_train0$AD <- as.numeric(data_train0$AD) - 1
  }

  var_list_x = var_list[!var_list %in% c("AD", "study", "projid")] # Exclude AD from var_list
  data_train0[, eval(var_list_x)] <- scale(data_train0[, eval(var_list_x)])

  set.seed(2019)
  trainIndex_folds = groupKFold(data_train$projid, k = cv_fold)
  interaction_depth_vec = 1:tune_length
  auc_vec <- best_iter <- best_interaction.depth <- rep(NA, tune_length)

  for(i in 1:tune_length) {
    Test_AD = NULL
    Test_pred = NULL
    for(k in 1:cv_fold){
      temp_train = data_train0[trainIndex_folds[[k]], ] %>% as.data.frame()
      temp_test = data_train0[-trainIndex_folds[[k]], ] %>% as.data.frame()
      Test_AD = c(Test_AD, temp_test$AD)

      fit_temp = gbm(AD ~ ., data = temp_train, distribution = family,
                     n.minobsinnode = 20, shrinkage = 0.1, n.trees = 100,
                     interaction.depth = interaction_depth_vec[i],
                     cv.folds = 5, train.fraction = 0.8, bag.fraction = 0.5,
                     keep.data = TRUE, verbose = FALSE, n.cores = 1)
      best.iter <- gbm.perf(fit_temp, method = "cv", plot.it = FALSE)
      pred_temp <- predict(fit_temp, newdata = temp_test, n.trees = best.iter,
                           type = "response")
      if(family == "bernoulli"){
        Test_pred = c(Test_pred, as.vector(pred_temp))
      }else{
        Test_pred = abind(Test_pred, pred_temp, along = 1)
      }
    }

    if(family == "bernoulli"){
      auc_vec[i] = getAUC(Test_AD, Test_pred)
    }else{
      # multiclass.roc( D, Test_pred[, , 1])
      D = factor(Test_AD); levels(D) = c("NCI", "MCI", "AD")
      auc_vec[i] = getMulticlassAUC(D, Test_pred[, , 1])
    }
  }
  best_interaction_depth = interaction_depth_vec[which.max(auc_vec)]
  # print(c(best_interaction_depth, max(auc_vec)))

  best_fit = gbm(AD ~ ., data = data_train0, distribution = family,
                 n.minobsinnode = 20, shrinkage = 0.1, n.trees = 100,
                 interaction.depth = best_interaction_depth,
                 cv.folds = 5, train.fraction = 0.8, bag.fraction = 0.5,
                 keep.data = TRUE, verbose = FALSE, n.cores = 1)
  best.iter <- gbm.perf(best_fit, method = "cv", plot.it = FALSE)

  return(list(best_iter = best.iter, best_interaction_depth = best_interaction_depth,
              best_auc = max(auc_vec), best_fit = best_fit) )
}


## My RF_CV function
myRFcv <- function(data_train, var_list, family = "bernoulli", cv_fold = 10,
                   tune_length = 10){
  #var_list <- colnames(dataset_fit_lm_noMCI_sub3)
  data_train0 = data_train %>% select(eval(var_list)) %>% select(-study)
  var_list_x = var_list[!var_list %in% c("AD", "study")] # Exclude AD from var_list
  data_train0[, eval(var_list_x)] <- scale(data_train0[, eval(var_list_x)])

  set.seed(2019)
  m_var = length(var_list_x)
  tune_length = min(tune_length, ceiling(sqrt(m_var)) ) # adjust to total variable number
  trainIndex_folds = groupKFold(data_train$projid, k = cv_fold)
  mtry_vec <- ceiling(seq(1, ceiling(sqrt(m_var)), length.out = tune_length))
  auc_vec <- rep(NA, tune_length)

  for(i in 1:tune_length) {
    Test_AD = NULL
    Test_pred = NULL
    for(k in 1:cv_fold){
      temp_train = data_train0[trainIndex_folds[[k]], ] %>% as.data.frame()
      temp_test = data_train0[-trainIndex_folds[[k]], ] %>% as.data.frame()
      Test_AD = c(Test_AD, temp_test$AD)

      fit_temp = cforest(AD ~ ., data = temp_train,
                         control = cforest_control(mtry = mtry_vec[i], ntree = 500))
      if(family == "bernoulli"){
        pred_temp = unlist ( predict(fit_temp, newdata = temp_test, type = "prob") )
        pred_temp = pred_temp[seq(2, length(pred_temp), by = 2)]
        Test_pred = c(Test_pred, pred_temp)
      }else{
        pred_temp = unlist ( predict(fit_temp, newdata = temp_test, type = "prob") )
        pred_temp = matrix(pred_temp, ncol = 3, byrow = TRUE)
        Test_pred = rbind(Test_pred, pred_temp)
      }
    }
    if(family == "bernoulli"){
      auc_vec[i] = getAUC(Test_AD, Test_pred)
    }else{
      # multiclass.roc( D, Test_pred[, , 1])
      D = factor(Test_AD); levels(D) = c("NCI", "MCI", "AD")
      colnames(Test_pred) = c("NCI", "MCI", "AD")
      auc_vec[i] = getMulticlassAUC(D, Test_pred)
    }
  }
  best_mtry = mtry_vec[which.max(auc_vec)]
  # print(c(best_mtry, max(auc_vec)))

  best_fit = cforest(AD ~ ., data = data_train0,
                     control = cforest_control(mtry = best_mtry, ntree = 500))
  return(list(best_mtry = best_mtry,
              best_auc = max(auc_vec), best_fit = best_fit) )
}

