# ==============================================================================
# Project: AD Pathology Trajectory Prediction (Reproducible Workflow)
# ==============================================================================

# --- Executables ---
PYTHON = python3
RSCRIPT = Rscript
JUPYTER = jupyter nbconvert --to notebook --execute --inplace

# --- Directories ---
SCRIPTS_DIR = scripts
RESULTS_DIR = results
CONFIG_FILE = configs/best_models.py

# ==============================================================================
# 1. Define Script Paths
# ==============================================================================

# S1: Data Prep
S1_1_NB   = $(SCRIPTS_DIR)/S1_Data_Preparation/S1.1_all_decedents/data_prep.ipynb
S1_2_RMD  = $(SCRIPTS_DIR)/S1_Data_Preparation/S1.2_all_livings/Checking_data_and_generating_all_livings_participants.Rmd

# S2: GLM Baseline
S2_NB     = $(SCRIPTS_DIR)/S2_GLM/NIAvalueUpdatedVersion_elastic_net.ipynb

# S3: Decedents Prediction (S3.1 Skipped, Jump to S3.2)
S3_2_AMYLOID = $(SCRIPTS_DIR)/S3_Training_ML_Models_and_Predicting_Pathologies_only_for_Decedents/S3.2_Predicting_Pathologies_only_for_Decedents/BiLSTM_amyloid_imputing_decedents_pathologies.py
S3_2_NIA     = $(SCRIPTS_DIR)/S3_Training_ML_Models_and_Predicting_Pathologies_only_for_Decedents/S3.2_Predicting_Pathologies_only_for_Decedents/BiLSTM_nia_imputing_decedents_pathologies.py
S3_2_TANGLES = $(SCRIPTS_DIR)/S3_Training_ML_Models_and_Predicting_Pathologies_only_for_Decedents/S3.2_Predicting_Pathologies_only_for_Decedents/LSTM_tangles_imputing_decedents_pathologies.py
S3_2_GPATH   = $(SCRIPTS_DIR)/S3_Training_ML_Models_and_Predicting_Pathologies_only_for_Decedents/S3.2_Predicting_Pathologies_only_for_Decedents/LSTMReLU_gpath_imputing_decedents_pathologies.py
S3_COMBINE   = $(SCRIPTS_DIR)/S3_Training_ML_Models_and_Predicting_Pathologies_only_for_Decedents/S3.2_Predicting_Pathologies_only_for_Decedents/Combining_clean_imputed_dataset_for_all_decedentstled.Rmd

# S4: Livings Prediction
S4_1_NB   = $(SCRIPTS_DIR)/S4_Training_ML_Models_and_Predicting_Pathologies_only_for_Livings/S4.1_Retraining_using_all_decedents_samples_for_imputing_pathologies_for_livings/lstm_final_model.ipynb
S4_2_NB   = $(SCRIPTS_DIR)/S4_Training_ML_Models_and_Predicting_Pathologies_only_for_Livings/S4.2_ImputingLongitudinalValuesOnlyForLivings/prediction_WithModifiedNIA_Only_For_Livings.ipynb

# S5: Plotting & Clustering
S5_1_RMD   = $(SCRIPTS_DIR)/S5_Plotting/S5.1_Combining_Clean_decedents_Imputation_and_Livings_Imputation/Combining_for_final_imputation_including_both_decedents_and_livings.Rmd
S5_2_1_RMD = $(SCRIPTS_DIR)/S5_Plotting/S5.2_ploting_the_trajectories_of_pathologies/S5.2.1_PADPPlottingwithADDas0_ModifiedNIAValue.Rmd
S5_2_2_RMD = $(SCRIPTS_DIR)/S5_Plotting/S5.2_ploting_the_trajectories_of_pathologies/S5.2.2_CleanedVersion.Rmd

# ==============================================================================
# 2. Phony Targets (Commands)
# ==============================================================================

.PHONY: all check_config s1_prep s2_glm s3_predict s4_livings s5_plots clean install dummy_data

# --- Environment Setup Targets ---
install:
	@echo " Installing Python dependencies..."
	pip install -r requirements.txt
	@echo " Installing R dependencies via renv..."
	$(RSCRIPT) -e "renv::restore(prompt = FALSE)"

dummy_data:
	@echo " Generating synthetic data..."
	python scripts/generate_dummy.py


all: check_config s1_prep s2_glm s3_predict s4_livings s5_plots
	@echo "======================================================="
	@echo " Full Pipeline Completed Successfully!"
	@echo "    Outputs are in: $(RESULTS_DIR)/"
	@echo "======================================================="

# --- Step 0: Check for Pre-computed Configs ---
check_config:
	@if [ ! -f "$(CONFIG_FILE)" ]; then \
		echo " Error: $(CONFIG_FILE) not found!"; \
		echo "   S3.1 (Model Training) was run on a cluster. Put best params into $(CONFIG_FILE)."; \
		exit 1; \
	else \
		echo " Found best model configurations. Skipping S3.1 training..."; \
	fi

# --- Step 1: Data Preparation ---
s1_prep:
	@echo " [S1] Running Data Preparation..."
	$(JUPYTER) "$(S1_1_NB)"
	$(RSCRIPT) -e "rmarkdown::render('$(S1_2_RMD)', output_dir='$(RESULTS_DIR)/S1/S1.2')"
	@echo " [S1] Done."

# --- Step 2: Baseline GLM ---
s2_glm: s1_prep
	@echo " [S2] Running GLM Baseline..."
	$(JUPYTER) "$(S2_NB)"
	@echo " [S2] Done."

# --- Step 3: Predict Decedents (Skipping Training) ---
s3_predict: check_config s1_prep
	@echo " [S3.2] Running Predictions for Decedents..."
	$(PYTHON) "$(S3_2_AMYLOID)"
	$(PYTHON) "$(S3_2_NIA)"
	$(PYTHON) "$(S3_2_TANGLES)"
	$(PYTHON) "$(S3_2_GPATH)"
	$(RSCRIPT) -e "rmarkdown::render('$(S3_COMBINE)', output_dir='$(RESULTS_DIR)/S3/S3.2')"
	@echo " [S3.2] Done."

# --- Step 4: Predict Livings ---
s4_livings: s3_predict
	@echo " [S4] Retraining Final Models & Predicting for Livings..."
	$(JUPYTER) "$(S4_1_NB)"
	$(JUPYTER) "$(S4_2_NB)"
	@echo " [S4] Done."

# --- Step 5: Visualization & Clustering ---
s5_plots: s4_livings
	@echo " [S5] Generating Plots and Clusters..."
	$(RSCRIPT) -e "rmarkdown::render('$(S5_1_RMD)', output_dir='$(RESULTS_DIR)/S5/S5.1')"
	$(RSCRIPT) -e "rmarkdown::render('$(S5_2_1_RMD)', output_dir='$(RESULTS_DIR)/S5/S5.2')"
	$(RSCRIPT) -e "rmarkdown::render('$(S5_2_2_RMD)', output_dir='$(RESULTS_DIR)/S5/S5.2')"
	@echo " [S5] Done."


# --- Clean Up ---
clean:
	@echo "Cleaning all generated results..."
	@rm -rf $(RESULTS_DIR)/*
	@echo "All results removed from $(RESULTS_DIR)/"
