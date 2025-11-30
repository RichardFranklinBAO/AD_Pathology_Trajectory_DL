###### Remove na records
my.max <- function(x){max(x, na.rm = TRUE)}
my.min <- function(x){min(x, na.rm = TRUE)}
my.sum <- function(x){sum(x, na.rm = TRUE)}
my.mean <- function(x){mean(x, na.rm = TRUE)}

##### Function for inverse normalization
my.invnorm = function(x)
{
  n_na = sum(is.na(x))
  if(n_na > 0){
    print(paste("There are ", n_na, " NAs, which will be removed from output."))
  }
  
  res = rank(x, na.last = NA) # remove NAs
  res = qnorm(res/(length(res)+0.5))
  return(res)
}

######## Function for qq plot with -log10 pvalues
myqq <- function(pvector, title="Quantile-Quantile Plot of -log10(P-values)", add=FALSE, colour = "blue", lty = 1) {
    o = -log10(sort(pvector,decreasing=F))
    e = -log10( 1:length(o)/length(o) )
    if(!add) {
        plot(e, o, type = "l", main = title, xlim=c(0,max(e)), ylim=c(0,max(o[o<Inf])), col = colour, lty = lty, lwd = 2, xlab = "expectation", ylab = "observation")
        abline(0, 1)
    }else{
        points(e, o, type = "l", col = colour, lty = lty, lwd = 2)
    }
}

#### Function to run linear mixed mdoel with model input "mod" and data input "dt" in data frame ####
my_lmer <- function(mod, dt){
	# mod: model formula in string
	# dt: data frame containing longitudinal data (non-longitudinal covariates will have the same value per sample across all time points)
  fm_temp <- lmer(as.formula(mod), dt) # run Linear Mixed Model
  fixed_coeff = summary(fm_temp)$coefficients # get coefficient matrix
  k = dim(fixed_coeff)[1] # get the index of the last covariate
  # Test if the kth interested covariate associated with the response variable (by Chisquare Test)
  chisq_test = (fixed_coeff[k, 1] / fixed_coeff[k, 2] )^2
  beta = fixed_coeff[k, 1] # effect size
  beta_se = fixed_coeff[k, 2] # effect size standard error
  pval = pchisq(chisq_test, 1, 0, lower.tail = FALSE) # obtain p-value
  return(list(beta = beta, beta_se = beta_se, pval = pval, model = fm_temp)) # output
}

#### Function to generate PED files for given phenotype
Create_PED_Basic <- function(Phe_basic, GWAS_sample_fam, Illumina_sample_fam, WGS_PCs,
                            GWAS_PCs, WGS_SampleID_qc, ROSMAP_WGS_IDs, pheno="amyloid", 
                            out_dir = "/Users/jingjingyang/Dropbox (Yang@Emory)/Research/WingoLab/ROSMAP/PED_Files/PED_Files/", 
                            SQRT = TRUE, InverseNormal = FALSE)
  {
  # pheno = "tangles"
  rownames(GWAS_PCs) = GWAS_PCs$IID_ID
  Phe_basic_sub = subset(Phe_basic, select = 
                           c("projid", "study", "msex", "gpath", "amyloid", 
                             "cogdx", "cogng_random_slope", "braaksc", "ceradsc",
                             "tangles", "gwas_id", "wgs_id", "smoking", "age_death", "race"))

  # Obtain the corresponding projid for WGS samples passed QC
  ROSMAP_WGS_IDs_qc = ROSMAP_WGS_IDs[ROSMAP_WGS_IDs$WGS_id %in% WGS_SampleID_qc$V1, ]
  rownames(ROSMAP_WGS_IDs_qc) = ROSMAP_WGS_IDs_qc$projid
  Phe_basic_wgs = Phe_basic_sub[Phe_basic_sub$projid %in% ROSMAP_WGS_IDs_qc$projid, ]
  
  ######## Set up ped data for WGS samples
  Phe_basic_wgs = Phe_basic_wgs[!is.na(Phe_basic_wgs[, pheno]), ]
  IND_ID = ROSMAP_WGS_IDs_qc[as.character(Phe_basic_wgs$projid), "WGS_id"]
  n_sample = length(IND_ID)
  
  SEX = Phe_basic_wgs$msex
  SEX[SEX == 0] = 2
  
  # Set ROS study index as 0
  STUDY = Phe_basic_wgs$study
  STUDY[STUDY=="MAP"] = 1
  STUDY[STUDY=="ROS"] = 0
  
  # Take the square root of gpath
  if(SQRT){
    PHENO = sqrt(Phe_basic_wgs[, pheno])
  }else if (InverseNormal){
    PHENO = my.invnorm(Phe_basic_wgs[, pheno])
  }else{
    PHENO = Phe_basic_wgs[, pheno]
  }
  pheno_ped_wgs = data.frame(FAM_ID = IND_ID, 
                         IND_ID = IND_ID, 
                         FAT_ID = rep(NA, n_sample), MOT_ID = rep(NA, n_sample), 
                         SEX = SEX, PHENO = PHENO, 
                         WGS_PCs[IND_ID, 2:4], 
                         AGE = Phe_basic_wgs$age_death, 
                         SMOKING = Phe_basic_wgs$smoking, 
                         STUDY = STUDY)
  pheno_ped_wgs = pheno_ped_wgs[ complete.cases(pheno_ped_wgs[, -(1:4)]),  ]
  print( dim(pheno_ped_wgs) )
  file_wgs = paste(out_dir, pheno, "_wgs.ped", sep = "")
  write.table(pheno_ped_wgs, file = file_wgs, sep = "\t", 
              quote = FALSE, row.names = FALSE, na = "X")
  
  ######## Set up ped data for GWAS samples
  # Subset for GWAS samples
  Phe_basic_gwas = Phe_basic_sub[Phe_basic_sub$gwas_id %in% GWAS_sample_fam$IND_ID, ]
  Phe_basic_gwas = Phe_basic_gwas[(Phe_basic_gwas$race == 1) & (!is.na(Phe_basic_gwas[, pheno] )) , ]
  
  IND_ID = Phe_basic_gwas$gwas_id
  n_sample = length(IND_ID)
  
  SEX = Phe_basic_gwas$msex
  SEX[SEX == 0] = 2
  
  # Set ROS study index as 0
  STUDY = Phe_basic_gwas$study
  STUDY[STUDY=="MAP"] = 1
  STUDY[STUDY=="ROS"] = 0
  
  # Set illumina sample index as 0
  Array_Index = rep(1, n_sample)
  Array_Index[Phe_basic_gwas$gwas_id %in% Illumina_sample_fam$IND_ID] = 0
  table(Array_Index)
  
  # Take the square root of gpath
  if(SQRT){
    PHENO = sqrt(Phe_basic_gwas[, pheno])
  }else if (InverseNormal){
    PHENO = my.invnorm(Phe_basic_gwas[, pheno])
  }else{
    PHENO = Phe_basic_gwas[, pheno]
  }
  pheno_ped = data.frame(FAM_ID = IND_ID, 
                              IND_ID = IND_ID, 
                              FAT_ID = rep(NA, n_sample), MOT_ID = rep(NA, n_sample), 
                              SEX = SEX, PHENO = PHENO, 
                              GWAS_PCs[IND_ID, 4:6], 
                              AGE = Phe_basic_gwas$age_death, 
                              SMOKING = Phe_basic_gwas$smoking, 
                              STUDY = STUDY, Array_Index = Array_Index)
  pheno_ped = pheno_ped[ complete.cases(pheno_ped[, -(1:4)]),  ]
  print(dim(pheno_ped))
  file1 = paste(out_dir, pheno, "_gwas.ped", sep = "")
  write.table(pheno_ped, file = file1, sep = "\t", 
              quote = FALSE, row.names = FALSE, na = "X")

  # Subset samples that only have GWAS data but not WGS data
  pheno_ped_gwas_only = pheno_ped[! (pheno_ped$IND_ID %in%  Phe_basic_wgs$gwas_id),  ]
  print(dim(pheno_ped_gwas_only))
  file2 = paste(out_dir, pheno, "_gwas_only.ped", sep = "")
  write.table(pheno_ped_gwas_only, file = file2, 
              sep = "\t", quote = FALSE, row.names = FALSE, na = "X")
  return()
}

