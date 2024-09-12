#Set working directory ----
setwd("/mnt/work/workbench/fcfiguei/miRNA_periphery_scripts")

#----FULL SCRIPT BELOW ----
#Load personalized functions ----
source("Filipe_functions.R")

#Load tidyverse ----
library(tidyverse)
library(conflicted)
conflict_prefer("filter", "dplyr")
conflict_prefer("lag", "dplyr")
conflict_prefer("select", "dplyr")
conflict_prefer("rename", "dplyr")

library(lubridate)

#--- SECTION 1: DATA LOADING ----
#Load Elisabeth's files ----
library(readxl)
library(multidplyr)

cluster <- new_cluster(15)
cluster_library(cluster, "dplyr")

samples_raw <- read_excel(path="20230209_Thora_AllSamples.xlsx", sheet = 1, range = "A1:AG2652", col_names = TRUE, na=c("", "#NULL!", "NA")) %>%
  clean_names()

samples <- samples_raw %>%
  mutate(across(where(is.character), ~ enc2utf8(.x))) %>%
  filter(!is.na(dato_inn)) %>%
  mutate(across(where(lubridate::is.POSIXt), ~ lubridate::ymd(.x))) %>%
  rowwise() %>%
  partition(cluster) %>%
  mutate(pasid_corr=if_else(senter == "Lund/Skåne" & pasid == "DK 02006", "SE 02006", pasid),
         senter_corr=if_else(grepl("St. Olavs Hospital", senter) & pasid == "NO 01010", "Namsos", senter),
         visit_corr=if_else(grepl("strålestart\\, ikke inklusjon$", kommentar_1), "STR START", visit),
         visit_nr_corr=if_else(grepl("strålestart\\, ikke inklusjon$", kommentar_1), "01", visit_nr),
         provedato_corr=if_else(grepl("pasientperm \\= 28\\.08\\.2017$", kommentar_1), lubridate::ymd("2017-08-28"), provedato),
         provedato_corr=if_else(grepl("pasientperm \\= 08\\.06\\.2018$", kommentar_1), lubridate::ymd("2018-06-08"), provedato_corr),
         provedato_corr=if_else(grepl("1 ÅRS KTR$", visit) & pasid == "SE 04001", lubridate::ymd("2018-03-28"), provedato_corr),
         pasid_diff=if_else(pasid==pasid_corr, "0", "1"),
         senter_diff=if_else(senter==senter_corr, "0", "1"),
         visit_diff=if_else(visit==visit_corr, "0", "1"),
         visit_nr_diff=if_else(visit_nr==visit_nr_corr, "0", "1"),
         provedato_diff=if_else(provedato==provedato_corr, "0", "1")) %>%
  collect()

samples <- samples %>%
  select(-pasid, -senter, -visit, -visit_nr, -provedato,
         -pasid_diff, -senter_diff, -visit_diff, -visit_nr_diff, -provedato_diff) %>%
  rename(pasid=pasid_corr, senter=senter_corr, visit=visit_corr, visit_nr=visit_nr_corr, provedato=provedato_corr) %>%
  mutate(visit_corr=factor(paste0(visit_nr, " - ",sub("^ +", "", visit)),
                           levels=c("00 - INKLUSJON", "01 - STR START",
                                    "02 - 12 UKER EVAL", "03 - 32 UKER EVAL",
                                    "04 - 1 ÅRS KTR", "05 - 2 ÅRS KTR",
                                    "06 - PROGRESJON", "99 - Utenom Visitt"))) %>%
  mutate(blodfraksjon=factor(case_when(
    str_detect(provetype, "EDTA") ~ "EDTA",
    str_detect(provetype, "PAX") ~ "PAX",
    str_detect(provetype, "P(LASMA|lasma)") ~ "Plasma",
    str_detect(provetype, "S(ERUM|erum)") ~ "Serum",
    TRUE ~ "Other"
  )))

samples_serum <- samples %>%
  filter(blodfraksjon=="Serum") %>%
  droplevels() %>%
  select(senter, pasid, visit_corr, provedato) %>%
  distinct()

samples_serum <- samples_serum %>%
  separate(pasid, sep = " ", into = c("country", NA), remove = FALSE) %>%
  group_by(country, senter, pasid) %>%
  count() %>%
  group_by(country) %>%
  mutate(an_id_num=row_number()) %>%
  ungroup() %>%
  mutate(id_corr=paste0(country, an_id_num)) %>%
  select(-n, -an_id_num) %>%
  right_join(samples_serum, multiple = "all")

#Create key with match between all IDs ----

idtable <- read_excel(path="Kobling Thora ID og biomarkørID_corr.xlsx", sheet = 1, range = "A1:G178", col_names = TRUE, na=c("", "NA")) %>%
  clean_names() %>%
  mutate(across(where(is.character), ~ enc2utf8(.x))) %>%
  mutate(across(where(is.POSIXt), ~ ymd(.x)))

duplicates_idtable <- idtable %>%
  filter(!is.na(id_i_database)) %>%
  group_by(id_i_database) %>%
  count %>%
  filter(n>1) %>%
  select(id_i_database) %>%
  pull

duplicates_samples_serum <- samples_serum %>%
  group_by(pasid, id_corr) %>%
  count %>%
  group_by(pasid) %>%
  count() %>%
  filter(n>1) %>%
  select(pasid) %>%
  pull

key <- bind_rows(idtable %>%
                   filter(id_i_database %nin% duplicates_idtable,
                          !is.na(id_i_database)) %>%
                   select(pasient_id, id_i_database) %>%
                   left_join(samples_serum %>%
                               select(id_i_database=pasid, id_corr) %>%
                               distinct_all()),
                 idtable %>%
                   filter(id_i_database %in% duplicates_idtable,
                          !is.na(id_i_database)) %>%
                   select(pasient_id, id_i_database, inclusion_site) %>%
                   left_join(samples_serum %>%
                               select(inclusion_site=senter, id_i_database=pasid, id_corr) %>%
                               distinct_all()) %>%
                   select(-inclusion_site)) %>%
  select(studieid=pasient_id, id_corr) %>%
  filter(!is.na(id_corr)) %>%
  left_join(samples_serum %>%
              select(country, inclusion_site=senter, id_i_database=pasid, id_corr) %>%
              distinct_all())

error_table <- read_excel(path="Check_visit_order_clin.xlsx", sheet = 9, range = "A1:C178", col_names = TRUE, na=c("", "#NULL!", "NA")) %>%
  clean_names() %>%
  rename(studieid=id)

key_full <- samples %>%
  filter(blodfraksjon=="Serum") %>%
  droplevels() %>%
  select(id_i_database=pasid, inclusion_site=senter, strekkode, provedato, visit_corr) %>%
  left_join(key) %>%
  left_join(error_table, relationship = "many-to-many") %>%
  rowwise() %>%
  mutate(pasnr=as.integer(str_extract(studieid, "(\\d)+"))) %>%
  ungroup()

#Load clinical data ----
library(haven)

thora_clinical_data <- read_sav(file="THORAMAY2024.sav")

thora_clinical_data_labels <- lapply(thora_clinical_data, attr, "label")

thora_clinical_data_labels <- as.data.frame(do.call(rbind, thora_clinical_data_labels)) %>%
  clean_names() %>%
  select(1) %>%
  rownames_to_column() %>%
  mutate(column_clean=clean_names(rowname))

thora_clinical_data_columns <- as.data.frame(colnames(thora_clinical_data)) %>%
  rename(column=1) %>%
  mutate(column_clean=clean_names(column)) %>%
  left_join(thora_clinical_data_labels)

thora_clinical_data_selected <- clean_names(thora_clinical_data) %>%
  filter(pasnr!=68) %>%
  mutate(start_treatment_corr=if_else(pasnr %in% c("126", "129") & is.na(start_strale_1), start_strale_2, start_strale_1),
         kur2_corr=case_when(pasnr == "129" & is.na(kur2) ~ as.Date("2017-05-29"),
                             pasnr == "176" & is.na(kur2) ~ as.Date("2018-06-12"),
                             .default=kur2),
         kur3_corr=if_else(pasnr == "129" & is.na(kur3), as.Date("2017-06-19"), kur3),
         kur4_corr=if_else(pasnr == "148" & is.na(kur4), as.Date("2017-11-13"), kur4),
         f_dato_corr=if_else(pasnr == "6", as.Date("1943-04-28"), f_dato)) %>%
  select(pasnr, sykehus, randomisering_arm, f_dato=f_dato_corr, kjonn, ink_1, dato_behstart, stage_tnm7_taha, ps_2023,
         kur1, kur2=kur2_corr, kur3=kur3_corr, kur4=kur4_corr, start_strale=start_treatment_corr, end_strale=strale_1, n_kurer=no_kurer, tot_straledose,
         dodsdato_taha2, status_os_taha2, os_k24, progresjon, progdato_taha2, last_pfsassessment_taha2, status_pfs_taha2, status_ttp_taha2, ttp_2023_taha, ttp_05_24, pfs_2023_taha)

thora_clinical_data_selected <- thora_clinical_data_selected %>%
  rowwise %>%
  mutate(pfs_date_corr=case_when(
    !is.na(progdato_taha2) ~ progdato_taha2,
    !is.na(dodsdato_taha2) ~ dodsdato_taha2,
    !is.na(last_pfsassessment_taha2) ~ last_pfsassessment_taha2,
    TRUE ~ NA
  ),
  ttp_date_corr=case_when(
    !is.na(progdato_taha2) ~ progdato_taha2,
    !is.na(last_pfsassessment_taha2) ~ last_pfsassessment_taha2,
    TRUE ~ NA
  ),
  pfs_status_corr=case_when(
    !is.na(progdato_taha2) ~ 1,
    !is.na(dodsdato_taha2) ~ 1,
    !is.na(last_pfsassessment_taha2) ~ 0,
    TRUE ~ NA
  ),
  ttp_status_corr=case_when(
    !is.na(progdato_taha2) ~ 1,
    !is.na(last_pfsassessment_taha2) ~ 0,
    TRUE ~ NA
  ),
  pfs_corr=(kur1 %--% pfs_date_corr)/months(1),
  pfs_corr=if_else(pfs_corr<0, 0, pfs_corr),
  ttp_corr=(kur1 %--% ttp_date_corr)/months(1),
  ttp_corr=if_else(ttp_corr<0, 0, ttp_corr),
  os_date_corr=if_else(is.na(dodsdato_taha2), ymd("2023-09-15"), dodsdato_taha2),
  os_corr=(kur1 %--% os_date_corr)/months(1),
  os_status_corr=if_else(is.na(dodsdato_taha2), 0, 1),
  time_since_t0_death=(kur1 %--% os_date_corr)/days(1),
  time_since_t0_progdato=(kur1 %--% progdato_taha2)/days(1),
  time_since_t0_ttp=(kur1 %--% ttp_date_corr)/days(1),
  diff_pfs_status=if_else(status_pfs_taha2==pfs_status_corr, 0, 1),
  diff_ttp_status=if_else(status_ttp_taha2==ttp_status_corr, 0, 1),
  diff_os_status=if_else(status_os_taha2==os_status_corr, 0, 1),
  abs_diff_pfs=round(abs(pfs_2023_taha-pfs_corr), digits=3),
  abs_diff_ttp=round(abs(ttp_2023_taha-ttp_corr), digits=3),
  abs_diff_os=round(abs(os_k24-os_corr), digits=3)) %>%
  ungroup() %>%
  mutate(randomisering_arm=na_if(randomisering_arm, ""),
         rt_arm=factor(randomisering_arm, labels = c("45 Gy", "60 Gy"))) %>%
  rename(sex=kjonn) %>%
  mutate(stad_bin=if_else(str_detect(stage_tnm7_taha, "III"), 1, 0),
         ps_bin=if_else(ps_2023>1, 1, 0))

# diff_pfsandttp %>%
#   filter(abs_diff_pfs > 2.5) %>%
#   select(pasnr, progdato_taha2, dodsdato_taha2, last_pfsassessment_taha2, Filipes_PFS_date=pfs_date_corr, Filipes_PFS_value=pfs_corr, existing_PFS_value=pfs_taha) %>%
#   arrange(pasnr) %>%
#   write_tsv(file="different_PFS_Filipe.txt", na="")

#Load sequencing files and sample sheet ----
seq_files <- list.files(path = "/mnt/archive/THORA/miRNA_periphery_GCF-2023-890/GCF-2023-890/", pattern = "*fastq.gz")

library(readxl)

sample_sheet_colnames <- colnames(clean_names(read_excel(path="/mnt/archive/THORA/miRNA_periphery_GCF-2023-890/Sample-Submission-Form.xlsx",
                                                         sheet=1,
                                                         range="A15:X15")))

sample_sheet <- read_excel(path="/mnt/archive/THORA/miRNA_periphery_GCF-2023-890/Sample-Submission-Form.xlsx",
                           sheet=1,
                           range="A16:X323",
                           col_names = sample_sheet_colnames) %>%
  rename(ID_sample=unique_sample_id, lane=batch)

seq_files_df <- as.data.frame(list(input=seq_files)) %>%
  mutate(strekkode=str_remove(input, pattern="_R1.fastq.gz")) %>%
  left_join(key_full) %>%
  left_join(read_delim(file="kristin_sugg.txt", delim=";", col_types = "cc") %>%
              mutate(ID_corr=paste0("00", ID)), by=c("strekkode"="ID_corr")) %>%
  mutate(provedato_corr=if_else(is.na(provedato), as.Date(kristin_sugg), provedato)) %>%
  rename(ID_sample=strekkode) %>%
  select(-ID, -kristin_sugg) %>%
  left_join(sample_sheet %>%
              select(ID_sample, lane, library_prep_plate))

correct_sample_nums <- seq_files_df %>%
  filter(str_detect(ID_sample, "Control", negate = TRUE),
         ID_sample!="00201130030607") %>%
  select(country, id_corr) %>%
  distinct() %>%
  mutate(id_corr_num=as.integer(str_extract(id_corr, "(\\d)+"))) %>%
  arrange(country, id_corr_num) %>%
  group_by(country) %>%
  mutate(rank_country=rank(id_corr_num)) %>%
  left_join(seq_files_df %>%
              select(ID_sample, id_corr, provedato_corr)) %>%
  group_by(id_corr) %>%
  mutate(rank_ind=rank(provedato_corr)) %>%
  mutate(id_person_visit_corr=paste0(country, rank_country, "_", rank_ind),
         id_person_corr=paste0(country, rank_country)) %>%
  arrange(country, id_corr_num, provedato_corr) %>%
  ungroup()

seq_files_df <- seq_files_df %>%
  left_join(correct_sample_nums %>%
              select(ID_sample, id_person_corr, id_person_visit_corr))

levels(seq_files_df$id_person_visit_corr) <- correct_sample_nums$id_person_visit_corr
levels(seq_files_df$id_person_corr) <- unique(correct_sample_nums$id_person_corr)

check_lanes <- correct_sample_nums %>%
  left_join(sample_sheet %>%
              select(ID_sample, lane)) %>%
  group_by(id_person_corr) %>%
  reframe(lanes=str_flatten_comma(unique(lane)))

#Join clinical data to sequencing files ----

seq_files_df_with_clinical <- seq_files_df %>%
  filter(!is.na(id_corr),
         !is.na(provedato_corr)) %>%
  left_join(thora_clinical_data_selected) %>%
  rowwise() %>%
  mutate(middle_strale=if_else(provedato_corr > start_strale & provedato_corr <= end_strale, "Yes", "No")) %>%
  mutate(
    time_since_t0=case_when(
      provedato_corr<=kur1 ~ 0,
      provedato_corr>kur1 ~ (kur1 %--% provedato_corr)/days(1),
      .default=NA),
    rt_dosage=case_when(
      provedato_corr <= start_strale ~ 0,
      provedato_corr > end_strale ~ tot_straledose,
      provedato_corr > start_strale & provedato_corr <= end_strale & ((start_strale %--% provedato_corr)/days(1)) == 1 ~ 3,
      provedato_corr > start_strale & provedato_corr <= end_strale & studieid=="TH145" & provedato_corr == "2017-09-25" ~ 15,
      # provedato_corr > start_strale & provedato_corr <= end_strale ~ (tot_straledose/((start_strale %--% end_strale)/days(1)+1))*((start_strale %--% provedato_corr)/days(1)),
      .default = NA
    ),
    chemo_dosage=case_when(
      provedato_corr <= kur1 ~ 0,
      provedato_corr > kur1 & n_kurer == 1 ~ 1,
      provedato_corr > kur1 & provedato_corr <= kur2 ~ 1,
      provedato_corr > kur2 & n_kurer == 2 ~ 2,
      provedato_corr > kur2 & provedato_corr <= kur3 ~ 2,
      provedato_corr > kur3 & n_kurer == 3 ~ 3,
      provedato_corr > kur3 & provedato_corr <= kur4 ~ 3,
      provedato_corr > kur4 & n_kurer == 4 ~ 4,
      .default = NA
    ),
    chemo_radiotherapy=case_when(
      chemo_dosage==0 & rt_dosage==0 ~ "Baseline",
      chemo_dosage==1 & time_since_t0<=2 & rt_dosage==0 ~ "Baseline",
      chemo_dosage>1 & rt_dosage==0 ~ "Chemo effect",
      chemo_dosage>=1 & rt_dosage<(tot_straledose/2) ~ "Chemo effect",
      chemo_dosage==1 & n_kurer==1 & rt_dosage==tot_straledose & ((kur1 %--% provedato_corr)/months(1)) <=2 & ((end_strale %--% provedato_corr)/months(1)) <=2  ~ "Chemoradiation <=2 months",
      chemo_dosage==2 & n_kurer==2 & rt_dosage==tot_straledose & ((kur2 %--% provedato_corr)/months(1)) <=2 & ((end_strale %--% provedato_corr)/months(1)) <=2  ~ "Chemoradiation <=2 months",
      chemo_dosage==3 & n_kurer==3 & rt_dosage==tot_straledose & ((kur3 %--% provedato_corr)/months(1)) <=2 & ((end_strale %--% provedato_corr)/months(1)) <=2  ~ "Chemoradiation <=2 months",
      chemo_dosage==4 & n_kurer==4 & rt_dosage==tot_straledose & ((kur4 %--% provedato_corr)/months(1)) <=2 & ((end_strale %--% provedato_corr)/months(1)) <=2  ~ "Chemoradiation <=2 months",
      chemo_dosage==1 & n_kurer==1 & rt_dosage==tot_straledose & ((kur1 %--% provedato_corr)/months(1)) >2 & ((end_strale %--% provedato_corr)/months(1)) <=2  ~ "Chemoradiation <=2 months",
      chemo_dosage==2 & n_kurer==2 & rt_dosage==tot_straledose & ((kur2 %--% provedato_corr)/months(1)) >2 & ((end_strale %--% provedato_corr)/months(1)) <=2  ~ "Chemoradiation <=2 months",
      chemo_dosage==3 & n_kurer==3 & rt_dosage==tot_straledose & ((kur3 %--% provedato_corr)/months(1)) >2 & ((end_strale %--% provedato_corr)/months(1)) <=2  ~ "Chemoradiation <=2 months",
      chemo_dosage==4 & n_kurer==4 & rt_dosage==tot_straledose & ((kur4 %--% provedato_corr)/months(1)) >2 & ((end_strale %--% provedato_corr)/months(1)) <=2  ~ "Chemoradiation <=2 months",
      chemo_dosage==1 & n_kurer==1 & rt_dosage==tot_straledose & ((kur1 %--% provedato_corr)/months(1)) <=2 & ((end_strale %--% provedato_corr)/months(1)) >2  ~ "Chemoradiation <=2 months",
      chemo_dosage==2 & n_kurer==2 & rt_dosage==tot_straledose & ((kur2 %--% provedato_corr)/months(1)) <=2 & ((end_strale %--% provedato_corr)/months(1)) >2  ~ "Chemoradiation <=2 months",
      chemo_dosage==3 & n_kurer==3 & rt_dosage==tot_straledose & ((kur3 %--% provedato_corr)/months(1)) <=2 & ((end_strale %--% provedato_corr)/months(1)) >2  ~ "Chemoradiation <=2 months",
      chemo_dosage==4 & n_kurer==4 & rt_dosage==tot_straledose & ((kur4 %--% provedato_corr)/months(1)) <=2 & ((end_strale %--% provedato_corr)/months(1)) >2  ~ "Chemoradiation <=2 months",
      chemo_dosage==1 & n_kurer==1 & rt_dosage==tot_straledose & ((kur1 %--% provedato_corr)/months(1)) >2 & ((end_strale %--% provedato_corr)/months(1)) >2  ~ "Chemoradiation >2 months",
      chemo_dosage==2 & n_kurer==2 & rt_dosage==tot_straledose & ((kur2 %--% provedato_corr)/months(1)) >2 & ((end_strale %--% provedato_corr)/months(1)) >2  ~ "Chemoradiation >2 months",
      chemo_dosage==3 & n_kurer==3 & rt_dosage==tot_straledose & ((kur3 %--% provedato_corr)/months(1)) >2 & ((end_strale %--% provedato_corr)/months(1)) >2  ~ "Chemoradiation >2 months",
      chemo_dosage==4 & n_kurer==4 & rt_dosage==tot_straledose & ((kur4 %--% provedato_corr)/months(1)) >2 & ((end_strale %--% provedato_corr)/months(1)) >2  ~ "Chemoradiation >2 months",
      TRUE ~ "Other"
    ),
    after_progression=case_when(
      is.na(progdato_taha2) ~ NA,
      !is.na(progdato_taha2) ~  (progdato_taha2 %--% provedato_corr)/months(1),
      .default = NA
    ),
    age_at_sample=trunc((f_dato %--% provedato_corr)/years(1)),
    before_after_progression=case_when(
      status_ttp_taha2==0 ~ 0,
      status_ttp_taha2==1 & after_progression < -2 ~ 0,
      status_ttp_taha2==1 & after_progression >= -2 ~ 1,
      .default = NA
    )) %>%
  ungroup()

seq_files_df_with_clinical %>%
  group_by(chemo_dosage, rt_dosage) %>%
  count

check_chemo_radio <- seq_files_df_with_clinical  %>%
  select(studieid, provedato_corr, kur1, kur2, kur3, kur4, start_strale, end_strale, time_since_t0, chemo_dosage, rt_dosage, n_kurer, tot_straledose, chemo_radiotherapy) %>%
  arrange(time_since_t0, chemo_dosage, rt_dosage) %>%
  rename(time_since_kur1=time_since_t0)

# write_tsv(check_chemo_radio, file="check_chemo_radio.txt", quote = "needed", na="")

seq_files_df_with_clinical %>%
  group_by(chemo_radiotherapy, before_after_progression) %>%
  count()

#TTP only after completing treatment; redefine completed trt -> remove time and add tot_straledose (dichotomize); TTP and OS (test with before and after treatment samples -> baseline vs. after treatment)

save.image(file="21.08.2024.RData")