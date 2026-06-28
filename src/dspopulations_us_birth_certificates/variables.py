"""Column utilities."""

from enum import StrEnum

import pandas as pd


class Variables(StrEnum):
    DATAYEAR = "datayear"
    """Year Birth of Child (Data Year). 1989-revision files only (positions 1-4)."""
    BIRYR = "biryr"
    """Year of Birth. 1989-revision files only (positions 176-179)."""
    DOB_YY = "dob_yy"
    """Year of birth (4-digit, e.g. 2023). 2003-revision files."""
    DOB_MM = "dob_mm"
    """Month of birth (01 January ... 12 December)."""
    DOB_WK = "dob_wk"
    """Birth day of week (1 Sunday ... 7 Saturday)."""
    DOB_TT = "dob_tt"
    """Time of Birth (HHMM, 0000-2359; 9999 Not Stated)."""
    BFACIL3 = "bfacil3"
    """Birth place recode (1 In Hospital, 2 Not in Hospital, 3 Unknown or Not Stated)."""
    MAGER = "mager"
    """Mother's Single Years of Age (12 = 10-12 years, 13-49 = single years, 50 = 50 years and over)."""
    DMAGE = "dmage"
    """Age of Mother (pre-2004, 1989 cert). Single years, 10-54; this is the age item NCHS publications used."""
    DMAGERPT = "dmagerpt"
    """Reported Age of Mother (pre-2004). 10-54 single years; 99 Unknown or not stated."""
    MAGER14 = "mager14"
    """Mother's Age Recode 14 (01 Under 15, 03 15, 04 16, 05 17, 06 18, 07 19, 08 20-24, 09 25-29, 10 30-34, 11 35-39, 12 40-44, 13 45-49, 14 50-54)."""
    MAGER9 = "mager9"
    """Mother's Age Recode 9 (1 Under 15, 2 15-19, 3 20-24, 4 25-29, 5 30-34, 6 35-39, 7 40-44, 8 45-49, 9 50-54)."""
    MAGE36 = "mage36"
    """Age of Mother Recode 36 (pre-2004). 01 Under 15, 02 15, 03 16 ... 41 54; single-year bins. In mage_c the +13 offset maps code 01 ("Under 15", a 10-14 bin) to age 14 - a lower-bound approximation that is immaterial above the lowest analytic age boundary (20)."""
    MAGER12 = "mager12"
    """Age of Mother Recode 12 (pre-2004; named MAGE12 in the source). 01 Under 15, 03 15, 04 16, 05 17, 06 18, 07 19, 08 20-24, 09 25-29, 10 30-34, 11 35-39, 12 40-44, 13 45-49, 14 50-54."""
    MAGER41 = "mager41"
    """Mother's Age Recode 41 (2003 transition file only). 01 Under 15, 02 15, 03 16 ... 41 54; single-year bins in the same coding as MAGE36. The 2003 file carries no MAGER/DMAGE/MAGE36, so this is the only single-year maternal-age source for 2003 (mage_c = mager41 + 13)."""
    MBSTATE_REC = "mbstate_rec"
    """Mother's Nativity (1 Born in the U.S. [50 states + DC], 2 Born outside the U.S. [includes possessions], 3 Unknown or Not Stated)."""
    RESTATUS = "restatus"
    """Mother's Residence Status. US occurrence: 1 Resident, 2 Intrastate non-resident, 3 Interstate non-resident, 4 Foreign resident. Territories use analogous codes."""
    MBRACE = "mbrace"
    """Mother's Bridged Race (2003-2019 revised-cert files; absent from 2020). Two schemes: 2003-2013 uses 2-digit codes (01-14 single-race, 21-24 multiple-race bridged to a single race); from 2014 a 1-digit recode (1 White, 2 Black, 3 AIAN, 4 Asian or Pacific Islander; Puerto Rico 0 Other, 1 White, 2 Black). Blank = not on certificate. In the mrace_c chain MRACEREC/MRACE15 precede MBRACE, so it is only the fallback for Puerto Rico 2014-2019 in practice."""
    MRACE = "mrace"
    """Mother's Race, 1989 cert (available 1989-2013). 01 White, 02 Black, 03 AIAN, 04 Chinese, 05 Japanese, 06 Hawaiian, 07 Filipino, 18/28/38/48/58 Asian Indian/Korean/Samoan/Vietnamese/Guamanian, 68 Other A/PI in areas reporting 18-58, 78 Combined other A/PI (includes 18-68)."""
    MRACEREC = "mracerec"
    """Mother's Race Recode (2005-2013). 1 White, 2 Black, 3 AIAN, 4 Asian/Pacific Islander; Puerto Rico: 0 Other, 1 White, 2 Black."""
    MRACE31 = "mrace31"
    """Mother's Race Recode 31 (2014+). 01 White only ... 31 Black, AIAN, Asian, NHOPI, and White; single-race 01-05, multi-race 06-31."""
    MRACE6 = "mrace6"
    """Mother's Race Recode 6 (2014+). 1 White only, 2 Black only, 3 AIAN only, 4 Asian only, 5 NHOPI only, 6 More than one race."""
    MRACE15 = "mrace15"
    """Mother's Race Recode 15 (2014+). 01 White only ... 14 Other Pacific Islander only, 15 More than one race."""
    MRACEIMP = "mraceimp"
    """Mother's Race Imputed Flag (Blank Not imputed, 1 Unknown race imputed, 2 All other races [formerly coded 09] imputed)."""
    ORMOTH = "ormoth"
    """Hispanic Origin of Mother (1989 cert, pre-2004). 0 Non-Hispanic, 1 Mexican, 2 Puerto Rican, 3 Cuban, 4 Central or South American, 5 Other and unknown Hispanic, 9 Origin unknown or not stated. Not reported by Puerto Rico, Northern Marianas, American Samoa."""
    ORRACEM = "orracem"
    """Hispanic Origin and Race of Mother Recode (1989 cert, pre-2004). 1 Mexican, 2 Puerto Rican, 3 Cuban, 4 Central/South American, 5 Other/Unknown Hispanic, 6 Non-Hispanic White, 7 Non-Hispanic Black, 8 Non-Hispanic other races, 9 Origin unknown or not stated."""
    UMHISP = "umhisp"
    """Mother's Hispanic Origin, 1989-cert variant (2004-2013 files). 0 Non-Hispanic, 1 Mexican, 2 Puerto Rican, 3 Cuban, 4 Central or South American, 5 Other and Unknown Hispanic, 9 Origin unknown or not stated."""
    MHISPX = "mhispx"
    """Mother's Hispanic Origin (2018+; positions 112-114 are FILLER 2014-2017). 0 Non-Hispanic, 1 Mexican, 2 Puerto Rican, 3 Cuban, 4 Central or South American, 5 Dominican, 6 Other and Unknown Hispanic, 9 Origin unknown or not stated. The Dominican (5) breakout was introduced in 2018."""
    MHISP_R = "mhisp_r"
    """Mother's Hispanic Origin Recode (2014+). 0 Non-Hispanic, 1 Mexican, 2 Puerto Rican, 3 Cuban, 4 Central and South American, 5 Other and Unknown Hispanic origin, 9 Hispanic origin not stated."""
    MRACEHISP = "mracehisp"
    """Mother's Race/Hispanic Origin. NCHS uses two incompatible schemes across eras: 2003-2013 = combined Hispanic-origin-and-race (1-5 Hispanic subgroups, 6 NH White, 7 NH Black, 8 NH Other, 9 unknown); 2014+ = race recode built from MRACE31/MRACE6/MRACE15 (1 NH White only, 2 NH Black only, 3 NH AIAN only, 4 NH Asian only, 5 NH NHOPI only, 6 NH more than one race, 7 Hispanic, 8 Origin unknown or not stated). Absent pre-2003. NOTE: this raw field is NOT used for the combined mracehisp_c, which is reconstructed from mhisp_c + mrace_c for a single consistent 1989-2024 coding (see MRACEHISP_C)."""
    MAR = "mar"
    """Mother's Marital Status (1989-2013 files). US/Virgin Islands/Guam/AS/NMI: 1 Married, 2 Unmarried, 9 Unknown or not stated. Puerto Rico: 1 Married, 2 Unmarried parents living together, 3 Unmarried parents not living together, 9 Unknown or not stated."""
    MAR_P = "mar_p"
    """Paternity Acknowledged (Y Yes, N No, U Unknown, X Not Applicable)."""
    DMAR = "dmar"
    """Marital Status (2014+). US/outlying except PR: 1 Married, 2 Unmarried. Puerto Rico: 1 Married, 2 Unmarried parents living together, 3 Unmarried parents not living together, 9 Unknown or not stated."""
    DMEDUC = "dmeduc"
    """Mother's Education, 1989 cert (pre-2003). 00 No formal education, 01-08 elementary years, 09-12 high school years, 13-16 college years, 17 5+ years college, 99 Not stated."""
    MEDUC = "meduc"
    """Mother's Education, 2003 cert. 1 8th grade or less, 2 9th-12th grade no diploma, 3 HS grad/GED, 4 Some college no degree, 5 Associate (AA/AS), 6 Bachelor's, 7 Master's, 8 Doctorate/Professional, 9 Unknown."""
    UMEDUC = "umeduc"
    """Mother's Education, 1989-cert variant (2003+ files). Same codes as DMEDUC: 00 No formal education, 01-08 elementary, 09-12 high school, 13-16 college, 17 5+ years college, 99 Not stated."""
    MEDUC6 = "meduc6"
    """Education of Mother Recode 6 (pre-2003). 1 0-8 years, 2 9-11 years, 3 12 years, 4 13-15 years, 5 16+ years, 6 Not stated."""
    MEDUC_REC = "meduc_rec"
    """Mother's Education Recode, 1989-cert (2003+ files; same code structure as MEDUC6). 1 0-8 years, 2 9-11 years, 3 12 years, 4 13-15 years, 5 16+ years, 6 Not stated."""
    MPLBIR = "mplbir"
    """Place of Birth of Mother. 01-51 US states/DC, 52 Puerto Rico, 53 Virgin Islands, 54 Guam, 55 Canada, 56 Cuba, 57 Mexico, 59 Remainder of the World, 61 American Samoa, 62 Northern Marianas, 99 Not classifiable."""
    DFAGE = "dfage"
    """Age of Father (pre-2004, 1989 cert). 10-98 single years, 99 Unknown or not stated."""
    DFAGERPT = "dfagerpt"
    """Reported Age of Father (pre-2004). 10-98 single years, 99 Unknown or not stated."""
    FAGE11 = "fage11"
    """Age of Father Recode 11 (pre-2004). 01 Under 15, 02 15-19, 03 20-24, 04 25-29, 05 30-34, 06 35-39, 07 40-44, 08 45-49, 09 50-54, 10 55-98, 11 Not stated."""
    FAGERPT = "fagerpt"
    """Father's Reported Age (2003+). 09-98 single years, 99 Unknown or not stated."""
    UFAGECOMB = "ufagecomb"
    """Father's Combined Age, 1989-cert variant (2003+ files). 10-98 single years, 99 Unknown or not stated."""
    FAGECOMB = "fagecomb"
    """Father's Combined Age, 2003 cert. 09-98 single years, 99 Unknown or not stated."""
    FAGEREC11 = "fagerec11"
    """Father's Age Recode 11 (2003 cert). 01 Under 15, 02 15-19, 03 20-24, 04 25-29, 05 30-34, 06 35-39, 07 40-44, 08 45-49, 09 50-54, 10 55-98, 11 Not stated."""
    FBRACE = "fbrace"
    """Father's Bridged Race (2003-2013 revised-cert files). 01-14 single-race, 21-24 multiple-race bridged to single race, 99 Unknown/not stated/non-reporting, Blank Not on certificate."""
    ORFATH = "orfath"
    """Hispanic Origin of Father (1989 cert, pre-2004). 0 Non-Hispanic, 1 Mexican, 2 Puerto Rican, 3 Cuban, 4 Central or South American, 5 Other and unknown Hispanic, 9 Origin unknown or not stated. Not reported by Puerto Rico, Northern Marianas, American Samoa."""
    ORRACEF = "orracef"
    """Hispanic Origin and Race of Father Recode (1989 cert, pre-2004). 1-5 Hispanic subgroups, 6 Non-Hispanic White, 7 Non-Hispanic Black, 8 Non-Hispanic other or unknown race, 9 Origin unknown or not stated."""
    FRACE = "frace"
    """Race of Father, 1989 cert (available 1989-2013). Same code structure as MRACE: 01-07 primary races, 18/28/38/48/58 expanded A/PI, 68 Other A/PI in reporting areas, 78 Combined other A/PI, 99 Unknown or Not Stated."""
    FRACEIMP = "fraceimp"
    """Father's Race Imputed Flag (1989 cert). Blank race not changed, 3 All other races formerly coded 09 changed to code 99."""
    FRACEREC = "fracerec"
    """Father's Race Recode (2005-2013). 1 White, 2 Black, 3 AIAN, 4 Asian/Pacific Islander, 9 Unknown or not stated; Puerto Rico adds 0 Other (not classified as White or Black)."""
    UFHISP = "ufhisp"
    """Father's Hispanic Origin, 1989-cert variant (2004-2013 files). 0 Non-Hispanic, 1 Mexican, 2 Puerto Rican, 3 Cuban, 4 Central or South American, 5 Other and Unknown Hispanic, 9 Origin unknown or not stated."""
    FRACEHISP = "fracehisp"
    """Father's Race/Hispanic Origin (2014+; built from FRACE31/FRACE6/FRACE15). 1-6 as for MRACEHISP, 7 Hispanic, 8 Origin unknown or not stated, 9 Race unknown or not stated (Non-Hispanic)."""
    FRACE31 = "frace31"
    """Father's Race Recode 31 (2014+). 01-31 same structure as MRACE31; 99 Unknown or Not Stated."""
    FRACE6 = "frace6"
    """Father's Race Recode 6 (2014+). 1 White only, 2 Black only, 3 AIAN only, 4 Asian only, 5 NHOPI only, 6 More than one race, 9 Unknown or Not Stated."""
    FRACE15 = "frace15"
    """Father's Race Recode 15 (2014+). 01-15 same structure as MRACE15; 99 Unknown or Not Stated."""
    FHISPX = "fhispx"
    """Father's Hispanic Origin (2018+; position 159 FILLER 2014-2017). Same codes as MHISPX: 0 Non-Hispanic, 1 Mexican, 2 Puerto Rican, 3 Cuban, 4 Central or South American, 5 Dominican, 6 Other and Unknown Hispanic, 9 Origin unknown or not stated."""
    FHISP_R = "fhisp_r"
    """Father's Hispanic Origin Recode (2014+). Same codes as MHISP_R: 0 Non-Hispanic, 1 Mexican, 2 Puerto Rican, 3 Cuban, 4 Central and South American, 5 Other and Unknown Hispanic origin, 9 Hispanic origin not stated."""
    FEDUC = "feduc"
    """Father's Education (2003 cert). Same codes as MEDUC: 1 8th grade or less, 2 9th-12th grade no diploma, 3 HS grad/GED, 4 Some college no degree, 5 Associate, 6 Bachelor's, 7 Master's, 8 Doctorate/Professional, 9 Unknown."""
    PRIORLIVE = "priorlive"
    """Prior Births Now Living. 00-30 count of children still living from previous live births, 99 Unknown or not stated."""
    PRIORDEAD = "priordead"
    """Prior Births Now Dead. 00-30 count of children dead from previous live births, 99 Unknown or not stated."""
    PRIORTERM = "priorterm"
    """Prior Other Terminations. 00-30 count of other terminations (spontaneous/induced at any time after conception), 99 Unknown or not stated."""
    LBO_REC = "lbo_rec"
    """Live Birth Order Recode. 1-7 live-birth order, 8 8 or more live births, 9 Unknown or not stated."""
    TBO_REC = "tbo_rec"
    """Total Birth Order Recode. 1-7 total-birth order, 8 8 or more total births, 9 Unknown or not stated."""
    ILLB_R11 = "illb_r11"
    """Interval Since Last Live Birth Recode 11. 00 0-3 months (plural delivery), 01 4-11, 02 12-17, 03 18-23, 04 24-35, 05 36-47, 06 48-59, 07 60-71, 08 72+ months, 88 Not applicable (1st live birth), 99 Unknown or not stated."""
    ILOP_R11 = "ilop_r11"
    """Interval Since Last Other Pregnancy Recode 11. Same bins as ILLB_R11 but measured since last other pregnancy; 88 Not applicable (1st natality event)."""
    ILP_R11 = "ilp_r11"
    """Interval Since Last Pregnancy Recode 11. Same bins as ILLB_R11 but measured since last pregnancy of any kind; 88 Not applicable (no previous pregnancy)."""
    PRECARE = "precare"
    """Month Prenatal Care Began. 00 No prenatal care, 01-10 month care began, 99 Unknown or not stated."""
    PAY = "pay"
    """Payment Source for Delivery. 1 Medicaid, 2 Private Insurance, 3 Self-Pay, 4 Indian Health Service, 5 CHAMPUS/TRICARE, 6 Other Government (Fed/State/Local), 8 Other, 9 Unknown."""
    PAY_REC = "pay_rec"
    """Payment Recode. 1 Medicaid, 2 Private Insurance, 3 Self Pay, 4 Other, 9 Unknown."""
    APGAR5 = "apgar5"
    """Five-Minute APGAR Score. 00-10 raw score, 99 Unknown or not stated."""
    APGAR5R = "apgar5r"
    """Five-Minute APGAR Recode. 1 score 0-3, 2 score 4-6, 3 score 7-8, 4 score 9-10, 5 Unknown or not stated."""
    APGAR10 = "apgar10"
    """Ten-Minute APGAR Score. 00-10 raw score, 88 Not applicable, 99 Unknown or not stated."""
    APGAR10R = "apgar10r"
    """Ten-Minute APGAR Recode. 1 score 0-3, 2 score 4-6, 3 score 7-8, 4 score 9-10, 5 Not stated/not applicable."""
    DPLURAL = "dplural"
    """Plurality Recode. 1 Single, 2 Twin, 3 Triplet, 4 Quadruplet or higher."""
    IMP_PLURAL = "imp_plural"
    """Plurality Imputed Flag (IMP_PLUR in source). Blank Plurality not imputed, 1 Plurality is imputed."""
    SETORDER_R = "setorder_r"
    """Set Order Recode (for multiples). 1 1st, 2 2nd, 3 3rd, 4 4th, 5 5th-16th, 9 Unknown or not stated."""
    SEX = "sex"
    """Sex of Infant (M Male, F Female). Post-2004."""
    GESTREC10 = "gestrec10"
    """Combined Gestation Recode 10. 01 Under 20 weeks, 02 20-27, 03 28-31, 04 32-33, 05 34-36, 06 37-38, 07 39, 08 40, 09 41, 10 42+ weeks, 99 Unknown."""
    DBWT = "dbwt"
    """Birth Weight in grams (edited). 0227-8165 grams, 9999 Not stated birth weight."""
    DWGT_R = "dwgt_r"
    """Delivery Weight Recode (pounds). 100-400 weight in pounds, 999 Unknown or not stated."""
    AB_AVEN1 = "ab_aven1"
    """Assisted Ventilation (immediately). Y Yes, N No, U Unknown or not stated."""
    AB_AVEN6 = "ab_aven6"
    """Assisted Ventilation > 6 hrs. Y Yes, N No, U Unknown or not stated."""
    AB_NICU = "ab_nicu"
    """Admission to NICU. Y Yes, N No, U Unknown or not stated."""
    AB_SURF = "ab_surf"
    """Newborn Surfactant Replacement. Y Yes, N No, U Unknown or not stated."""
    AB_ANTI = "ab_anti"
    """Antibiotics for Newborn. Y Yes, N No, U Unknown or not stated."""
    AB_SEIZ = "ab_seiz"
    """Seizures (newborn). Y Yes, N No, U Unknown or not stated."""
    NO_ABNORM = "no_abnorm"
    """No Abnormal Conditions Checked. 1 True, 0 False, 9 Not Reported."""
    CA_ANEN = "ca_anen"
    """Anencephaly (2003 cert). Y Yes, N No, U Unknown or not stated."""
    CA_MNSB = "ca_mnsb"
    """Meningomyelocele / Spina Bifida (2003 cert). Y Yes, N No, U Unknown or not stated."""
    CA_CCHD = "ca_cchd"
    """Cyanotic Congenital Heart Disease (2003 cert). Y Yes, N No, U Unknown or not stated."""
    CA_CDH = "ca_cdh"
    """Congenital Diaphragmatic Hernia (2003 cert). Y Yes, N No, U Unknown or not stated."""
    CA_OMPH = "ca_omph"
    """Omphalocele (2003 cert). Y Yes, N No, U Unknown or not stated."""
    CA_GAST = "ca_gast"
    """Gastroschisis (2003 cert). Y Yes, N No, U Unknown or not stated."""
    CA_LIMB = "ca_limb"
    """Limb Reduction Defect (2003 cert). Y Yes, N No, U Unknown or not stated."""
    CA_CLEFT = "ca_cleft"
    """Cleft Lip w/ or w/o Cleft Palate (2003 cert). Y Yes, N No, U Unknown or not stated."""
    CA_CLPAL = "ca_clpal"
    """Cleft Palate alone (2003 cert). Y Yes, N No, U Unknown or not stated."""
    DOWNS = "downs"
    """Down's syndrome, 1989 cert (1989-2002). Checkbox-style item: 1 Anomaly reported, 2 Anomaly not reported, 8 Anomaly not on certificate, 9 Anomaly not classifiable."""
    UCA_DOWNS = "uca_downs"
    """Down Syndrome, 1989-cert variant (2003-2013 files). Checkbox-style item: 1 Anomaly reported, 2 Anomaly not reported, 9 Anomaly not classifiable, Blank Not on certificate. (The 2003 transition file may carry the 1989-standard header with 8 = anomaly not on certificate; 8 is unmatched in the ca_down_c derivation and treated as missing.)"""
    CA_DOWN = "ca_down"
    """Down Syndrome, 2003 cert (2004-2006 and 2014-present; field is CA_DOWNS 2007-2013). C Confirmed, P Pending, N No, U Unknown, Blank Not on certificate."""
    CA_DOWNS = "ca_downs"
    """Down Syndrome, 2003 cert (2007-2013 only; same C/P/N/U coding as CA_DOWN). NCHS renamed the field to CA_DOWN from 2014 onward, keeping the reporting flag F_CA_DOWNS."""
    CA_DOWN_C = "ca_down_c"
    """Combined Down Syndrome indicator — harmonises DOWNS/UCA_DOWNS/CA_DOWN/CA_DOWNS across years to a single C/P/N/U categorical."""
    CA_DISOR = "ca_disor"
    """Suspected Chromosomal Disorder (2003 cert). C Confirmed, P Pending, N No, U Unknown, Blank Not on certificate."""
    CA_HYPO = "ca_hypo"
    """Hypospadias (2003 cert). Y Yes (anomaly reported), N No (anomaly not reported), U Unknown, Blank Not on certificate."""
    NO_CONGEN = "no_congen"
    """No Congenital Anomalies Checked. 1 True, 0 False, 9 Not Reported."""
    BFED = "bfed"
    """Infant Breastfed at Discharge. Y Yes, N No, U Unknown or not stated."""
    PREVIS = "previs"
    """Number of Prenatal Visits. 00-98 visits, 99 Unknown or not stated."""
    PREVIS_REC = "previs_rec"
    """Number of Prenatal Visits Recode. 01 No visits, 02 1-2, 03 3-4, 04 5-6, 05 7-8, 06 9-10, 07 11-12, 08 13-14, 09 15-16, 10 17-18, 11 19+ visits, 12 Unknown or not stated."""
    WIC = "wic"
    """WIC food received during pregnancy. Y Yes, N No, U Unknown or not stated."""
    M_HT_IN = "m_ht_in"
    """Mother's Height in Total Inches. 30-78 inches, 99 Unknown or not stated."""
    BMI = "bmi"
    """Body Mass Index (computed from M_HT_IN and PWGT_R). 13.0-69.9 BMI, 99.9 Unknown or not stated."""
    BMI_R = "bmi_r"
    """Body Mass Index Recode. 1 Underweight (<18.5), 2 Normal (18.5-24.9), 3 Overweight (25.0-29.9), 4 Obesity I (30.0-34.9), 5 Obesity II (35.0-39.9), 6 Extreme Obesity III (>=40.0), 9 Unknown or not stated."""
    PWGT_R = "pwgt_r"
    """Pre-pregnancy Weight Recode (pounds). 075-375 weight in pounds, 999 Unknown or not stated."""
    WTGAIN = "wtgain"
    """Weight Gain during pregnancy (pounds). 00-97 pounds, 98 98 pounds and over, 99 Unknown or not stated."""
    RF_PDIAB = "rf_pdiab"
    """Pre-pregnancy Diabetes. Y Yes, N No, U Unknown or not stated."""
    RF_GDIAB = "rf_gdiab"
    """Gestational Diabetes. Y Yes, N No, U Unknown or not stated."""
    RF_PHYPE = "rf_phype"
    """Pre-pregnancy Hypertension. Y Yes, N No, U Unknown or not stated."""
    RF_GHYPE = "rf_ghype"
    """Gestational Hypertension. Y Yes, N No, U Unknown or not stated."""
    RF_EHYPE = "rf_ehype"
    """Hypertension Eclampsia. Y Yes, N No, U Unknown or not stated."""
    RF_PPTERM = "rf_ppterm"
    """Previous Preterm Birth. Y Yes, N No, U Unknown or not stated."""
    RF_INFTR = "rf_inftr"
    """Infertility Treatment Used. Y Yes, N No, U Unknown or not stated."""
    RF_FEDRG = "rf_fedrg"
    """Fertility-Enhancing Drugs. Y Yes, N No, X Not applicable, U Unknown or not stated."""
    RF_ARTEC = "rf_artec"
    """Assisted Reproductive Technology. Y Yes, N No, X Not applicable, U Unknown or not stated."""
    RF_CESAR = "rf_cesar"
    """Previous Cesarean. Y Yes, N No, U Unknown or not stated."""
    RF_CESARN = "rf_cesarn"
    """Number of Previous Cesareans. 00 None, 01-30 count, 99 Unknown or not stated."""
    NO_RISKS = "no_risks"
    """No Risk Factors Reported. 1 True, 0 False, 9 Not Reported."""
    LD_INDL = "ld_indl"
    """Induction of Labor. Y Yes, N No, U Unknown or not stated."""
    LD_AUGM = "ld_augm"
    """Augmentation of Labor. Y Yes, N No, U Unknown or not stated."""
    LD_ANES = "ld_anes"
    """Anesthesia. Y Yes, N No, U Unknown or not stated."""
    ME_PRES = "me_pres"
    """Fetal Presentation at Delivery. 1 Cephalic, 2 Breech, 3 Other, 9 Unknown or not stated."""
    RDMETH_REC = "rdmeth_rec"
    """Delivery Method Recode (detailed). 1 Vaginal (excludes VBAC), 2 Vaginal after previous c-section, 3 Primary C-section, 4 Repeat C-section, 5 Vaginal (unknown if previous c-section), 6 C-section (unknown if previous c-section), 9 Not stated."""
    DMETH_REC = "dmeth_rec"
    """Delivery Method Recode (collapsed). 1 Vaginal, 2 C-Section, 9 Unknown."""
    ATTEND = "attend"
    """Attendant at Birth. 1 Doctor of Medicine (MD), 2 Doctor of Osteopathy (DO), 3 Certified Nurse Midwife/Certified Midwife (CNM/CM), 4 Other Midwife, 5 Other, 9 Unknown or not stated."""

    # added/computed columns

    YEAR = "year"

    MAGE_C = "mage_c"

    MRACE_C = "mrace_c"
    """Combined maternal race, harmonised 1989-2024 to 1 White, 2 Black, 3 AIAN,
    4 Asian or Pacific Islander, 5 More than one race, via the fallback chain
    MRACE15 > MRACEREC > MBRACE > MRACE. Asian and NHOPI single-race codes are
    collapsed into 4. Category 5 is only identifiable from MRACE15=15 (2014+) and
    MBRACE bridged-multiple 21-24 (2003-2013; unreachable in practice); MRACEREC
    and the 1989-cert MRACE carry no multi-race code, so multi-race births in
    1989-2013 are folded into single-race categories. Unknown/out-of-range codes
    are NULL (the 1989-cert MRACE has no unknown code - race is imputed)."""

    MHISP_C = "mhisp_c"
    """Combined maternal Hispanic origin, harmonised 1989-2024 to 0 Non-Hispanic,
    1 Mexican, 2 Puerto Rican, 3 Cuban, 4 Other and Unknown Hispanic, 5 Origin
    unknown or not stated, via the fallback chain MHISP_R > MHISPX > UMHISP >
    ORRACEM (ORRACEM's non-Hispanic race codes 6-8 map to 0)."""

    MRACEHISP_C = "mracehisp_c"
    """Combined maternal race/Hispanic origin: 1 NH White, 2 NH Black, 3 NH AIAN,
    4 NH Asian or Pacific Islander, 5 Hispanic, 6 NH more than one race. Reconstructed
    from mhisp_c + mrace_c (NOT the raw NCHS MRACEHISP, which is dual-coded across eras
    and absent pre-2003) so the coding is consistent across 1989-2024: mhisp_c 1-4 -> 5
    (Hispanic); mhisp_c = 5 (origin unknown) -> NULL (race discarded); non-Hispanic
    multi-race (mrace_c = 5) -> 6; mhisp_c = 0 or NULL -> mrace_c (non-Hispanic race 1-4).
    NB: selection.data and derive_recording_rates currently route code 6 to the
    'Unknown' race bucket (same cell as NULL); giving multi-race its own model group
    is a follow-up decision."""

    DOWN_IND = "down_ind"  # DS indicated (DOWNS | UCA_DOWNS | CA_DOWNS | CA_DOWN)

    DS = "ds"  # phase out for DS_CORP

    DS_C = "ds_c"
    DS_P = "ds_p"
    DS_N = "ds_n"
    DS_U = "ds_u"
    DS_CORP = "ds_corp"

    P_DS_LB_WT = "p_ds_lb_wt"
    """
    Probability of Down syndrome live birth with terminations. Estimated from surveillance-based
    prevalence for the given year with no additional adjustments (for maternal age or ethnicity).
    """

    P_DS_LB_NT = "p_ds_lb_nt"
    """
    Probability of Down syndrome live birth absent terminations. Estimated from maternal age
    using Morris formula.
    """

    P_DS_LB_WT_MAGE = "p_ds_lb_wt_mage"
    """
    Probability of Down syndrome live birth with terminations. Estimated from surveillance-based
    prevalence for the given year and maternal age.
    """

    P_DS_LB_NT_MAGE = "p_ds_lb_nt_mage"
    """
    Placeholder - declared but NOT yet populated by the pipeline (remains NULL).
    Intended: probability of Down syndrome live birth absent terminations, estimated
    from surveillance-based prevalence for the given year and maternal age.
    """

    P_DS_LB_WT_ETHN = "p_ds_lb_wt_ethn"
    """
    Placeholder - declared but NOT yet populated by the pipeline (remains NULL). The
    ethnicity prevalence table (us-births-estimated-prevalence-ethnicity-2000-2018.csv)
    is loaded but never joined. Intended: probability of Down syndrome live birth with
    terminations, estimated from surveillance-based prevalence for the given year and ethnicity.
    """

    P_DS_LB_NT_ETHN = "p_ds_lb_nt_ethn"
    """
    Placeholder - declared but NOT yet populated by the pipeline (remains NULL).
    Intended: probability of Down syndrome live birth absent terminations, estimated
    from surveillance-based prevalence for the given year and ethnicity.
    """

    P_DS_LB_NT_REDUC = "p_ds_lb_nt_reduc"
    """
    Probability of Down syndrome live birth with terminations, estimated as the maternal-age
    Morris no-terminations risk reduced by the surveillance-based reduction rate for the year:
    P_DS_LB_NT * (1 - reduction[year]). (Renamed from p_ds_lb_wt_mage_reduc, whose `_mage` was a
    misnomer - the multiplicand is P_DS_LB_NT, not P_DS_LB_WT_MAGE.)
    """

    DS_CASE_WEIGHT = "ds_case_weight"
    """
    Per-record recording-rate weight for Down-syndrome cases, used to up-weight recorded cases
    toward estimated true counts. For down_ind=1 it is selected by mracehisp_c (1 nhw, 2 nhb,
    3 ai_an, 4 as_pi, 5 his) from us-births-ds-rec-weights.csv, falling back to the year's pooled
    `total` weight when mracehisp_c is NULL/other; 0 for non-cases. Materialised in
    scripts/duckdb_prepare.py.
    """


COMPUTED: dict[
    str, pd.UInt16Dtype | pd.Float64Dtype | pd.CategoricalDtype | pd.CategoricalDtype
] = {
    str(Variables.YEAR): pd.UInt16Dtype(),
    str(Variables.MAGE_C): pd.UInt16Dtype(),
    str(Variables.DS): pd.CategoricalDtype(),
    str(Variables.P_DS_LB_NT): pd.Float64Dtype(),
    str(Variables.P_DS_LB_WT): pd.Float64Dtype(),
    str(Variables.CA_DOWN_C): pd.CategoricalDtype(
        categories=["C", "P", "N", "U"], ordered=False
    ),
    str(Variables.DOWN_IND): pd.CategoricalDtype(),
    str(Variables.DS_C): pd.CategoricalDtype(),
    str(Variables.DS_P): pd.CategoricalDtype(),
    str(Variables.DS_N): pd.CategoricalDtype(),
    str(Variables.DS_U): pd.CategoricalDtype(),
    str(Variables.DS_CORP): pd.CategoricalDtype(),
    str(Variables.P_DS_LB_WT_MAGE): pd.Float64Dtype(),
    str(Variables.P_DS_LB_NT_MAGE): pd.Float64Dtype(),
    str(Variables.P_DS_LB_WT_ETHN): pd.Float64Dtype(),
    str(Variables.P_DS_LB_NT_ETHN): pd.Float64Dtype(),
    str(Variables.P_DS_LB_NT_REDUC): pd.Float64Dtype(),
}

IMPORTED: dict[
    str,
    pd.Float32Dtype
    | pd.Float64Dtype
    | pd.CategoricalDtype
    | pd.UInt16Dtype
    | pd.UInt32Dtype
    | pd.UInt64Dtype
    | pd.Int16Dtype
    | pd.Int32Dtype
    | pd.Int64Dtype
    | pd.CategoricalDtype,
] = {
    str(Variables.DATAYEAR): pd.UInt16Dtype(),
    str(Variables.BIRYR): pd.UInt16Dtype(),
    str(Variables.DOB_YY): pd.UInt16Dtype(),
    str(Variables.DOB_MM): pd.CategoricalDtype(),
    str(Variables.DOB_WK): pd.CategoricalDtype(),
    str(Variables.DOB_TT): pd.CategoricalDtype(),
    str(Variables.BFACIL3): pd.CategoricalDtype(),
    str(Variables.MAGER): pd.CategoricalDtype(),
    str(Variables.DMAGE): pd.CategoricalDtype(),
    str(Variables.DMAGERPT): pd.CategoricalDtype(),
    str(Variables.MAGER14): pd.CategoricalDtype(),
    str(Variables.MAGER9): pd.CategoricalDtype(),
    str(Variables.MAGE36): pd.CategoricalDtype(),
    str(Variables.MAGER12): pd.CategoricalDtype(),
    str(Variables.MAGER41): pd.CategoricalDtype(),
    str(Variables.MBSTATE_REC): pd.CategoricalDtype(),
    str(Variables.RESTATUS): pd.CategoricalDtype(),
    str(Variables.MBRACE): pd.CategoricalDtype(),
    str(Variables.MRACE): pd.CategoricalDtype(),
    str(Variables.MRACEREC): pd.CategoricalDtype(),
    str(Variables.MRACE31): pd.CategoricalDtype(),
    str(Variables.MRACE6): pd.CategoricalDtype(),
    str(Variables.MRACE15): pd.CategoricalDtype(),
    str(Variables.MRACEIMP): pd.CategoricalDtype(),
    str(Variables.ORMOTH): pd.CategoricalDtype(),
    str(Variables.ORRACEM): pd.CategoricalDtype(),
    str(Variables.UMHISP): pd.CategoricalDtype(),
    str(Variables.MHISPX): pd.CategoricalDtype(),
    str(Variables.MHISP_R): pd.CategoricalDtype(),
    str(Variables.MRACEHISP): pd.CategoricalDtype(),
    str(Variables.MAR): pd.CategoricalDtype(),
    str(Variables.MAR_P): pd.CategoricalDtype(),
    str(Variables.DMAR): pd.CategoricalDtype(),
    str(Variables.DMEDUC): pd.CategoricalDtype(),
    str(Variables.MEDUC): pd.CategoricalDtype(),
    str(Variables.UMEDUC): pd.CategoricalDtype(),
    str(Variables.MEDUC6): pd.CategoricalDtype(),
    str(Variables.MEDUC_REC): pd.CategoricalDtype(),
    str(Variables.MPLBIR): pd.CategoricalDtype(),
    str(Variables.DFAGE): pd.CategoricalDtype(),
    str(Variables.DFAGERPT): pd.CategoricalDtype(),
    str(Variables.FAGE11): pd.CategoricalDtype(),
    str(Variables.FAGERPT): pd.CategoricalDtype(),
    str(Variables.UFAGECOMB): pd.CategoricalDtype(),
    str(Variables.FAGECOMB): pd.CategoricalDtype(),
    str(Variables.FAGEREC11): pd.CategoricalDtype(),
    str(Variables.FBRACE): pd.CategoricalDtype(),
    str(Variables.ORFATH): pd.CategoricalDtype(),
    str(Variables.ORRACEF): pd.CategoricalDtype(),
    str(Variables.FRACEIMP): pd.CategoricalDtype(),
    str(Variables.FRACEREC): pd.CategoricalDtype(),
    str(Variables.UFHISP): pd.CategoricalDtype(),
    str(Variables.FRACEHISP): pd.CategoricalDtype(),
    str(Variables.FRACE31): pd.CategoricalDtype(),
    str(Variables.FRACE6): pd.CategoricalDtype(),
    str(Variables.FRACE15): pd.CategoricalDtype(),
    str(Variables.FHISPX): pd.CategoricalDtype(),
    str(Variables.FHISP_R): pd.CategoricalDtype(),
    str(Variables.FRACE): pd.CategoricalDtype(),
    str(Variables.FEDUC): pd.CategoricalDtype(),
    str(Variables.PRIORLIVE): pd.CategoricalDtype(),
    str(Variables.PRIORDEAD): pd.CategoricalDtype(),
    str(Variables.PRIORTERM): pd.CategoricalDtype(),
    str(Variables.LBO_REC): pd.CategoricalDtype(),
    str(Variables.TBO_REC): pd.CategoricalDtype(),
    str(Variables.ILLB_R11): pd.CategoricalDtype(),
    str(Variables.ILOP_R11): pd.CategoricalDtype(),
    str(Variables.ILP_R11): pd.CategoricalDtype(),
    str(Variables.PRECARE): pd.CategoricalDtype(),
    str(Variables.PAY): pd.CategoricalDtype(),
    str(Variables.PAY_REC): pd.CategoricalDtype(),
    str(Variables.APGAR5): pd.CategoricalDtype(),
    str(Variables.APGAR5R): pd.CategoricalDtype(),
    str(Variables.APGAR10): pd.CategoricalDtype(),
    str(Variables.APGAR10R): pd.CategoricalDtype(),
    str(Variables.DPLURAL): pd.CategoricalDtype(),
    str(Variables.IMP_PLURAL): pd.CategoricalDtype(),
    str(Variables.SETORDER_R): pd.CategoricalDtype(),
    str(Variables.PREVIS): pd.CategoricalDtype(),
    str(Variables.PREVIS_REC): pd.CategoricalDtype(),
    str(Variables.WIC): pd.CategoricalDtype(),
    str(Variables.SEX): pd.CategoricalDtype(),
    str(Variables.GESTREC10): pd.CategoricalDtype(),
    str(Variables.DBWT): pd.CategoricalDtype(),
    str(Variables.DWGT_R): pd.CategoricalDtype(),
    str(Variables.AB_AVEN1): pd.CategoricalDtype(),
    str(Variables.AB_AVEN6): pd.CategoricalDtype(),
    str(Variables.AB_NICU): pd.CategoricalDtype(),
    str(Variables.AB_SURF): pd.CategoricalDtype(),
    str(Variables.AB_ANTI): pd.CategoricalDtype(),
    str(Variables.AB_SEIZ): pd.CategoricalDtype(),
    str(Variables.NO_ABNORM): pd.CategoricalDtype(),
    str(Variables.CA_ANEN): pd.CategoricalDtype(),
    str(Variables.CA_MNSB): pd.CategoricalDtype(),
    str(Variables.CA_CCHD): pd.CategoricalDtype(),
    str(Variables.CA_CDH): pd.CategoricalDtype(),
    str(Variables.CA_OMPH): pd.CategoricalDtype(),
    str(Variables.CA_GAST): pd.CategoricalDtype(),
    str(Variables.CA_LIMB): pd.CategoricalDtype(),
    str(Variables.CA_CLEFT): pd.CategoricalDtype(),
    str(Variables.CA_CLPAL): pd.CategoricalDtype(),
    str(Variables.DOWNS): pd.CategoricalDtype(),
    str(Variables.UCA_DOWNS): pd.CategoricalDtype(),
    str(Variables.CA_DOWN): pd.CategoricalDtype(
        categories=["C", "P", "N", "U"], ordered=False
    ),
    str(Variables.CA_DOWNS): pd.CategoricalDtype(
        categories=["C", "P", "N", "U"], ordered=False
    ),
    str(Variables.CA_DISOR): pd.CategoricalDtype(),
    str(Variables.CA_HYPO): pd.CategoricalDtype(),
    str(Variables.NO_CONGEN): pd.CategoricalDtype(),
    str(Variables.BFED): pd.CategoricalDtype(),
    str(Variables.M_HT_IN): pd.CategoricalDtype(),
    str(Variables.BMI): pd.Float32Dtype(),
    str(Variables.BMI_R): pd.CategoricalDtype(),
    str(Variables.PWGT_R): pd.CategoricalDtype(),
    str(Variables.WTGAIN): pd.CategoricalDtype(),
    str(Variables.RF_PDIAB): pd.CategoricalDtype(),
    str(Variables.RF_GDIAB): pd.CategoricalDtype(),
    str(Variables.RF_PHYPE): pd.CategoricalDtype(),
    str(Variables.RF_GHYPE): pd.CategoricalDtype(),
    str(Variables.RF_EHYPE): pd.CategoricalDtype(),
    str(Variables.RF_PPTERM): pd.CategoricalDtype(),
    str(Variables.RF_INFTR): pd.CategoricalDtype(),
    str(Variables.RF_FEDRG): pd.CategoricalDtype(),
    str(Variables.RF_ARTEC): pd.CategoricalDtype(),
    str(Variables.RF_CESAR): pd.CategoricalDtype(),
    str(Variables.RF_CESARN): pd.CategoricalDtype(),
    str(Variables.NO_RISKS): pd.CategoricalDtype(),
    str(Variables.LD_INDL): pd.CategoricalDtype(),
    str(Variables.LD_AUGM): pd.CategoricalDtype(),
    str(Variables.LD_ANES): pd.CategoricalDtype(),
    str(Variables.ME_PRES): pd.CategoricalDtype(),
    str(Variables.RDMETH_REC): pd.CategoricalDtype(),
    str(Variables.DMETH_REC): pd.CategoricalDtype(),
    str(Variables.ATTEND): pd.CategoricalDtype(),
}

COMPUTED_VARS = list(COMPUTED.keys())
IMPORTED_VARS = list(IMPORTED.keys())


def set_all_column_types(df: pd.DataFrame) -> pd.DataFrame:
    """Sets all (standard + computed) column types for the dataframe."""

    return set_computed_column_types(set_imported_column_types(df))


def ensure_imported_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensures all imported columns exist in the dataframe."""

    for col in IMPORTED_VARS:
        if col not in df.columns:
            df[col] = pd.NA

    return df


def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensures all imported columns exist in the dataframe."""

    for col in IMPORTED_VARS:
        if col not in df.columns:
            df[col] = pd.NA

    for col in COMPUTED_VARS:
        if col not in df.columns:
            df[col] = pd.NA

    return df


def set_imported_column_types(df: pd.DataFrame) -> pd.DataFrame:
    """Sets imported column types for the dataframe."""

    for col, dtype in IMPORTED.items():
        try:
            # importing 2005: TypeError: Cannot cast array data from dtype('float64') to dtype('uint16')
            if (
                dtype == pd.UInt8Dtype()
                or dtype == pd.UInt16Dtype()
                or dtype == pd.UInt32Dtype()
                or dtype == pd.UInt64Dtype()
                or dtype == pd.Int16Dtype()
                or dtype == pd.Int32Dtype()
                or dtype == pd.Int64Dtype()
            ):
                df[col] = pd.to_numeric(df[col], downcast="unsigned").astype(
                    dtype, errors="raise"
                )
            else:
                df[col] = df[col].astype(dtype, errors="raise")
        except ValueError as e:
            print(f"Warning: Could not convert column {col} to type {dtype}.")
            raise e
        except TypeError as e:
            print(f"Warning: Could not convert column {col} to type {dtype}.")
            raise e

    return df


def set_computed_column_types(df: pd.DataFrame) -> pd.DataFrame:
    """Sets computed column types for the dataframe."""

    for col, dtype in COMPUTED.items():
        try:
            df[col] = df[col].astype(dtype)
        except ValueError as e:
            print(f"Warning: Could not convert column {col} to type {dtype}.")
            raise e
        except TypeError as e:
            print(f"Warning: Could not convert column {col} to type {dtype}.")
            raise e

    return df


# ---------------------------------------------------------------------------
# Named feature sets (used by experiment configs)
# ---------------------------------------------------------------------------

NUMERIC_BASE: list[str] = [
    Variables.YEAR,
    Variables.DBWT,
    Variables.WTGAIN,
    Variables.BMI,
    Variables.MAGE_C,
    Variables.FAGECOMB,
]
"""Numeric features common to all predictor experiments."""

CATEGORICAL_BASE: list[str] = [
    Variables.BFACIL3,
    Variables.SEX,
    Variables.PRECARE,
    Variables.GESTREC10,
    Variables.RF_PDIAB,
    Variables.RF_GDIAB,
    Variables.RF_PHYPE,
    Variables.RF_GHYPE,
    Variables.RF_EHYPE,
    Variables.RF_PPTERM,
    Variables.RF_INFTR,
    Variables.RF_FEDRG,
    Variables.RF_ARTEC,
    Variables.LD_INDL,
    Variables.LD_AUGM,
    Variables.ME_PRES,
    Variables.DMETH_REC,
    Variables.APGAR5,
    Variables.APGAR10,
    Variables.AB_AVEN1,
    Variables.AB_AVEN6,
    Variables.AB_NICU,
    Variables.AB_SURF,
    Variables.AB_ANTI,
    Variables.AB_SEIZ,
    Variables.CA_ANEN,
    Variables.CA_MNSB,
    Variables.CA_CCHD,
    Variables.CA_CDH,
    Variables.CA_OMPH,
    Variables.CA_GAST,
    Variables.CA_LIMB,
    Variables.CA_CLEFT,
    Variables.CA_CLPAL,
    Variables.CA_HYPO,
    Variables.CA_DISOR,
    Variables.MEDUC,
    Variables.MRACEHISP,
    Variables.FEDUC,
    Variables.FRACEHISP,
    Variables.PAY_REC,
    Variables.WIC,
]
"""Categorical features common to predictor experiments 0009-0011."""


