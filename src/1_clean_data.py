# use FAIR environment
import os
import pandas as pd
import numpy as np

# --- Data paths -----------------------------------------------------------
# Raw NSDUH public-use files are NOT included in this repo (too large and
# subject to SAMHSA terms of use). Point the script at them via env vars:
#   NSDUH_DATA_DIR  -- directory containing NSDUH_<year>_Tab.{txt,tsv}
#   NSDUH_CLEAN_DIR -- destination for harmonized per-year CSVs (defaults to ./data_clean)
# If NSDUH_DATA_DIR is not set, fall back to a sibling ../data_raw path
# relative to this file, which matches the expected repo layout.
_DEFAULT_RAW = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data_raw"))
_DEFAULT_CLEAN = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data_clean"))
DATA_DIR = os.environ.get("NSDUH_DATA_DIR", _DEFAULT_RAW)
CLEAN_DIR = os.environ.get("NSDUH_CLEAN_DIR", _DEFAULT_CLEAN)
os.makedirs(CLEAN_DIR, exist_ok=True)
# --------------------------------------------------------------------------


# load data
#year = 2023   # change it here, which year do you want? [2011 - 2023]
save_clean_data = True  # save the cleaned data or not
save_statistics_table_seperate_years = True  # save the cleaned data or not

for year in range(2013, 2024):
    if year in [2019, 2020, 2021, 2022, 2023]:
        read_file = os.path.join(DATA_DIR, f"NSDUH_{year}_Tab.txt")
    if year in [2013, 2014, 2015, 2016, 2017, 2018]:
        read_file = os.path.join(DATA_DIR, f"NSDUH_{year}_Tab.tsv")
    df = pd.read_csv(read_file, sep="\t")

    
 
    # new selected predictors
    if year in [2023, 2022]:
        df['ADSUITPAYR'] = df.apply(
            lambda x: 1 if x['ADSUITPAYR'] == 1 
            else 0 if x['ADSUITPAYR'] == 2  
            else np.nan, axis=1)
        
        df['Health_Coverage'] = df.apply(
            lambda x: 1 if x['IRPRVHLT'] == 1  # 1 = Private insurance
            else 2 if x['IRMCDCHP'] == 1  # 2 = Medicaid/CHIP
            else 3 if x['IRMEDICR'] == 1    # 3 = Medicare
            else 4 if x['IROTHHLT'] == 2  # 5 = No insurance
            else 5, axis=1)  # other insurance
            
        df['STMWYNORX'] = df.apply(
            lambda x: 1 if x['STMWYNORX'] == 1  # 1 = Yes
            else 0, axis=1 # no
        )

        df['SEDWYNORX'] = df.apply(
            lambda x: 1 if x['SEDWYNORX'] == 1  # 1 = Yes
            else 0, axis=1 # no
        )

        df['IRIMPRESP'] = df.apply(
            lambda x: 1 if x['IRIMPRESP'] in [2,3,4,5]   # 1 = Yes
            else 0, axis=1 # no
            )
      

        df['AD_MDEA6'] = df.apply(
            lambda x: 1 if x['AD_MDEA6'] == 1  # 1 = Yes
            else 0, axis=1 # no
        )

        df['AD_MDEA7'] = df.apply(
            lambda x: 1 if x['AD_MDEA7'] == 1  # 1 = Yes
            else 0, axis=1 # no
        )


        df['IRINSUR4'] = df.apply(
            lambda x: 1 if x['IRINSUR4'] == 1   # 1 = Yes
            else 0, axis=1 # no
            )
        
        df['AMDEYR'] = df.apply(
            lambda x: 1 if x['AMDEYR'] == 1   # 1 = Yes
            else 0, axis=1 # no
            )
        
        df['RCVSUTOMHT'] = df.apply(
            lambda x: 1 if x['RCVSUTOMHT'] == 1   # 1 = Yes
            else 0, axis=1 # no
            )

        df_selected = df[['QUESTID2','CATAG6', 'IRSEX', 'NEWRACE2', 'IRMARIT', 'EDUHIGHCAT', 'POVERTY3', 'UD5ILALANY', 'SPDPSTMON', 'SPDPSTYR', 'RCVSUTOMHT', 'ADSUITPAYR', 'BMI2', 'AMDEYR', 'IRWRKSTAT18', 'INCOME', 'Health_Coverage','COUTYP4', 'IRPYUD5ALC','IRPYUD5MRJ', 'STMWYNORX', 'SEDWYNORX','IRPYUD5COC', 'IRPYUD5HER', 'IRPYUD5HAL', 'IRPYUD5INH', 'OXYCNANYYR', 'UD5ILLANY', 'BNGDRKMON', 'HVYDRKMON', 'IRINSUR4', 'IRIMPRESP', 'AD_MDEA6', 'AD_MDEA7', 'IRSUICTHNK', 'IRSUIPLANYR', 'IRSUITRYYR']]
        
    elif year in [2021]:  # does not have RCVSUTOMHT ; use RCVMHOSPTX4  instead 
        df['RCVSUTOMHT'] = df['RCVMHOSPTX4'].apply(lambda x: 0 if x == 0 else (1 if x == 1 else x))
        df['ADSUITPAYR'] = df.apply(
            lambda x: 1 if x['ADSUITPAYR'] == 1  
            else 0 if x['ADSUITPAYR'] == 2  
            else np.nan, axis=1)

        df['Health_Coverage'] = df.apply(
            lambda x: 1 if x['IRPRVHLT'] == 1  # 1 = Private insurance
            else 2 if x['IRMCDCHP'] == 1  # 2 = Medicaid/CHIP
            else 3 if x['IRMEDICR'] == 1    # 3 = Medicare
            else 4 if x['IROTHHLT'] == 2  # 5 = No insurance
            else 5, axis=1)  # other insurance
            
        df['STMWYNORX'] = df.apply(
            lambda x: 1 if x['STMWYNORX'] == 1  # 1 = Yes
            else 0, axis=1 # no
        )

        df['SEDWYNORX'] = df.apply(
            lambda x: 1 if x['SEDWYNORX'] == 1  # 1 = Yes
            else 0, axis=1 # no
        )

        df['IRIMPRESP'] = df.apply(
            lambda x: 1 if x['IRIMPRESP'] in [2,3,4,5]   # 1 = Yes
            else 0, axis=1 # no
            )
        

        df['AD_MDEA6'] = df.apply(
            lambda x: 1 if x['AD_MDEA6'] == 1  # 1 = Yes
            else 0, axis=1 # no
        )

        df['AD_MDEA7'] = df.apply(
            lambda x: 1 if x['AD_MDEA7'] == 1  # 1 = Yes
            else 0, axis=1 # no
        )


        df['IRINSUR4'] = df.apply(
            lambda x: 1 if x['IRINSUR4'] == 1   # 1 = Yes
            else 0, axis=1 # no
            )
        
        df['AMDEYR'] = df.apply(
            lambda x: 1 if x['AMDEYR'] == 1   # 1 = Yes
            else 0, axis=1 # no
            )
        
        df_selected = df[['QUESTID2','CATAG6', 'IRSEX', 'NEWRACE2', 'IRMARIT', 'EDUHIGHCAT', 'POVERTY3', 'UD5ILALANY', 'SPDPSTMON', 'SPDPSTYR', 'RCVSUTOMHT', 'ADSUITPAYR', 'BMI2', 'AMDEYR', 'IRWRKSTAT18', 'INCOME', 'Health_Coverage','COUTYP4', 'IRPYUD5ALC','IRPYUD5MRJ', 'STMWYNORX', 'SEDWYNORX','IRPYUD5COC', 'IRPYUD5HER', 'IRPYUD5HAL', 'IRPYUD5INH', 'OXYCNANYYR', 'UD5ILLANY', 'BNGDRKMON', 'HVYDRKMON', 'IRINSUR4', 'IRIMPRESP', 'AD_MDEA6', 'AD_MDEA7', 'IRSUICTHNK', 'IRSUIPLANYR', 'IRSUITRYYR']]
        # add RCVSUTOMHT
        
        print(df_selected['RCVSUTOMHT'].value_counts())
    elif year in [2020]:
        df['UD5ILALANY'] = df['UDYR5ILAL']
        df['SPDPSTMON'] = df['SPDMON']
        df['SPDPSTYR'] = df['SPDYR']
        df['RCVSUTOMHT'] = df.apply(
            lambda x: 1 if x['AMHTXRC3'] == 1 or x['TXYRALDGB'] in [1, 2] 
            else np.nan if (x['AMHTXRC3'] == np.nan or x['TXYRALDGB'] == np.nan) 
            else 0, axis=1)
        df['ADSUITPAYR'] = df.apply(
            lambda x: 1 if (x['MHSUITHK'] == 1 or x['MHSUIPLN'] == 1 or x['MHSUITRY'] == 1) 
            else 0 if (x['MHSUITHK'] == 0 and x['MHSUIPLN'] == 0 and x['MHSUITRY'] == 0)
            else np.nan, axis=1)
        
        df['Health_Coverage'] = df.apply(
            lambda x: 1 if x['IRPRVHLT'] == 1  # 1 = Private insurance
            else 2 if x['IRMCDCHP'] == 1  # 2 = Medicaid/CHIP
            else 3 if x['IRMEDICR'] == 1    # 3 = Medicare
            else 4 if x['IROTHHLT'] == 2  # 5 = No insurance
            else 5, axis=1)  # other insurance
            
        df['STMWYNORX'] = df.apply(
            lambda x: 1 if x['STMWYNORX'] == 1  # 1 = Yes
            else 0, axis=1 # no
        )

        df['SEDWYNORX'] = df.apply(
            lambda x: 1 if x['SEDWYNORX'] == 1  # 1 = Yes
            else 0, axis=1 # no
        )

        df['IRIMPRESP'] = df.apply(
            lambda x: 1 if x['IMPRESP'] in [2,3,4,5]   # 1 = Yes
            else 0, axis=1 # no
            )
        
        df['AD_MDEA6'] = df.apply(
            lambda x: 1 if x['AD_MDEA6'] == 1  # 1 = Yes
            else 0, axis=1 # no
        )

        df['AD_MDEA7'] = df.apply(
            lambda x: 1 if x['AD_MDEA7'] == 1  # 1 = Yes
            else 0, axis=1 # no
        )

        df['UD5ILLANY'] = df['UDYR5ILL']
        df['IRSUICTHNK'] = df['MHSUITHK']
        df['IRSUIPLANYR'] = df['MHSUIPLN']
        df['IRSUITRYYR'] = df['MHSUITRY']

        df['IRINSUR4'] = df.apply(
            lambda x: 1 if x['IRINSUR4'] == 1   # 1 = Yes
            else 0, axis=1 # no
            )
        
        df['AMDEYR'] = df.apply(
            lambda x: 1 if x['AMDEYR'] == 1   # 1 = Yes
            else 0, axis=1 # no
            )
        df_selected = df[['QUESTID2','CATAG6', 'IRSEX', 'NEWRACE2', 'IRMARIT', 'EDUHIGHCAT', 'POVERTY3', 'UD5ILALANY', 'SPDPSTMON', 'SPDPSTYR', 'RCVSUTOMHT', 'ADSUITPAYR', 'BMI2', 'AMDEYR', 'IRWRKSTAT18', 'INCOME', 'Health_Coverage','COUTYP4', 'IRPYUD5ALC','IRPYUD5MRJ', 'STMWYNORX', 'SEDWYNORX','IRPYUD5COC', 'IRPYUD5HER', 'IRPYUD5HAL', 'IRPYUD5INH', 'OXYCNANYYR', 'UD5ILLANY', 'BNGDRKMON', 'HVYDRKMON', 'IRINSUR4', 'IRIMPRESP', 'AD_MDEA6', 'AD_MDEA7', 'IRSUICTHNK', 'IRSUIPLANYR', 'IRSUITRYYR']]

    elif year in [2019, 2018, 2017, 2016]:
        df['UD5ILALANY'] = df['ILLORALC']
        df['SPDPSTMON'] = df['SPDMON']
        df['SPDPSTYR'] = df['SPDYR']
        df['RCVSUTOMHT'] = df.apply(
            lambda x: 1 if x['AMHTXRC3'] == 1 or x['TXYRALDGB'] in [1, 2] 
            else np.nan if (x['AMHTXRC3'] == np.nan or x['TXYRALDGB'] == np.nan) 
            else 0, axis=1)
        df['ADSUITPAYR'] = df.apply(
            lambda x: 1 if (x['MHSUITHK'] == 1 or x['MHSUIPLN'] == 1 or x['MHSUITRY'] == 1) 
            else 0 if (x['MHSUITHK'] == 0 and x['MHSUIPLN'] == 0 and x['MHSUITRY'] == 0)
            else np.nan, axis=1)
        
        df['Health_Coverage'] = df.apply(
            lambda x: 1 if x['IRPRVHLT'] == 1  # 1 = Private insurance
            else 2 if x['IRMCDCHP'] == 1  # 2 = Medicaid/CHIP
            else 3 if x['IRMEDICR'] == 1    # 3 = Medicare
            else 4 if x['IROTHHLT'] == 2  # 5 = No insurance
            else 5, axis=1)  # other insurance
            
        df['STMWYNORX'] = df.apply(
            lambda x: 1 if x['STMWYNORX'] == 1  # 1 = Yes
            else 0, axis=1 # no
        )

        df['SEDWYNORX'] = df.apply(
            lambda x: 1 if x['SEDWYNORX'] == 1  # 1 = Yes
            else 0, axis=1 # no
        )

        df['IRIMPRESP'] = df.apply(
            lambda x: 1 if x['IMPRESP'] in [2,3,4,5]   # 1 = Yes
            else 0, axis=1 # no
            )
        
        df['AD_MDEA6'] = df.apply(
            lambda x: 1 if x['AD_MDEA6'] == 1  # 1 = Yes
            else 0, axis=1 # no
        )

        df['AD_MDEA7'] = df.apply(
            lambda x: 1 if x['AD_MDEA7'] == 1  # 1 = Yes
            else 0, axis=1 # no
        )
        df['UD5ILLANY'] = df['ILLYR']
        df['IRSUICTHNK'] = df['MHSUITHK']
        df['IRSUIPLANYR'] = df['MHSUIPLN']
        df['IRSUITRYYR'] = df['MHSUITRY']

        df['IRPYUD5ALC'] = df['ALCYR']
        df['IRPYUD5MRJ'] = df['MRJYR']
        df['IRPYUD5COC'] =  df['COCYR']
        df['IRPYUD5HER'] =  df['HERYR']
        df['IRPYUD5HAL'] =  df['HALLUCYR']
        df['IRPYUD5INH'] = df['INHALYR']

        df['IRINSUR4'] = df.apply(
            lambda x: 1 if x['IRINSUR4'] == 1   # 1 = Yes
            else 0, axis=1 # no
            )
        
        df['AMDEYR'] = df.apply(
            lambda x: 1 if x['AMDEYR'] == 1   # 1 = Yes
            else 0, axis=1 # no
            )

        df_selected = df[['QUESTID2','CATAG6', 'IRSEX', 'NEWRACE2', 'IRMARIT', 'EDUHIGHCAT', 'POVERTY3', 'UD5ILALANY', 'SPDPSTMON', 'SPDPSTYR', 'RCVSUTOMHT', 'ADSUITPAYR', 'BMI2', 'AMDEYR', 'IRWRKSTAT18', 'INCOME', 'Health_Coverage','COUTYP4', 'IRPYUD5ALC','IRPYUD5MRJ', 'STMWYNORX', 'SEDWYNORX','IRPYUD5COC', 'IRPYUD5HER', 'IRPYUD5HAL', 'IRPYUD5INH', 'OXYCNANYYR', 'UD5ILLANY', 'BNGDRKMON', 'HVYDRKMON', 'IRINSUR4', 'IRIMPRESP', 'AD_MDEA6', 'AD_MDEA7', 'IRSUICTHNK', 'IRSUIPLANYR', 'IRSUITRYYR']]
    
    elif year in [2015]:
        df['IRMARIT'] = df['IRMARITSTAT']
        df['UD5ILALANY'] = df['ILLORALC']
        df['SPDPSTMON'] = df['SPDMON']
        df['SPDPSTYR'] = df['SPDYR']
        df['RCVSUTOMHT'] = df.apply(
            lambda x: 1 if x['AMHTXRC3'] == 1 or x['TXYRALDGB'] in [1, 2] 
            else np.nan if (x['AMHTXRC3'] == np.nan or x['TXYRALDGB'] == np.nan) 
            else 0, axis=1)
        df['ADSUITPAYR'] = df.apply(
            lambda x: 1 if (x['MHSUITHK'] == 1 or x['MHSUIPLN'] == 1 or x['MHSUITRY'] == 1) 
            else 0 if (x['MHSUITHK'] == 0 and x['MHSUIPLN'] == 0 and x['MHSUITRY'] == 0)
            else np.nan, axis=1)
        
        df['Health_Coverage'] = df.apply(
            lambda x: 1 if x['IRPRVHLT'] == 1  # 1 = Private insurance
            else 2 if x['IRMCDCHP'] == 1  # 2 = Medicaid/CHIP
            else 3 if x['IRMEDICR'] == 1    # 3 = Medicare
            else 4 if x['IROTHHLT'] == 2  # 5 = No insurance
            else 5, axis=1)  # other insurance
            
        df['STMWYNORX'] = df.apply(
            lambda x: 1 if x['STMWYNORX'] == 1  # 1 = Yes
            else 0, axis=1 # no
        )

        df['SEDWYNORX'] = df.apply(
            lambda x: 1 if x['SEDWYNORX'] == 1  # 1 = Yes
            else 0, axis=1 # no
        )

        df['IRIMPRESP'] = df.apply(
            lambda x: 1 if x['IMPRESP'] in [2,3,4,5]   # 1 = Yes
            else 0, axis=1 # no
            )
       

        df['AD_MDEA6'] = df.apply(
            lambda x: 1 if x['AD_MDEA6'] == 1  # 1 = Yes
            else 0, axis=1 # no
        )

        df['AD_MDEA7'] = df.apply(
            lambda x: 1 if x['AD_MDEA7'] == 1  # 1 = Yes
            else 0, axis=1 # no
        )
        
        df['UD5ILLANY'] = df['ILLYR']
        df['IRSUICTHNK'] = df['MHSUITHK']
        df['IRSUIPLANYR'] = df['MHSUIPLN']
        df['IRSUITRYYR'] = df['MHSUITRY']

        df['IRPYUD5ALC'] = df['ALCYR']
        df['IRPYUD5MRJ'] = df['MRJYR']
        df['IRPYUD5COC'] =  df['COCYR']
        df['IRPYUD5HER'] =  df['HERYR']
        df['IRPYUD5HAL'] =  df['HALLUCYR']
        df['IRPYUD5INH'] = df['INHALYR']

        df['IRINSUR4'] = df.apply(
            lambda x: 1 if x['IRINSUR4'] == 1   # 1 = Yes
            else 0, axis=1 # no
            )
        
        df['AMDEYR'] = df.apply(
            lambda x: 1 if x['AMDEYR'] == 1   # 1 = Yes
            else 0, axis=1 # no
            )

        df_selected = df[['QUESTID2','CATAG6', 'IRSEX', 'NEWRACE2', 'IRMARIT', 'EDUHIGHCAT', 'POVERTY3', 'UD5ILALANY', 'SPDPSTMON', 'SPDPSTYR', 'RCVSUTOMHT', 'ADSUITPAYR', 'BMI2', 'AMDEYR', 'IRWRKSTAT18', 'INCOME', 'Health_Coverage','COUTYP4', 'IRPYUD5ALC','IRPYUD5MRJ', 'STMWYNORX', 'SEDWYNORX','IRPYUD5COC', 'IRPYUD5HER', 'IRPYUD5HAL', 'IRPYUD5INH', 'OXYCNANYYR', 'UD5ILLANY', 'BNGDRKMON', 'HVYDRKMON', 'IRINSUR4', 'IRIMPRESP', 'AD_MDEA6', 'AD_MDEA7', 'IRSUICTHNK', 'IRSUIPLANYR', 'IRSUITRYYR']]
    
    elif year in [2014, 2013]:
        df['UD5ILALANY'] = df['ILORALC']
        df['SPDPSTMON'] = df['SPDMON']
        df['SPDPSTYR'] = df['SPDYR']
        df['EDUHIGHCAT'] = df["EDUCCAT2"]
        df['POVERTY3'] = df["POVERTY2"]
        df['RCVSUTOMHT'] = df.apply(
            lambda x: 1 if x['AMHTXRC3'] == 1 or x['TXYRADG'] in [1, 2] 
            else np.nan if (x['AMHTXRC3'] == np.nan or x['TXYRADG'] == np.nan) 
            else 0, axis=1)
        df['ADSUITPAYR'] = df.apply(
            lambda x: 1 if (x['MHSUITHK'] == 1 or x['MHSUIPLN'] == 1 or x['MHSUITRY'] == 1) 
            else 0 if (x['MHSUITHK'] == 0 and x['MHSUIPLN'] == 0 and x['MHSUITRY'] == 0)
            else np.nan, axis=1)
        
        df['Health_Coverage'] = df.apply(
            lambda x: 1 if x['IRPRVHLT'] == 1  # 1 = Private insurance
            else 2 if x['IRMCDCHP'] == 1  # 2 = Medicaid/CHIP
            else 3 if x['IRMEDICR'] == 1    # 3 = Medicare
            else 4 if x['IROTHHLT'] == 2  # 5 = No insurance
            else 5, axis=1)  # other insurance
            
        df['STMWYNORX'] = df.apply(
            lambda x: 1 if x['STDAYPYR'] in range(1,366)  # 1 = Yes
            else 0, axis=1 # no
        )

        df['SEDWYNORX'] = df.apply(
            lambda x: 1 if x['SEDYRTOT'] in range(1,366)  # 1 = Yes
            else 0, axis=1 # no
        )

        df['IRIMPRESP'] = df.apply(
            lambda x: 1 if x['IMPRESP'] in [2,3,4,5]   # 1 = Yes
            else 0, axis=1 # no
            )
        
        df['AD_MDEA6'] = df.apply(
            lambda x: 1 if x['AD_MDEA6'] == 1  # 1 = Yes
            else 0, axis=1 # no
        )

        df['AD_MDEA7'] = df.apply(
            lambda x: 1 if x['AD_MDEA7'] == 1  # 1 = Yes
            else 0, axis=1 # no
        )
        df['OXYCNANYYR'] = df['OXYYR']
        df['UD5ILLANY'] = df['SUMYR']
        df['IRSUICTHNK'] = df['MHSUITHK']
        df['IRSUIPLANYR'] = df['MHSUIPLN']
        df['IRSUITRYYR'] = df['MHSUITRY']

        df['IRPYUD5ALC'] = df['ALCYR']
        df['IRPYUD5MRJ'] = df['MRJYR']
        df['IRPYUD5COC'] =  df['COCYR']
        df['IRPYUD5HER'] =  df['HERYR']
        df['IRPYUD5HAL'] =  df['HALYR']
        df['IRPYUD5INH'] = df['INHYR']

        df['IRWRKSTAT18'] = df['EMPSTAT4']
        df['COUTYP4'] = df['COUTYP2']
        df['BNGDRKMON'] = df['BINGEDRK']
        df['HVYDRKMON'] = df['HVYDRK2']

        df['IRINSUR4'] = df.apply(
            lambda x: 1 if x['IRINSUR4'] == 1   # 1 = Yes
            else 0, axis=1 # no
            )
        
        df['AMDEYR'] = df.apply(
            lambda x: 1 if x['AMDEYR'] == 1   # 1 = Yes
            else 0, axis=1 # no
            )

        df_selected = df[['QUESTID2','CATAG6', 'IRSEX', 'NEWRACE2', 'IRMARIT', 'EDUHIGHCAT', 'POVERTY3', 'UD5ILALANY', 'SPDPSTMON', 'SPDPSTYR', 'RCVSUTOMHT', 'ADSUITPAYR', 'BMI2', 'AMDEYR', 'IRWRKSTAT18', 'INCOME', 'Health_Coverage','COUTYP4', 'IRPYUD5ALC','IRPYUD5MRJ', 'STMWYNORX', 'SEDWYNORX','IRPYUD5COC', 'IRPYUD5HER', 'IRPYUD5HAL', 'IRPYUD5INH', 'OXYCNANYYR', 'UD5ILLANY', 'BNGDRKMON', 'HVYDRKMON', 'IRINSUR4', 'IRIMPRESP', 'AD_MDEA6', 'AD_MDEA7', 'IRSUICTHNK', 'IRSUIPLANYR', 'IRSUITRYYR']]


    # --- NSDUH survey design variables (year-specific; see 2023 PUF data users' guide) ---
    # Weight variable name changed with NSDUH methodological redesigns:
    #   2013-2019: ANALWT_C  (in-person interviews only)
    #   2020    : ANALWTQ1Q4_C  (COVID-era weight built from Quarters 1 and 4 only)
    #   2021-2023: ANALWT2_C  (revised to accommodate web-interview mode)
    # Stratum name correspondingly changed (VESTR -> VESTRQ1Q4_C -> VESTR_C); VEREP unchanged.
    if year <= 2019:
        _weight_col, _stratum_col = 'ANALWT_C', 'VESTR'
    elif year == 2020:
        _weight_col, _stratum_col = 'ANALWTQ1Q4_C', 'VESTRQ1Q4_C'
    else:  # 2021-2023
        _weight_col, _stratum_col = 'ANALWT2_C', 'VESTR_C'
    _replicate_col = 'VEREP'

    df_selected = df_selected.copy()
    df_selected['SURVEY_WEIGHT'] = df[_weight_col].values if _weight_col in df.columns else np.nan
    df_selected['SURVEY_STRATUM'] = df[_stratum_col].values if _stratum_col in df.columns else np.nan
    df_selected['SURVEY_REPLICATE'] = df[_replicate_col].values if _replicate_col in df.columns else np.nan
    df_selected['SURVEY_WEIGHT_TYPE'] = _weight_col
    # -------------------------------------------------------------------------------

    # missing values
    print("year")
    missing_values = df_selected['ADSUITPAYR'].isnull().sum()
    print(f"Number of missing values in ADSUITPAYR: {missing_values}")
    df_selected = df_selected.dropna(subset=['ADSUITPAYR'])
   
    missing_values = df_selected['POVERTY3'].isnull().sum()
    print(f"Number of missing values in POVERTY3: {missing_values}")
    df_selected = df_selected.dropna(subset=['POVERTY3'])

    missing_values = df_selected['RCVSUTOMHT'].isnull().sum()
    print(f"Number of missing values in RCVSUTOMHT: {missing_values}")
    df_selected = df_selected.dropna(subset=['RCVSUTOMHT'])

    ################ Dictionary ###############
    
    # Age
    CATAG6_dict = {
        1 : "12-17 Years Old",
        2 : "18-25 Years Old",
        3 : "26-34 Years Old",
        4 : "35-49 Years Old",
        5 : "50-64 Years Old",
        6 : "65 or Older"
    }

    NEWRACE2_dict = {
        1 : "NonHisp White",
        2 : "NonHisp Black/Afr Am",
        3 : "NonHisp Native Am/AK Native",
        4 : "NonHisp Native HI/Other Pac Isl",
        5 : "NonHisp Asian",
        6 : "NonHisp more than",
        7 : "Hispanic"
    }
    EDUHIGHCAT_dict = {
        1 : "Less high school",
        2 : "High school grad",
        3 : "Some coll/Assoc Dg",
        4 : "College graduate",
        5 : "12 to 17 year olds"
    }
    IRSEX_dict = {
        1 : "Male", 
        2 : "Female" 
    }
    IRMARIT_dict = {
        1 : "Married",
        2 : "Widowed",
        3 : "Divorced or Separated",
        4 : "Never Been Married",
        99 : "LEGITIMATE SKIP Respondent is <= 14 years old"
    }

    # EMPLOYMENT STATUS 18+
    IRWRKSTAT18_dict = {
        1 : "Employed full time",
        2 : "Employed part time",
        3 : "Unemployed",
        4 : "Other",
        99 : "12-17 year olds"
    }

    INCOME_dict = {
        1 : "Less than $20,000",
        2 : "$20,000 - $49,999",
        3 : "$50,000 - $74,999",
        4 : "$75,000 or More"
    }

    POVERTY3_dict = {
        1 : "Living in Poverty",
        2 : "Income Up to 2X Fed Pov Thresh",
        3 : "Income More Than 2X Fed Pov Thresh"
    }

    #RC-DRUG OR ALCOHOL USE DISORDER - PAST YEAR USERS
    UD5ILALANY_dict = {
        0 : "No",
        1 : "Yes"
    }

    IRKI17_2_dict = {
        1 : "No children under 18",
        2 : "One child under 18",
        3 : "Two children under 18",
        4 : "Three or more children under 18"
    }

    IRHH65_2_dict = {
        1 : "No people 65 or older in household",
        2 : "One person 65 or older in household",
        3 : "Two or more people 65 or older in household"
    }
    #RC-PAST MONTH SERIOUS PSCYHOLOGICAL DISTRESS INDICATOR
    SPDPSTMON_dict = {
        0 : "No",
        1 : "Yes"
    }
    # RC-PAST YEAR SERIOUS PSCYHOLOGICAL DISTRESS INDICATOR
    SPDPSTYR_dict = {
        0 : "No",
        1 : "Yes"
    }

    ADSUITPACOM_dict = {    
        1 : "Thoughts only",
        2 : "Plans only",
        3 : "Attempts only",
        4 : "Thoughts/Plans only",
        5 : "Thoughts/Attempts only",
        6 : "Plans/Attempts only",
        7 : "Thoughts/Plans/Attempts",
        8 : "No suicide behaviors"
    }


    SUICTHNK_dict = {
        1 : "Yes",
        2 : "NO",
        85 : "BAD DATA Logically assigned",
        94 : "DON'T KNOW",
        97 : "REFUSED",
        98 : "BLANK (NO ANSWER)",
        99 : "LEGITIMATE SKIP"
    }

    SUIPLANYR_dict = {
        1 : "Yes",
        2 : "No",
        85 : "BAD DATA Logically assigned",
        94 : "DON'T KNOW",
        97 : "REFUSED",
        98 : "BLANK (NO ANSWER)",
        99 : "LEGITIMATE SKIP"
    }

    # RC-RCVD SUBSTANCE USE TRT OR MENTAL HEALTH TRT IN PAST YEAR
    RCVSUTOMHT_dict = {
        1 : "Yes",
        0 : "No"
    }

    # ADULT: PAST YEAR MAJOR DEPRESSIVE EPISODE (MDE)
    AMDEYR_dict = {
        1 : "Yes",
        0 : "No"
    }

    Health_Coverage_dict = {
        1 : "Private plan",
        2 : "Medicaid/CHIP",
        3 : "Medicare",
        4 : "Uninsured",
        5 : "Other"
    }

    # COUTYP4: METRO/NONMETRO STATUS (2013 3-LEVEL)
    COUTYP4_dict = {
        1 : "Large Metropolitan",
        2 : "Small Metropolitan",
        3 : "Nonmetropolitan"
    }

    # ALCOHOL USE DISORDER IN THE PAST YEAR - IMP REV
    IRPYUD5ALC_dict = {
        1 : "Yes",
        0 : "No"

    }

    # MARIJUANA USE DISORDER IN THE PAST YEAR 
    IRPYUD5MRJ_dict = {
        1 : "Yes",
        0 : "No"
    }

    # HEROIN USE DISORDER IN THE PAST YEAR
    IRPYUD5HER_dict = {
        1 : "Yes",
        0 : "No"
    }

    # HALLUCINOGEN USE DISORDER IN THE PAST YEAR
    IRPYUD5HAL_dict = {
        1 : "Yes",
        0 : "No"
    }

    # INHALANT USE DISORDER IN THE PAST YEAR
    IRPYUD5INH_dict = {
        1 : "Yes",
        0 : "No"
    }

    # USED STIMULANT W/O OWN RX PAST 12 MONTHS
    STMWYNORX_dict = {
        1 : "Yes",
        0 : "No"
    }

    # USED SEDATIVE W/O OWN RX PAST 12 MONTHS
    SEDWYNORX_dict = {
        1 : "Yes",
        0 : "No"
    }

    # COCAINE USE DISORDER IN THE PAST YEAR
    IRPYUD5COC_dict = {
        1 : "Yes",
        0 : "No"
    }

    # RC-OXYCONTIN - PAST YEAR USE
    OXYCNANYYR_dict = {
        1 : "Yes",
        0 : "No"
    }
    # RC-DRUG USE DISORDER - PAST YEAR USERS
    UD5ILLANY_dict = {
        1 : "Yes",
        0 : "No"
    }

    IRSUICTHNK_dict = {
        1 : "Yes",
        0 : "No",
    }

    IRSUIPLANYR_dict = {
        1 : "Yes",
        0 : "No",
    }

    IRSUITRYYR_dict = {
        1 : "Yes",
        0 : "No",
    }

    # RC-BINGE ALCOHOL USE PAST 30 DAYS
    BNGDRKMON_dict = {
        0 : "No",
        1 : "Yes",
    }

    # RC-HEAVY ALCOHOL USE PAST 30 DAYS
    HVYDRKMON_dict = {
        0 : "No",
        1 : "Yes",
    }
    # RC-OVERALL HEALTH INSURANCE 
    IRINSUR4_dict = {
        1 : "Yes",
        0 : "No",
    }

    # DIFFICULTY WORK RESPONS ONE MO IN PST 12 MOS
    IRIMPRESP_dict = {
        1 : "Yes",
        0 : "No"
    }


    # FELT TIRED/LOW ENERGY NEARLY EVERY DAY
    AD_MDEA6_dict = {
        1 : "Yes",
        0 : "No"
    }

    # FELT WORTHLESS NEARLY EVERY DAY
    AD_MDEA7_dict = {
        1 : "Yes",
        0 : "No"
    }



    # sharmin added this mapping outside from codebook
    if year not in [2012, 2011]:
        df_selected.loc[:, 'BMI2_decode'] = df_selected['BMI2'].apply(
            lambda x: 'Underweight' if x < 18.5 else
                    'Healthy' if 18.5 <= x <= 24.9 else
                    'Overweight' if 25 <= x <= 29.9 else
                    'Obesity' if 30 <= x <= 39.9 else
                    'Severe Obesity' if x >= 40 else
                    'Unknown'
        )

    preds = df[['CATAG6', 'IRSEX', 'NEWRACE2', 'IRMARIT', 'EDUHIGHCAT', 'POVERTY3', 'UD5ILALANY', 'SPDPSTMON', 'SPDPSTYR', 'RCVSUTOMHT', 'AMDEYR', 'IRWRKSTAT18', 'INCOME', 'Health_Coverage','COUTYP4', 'IRPYUD5ALC','IRPYUD5MRJ', 'STMWYNORX', 'SEDWYNORX','IRPYUD5COC', 'IRPYUD5HER', 'IRPYUD5HAL', 'IRPYUD5INH', 'OXYCNANYYR', 'UD5ILLANY', 'BNGDRKMON', 'HVYDRKMON', 'IRINSUR4', 'IRIMPRESP', 'AD_MDEA6', 'AD_MDEA7']]

    preds_decode = []
    for p in preds:
        p_dict = str(p) + '_dict'
        p_decode = str(p) + '_decode'
        df_selected[p_decode] = df_selected[p].map(globals()[p_dict])
        preds_decode.append(p_decode)


    if save_clean_data == True:
        df_table = pd.DataFrame(df_selected)
        # save file
        save_file = os.path.join(CLEAN_DIR, f"clean_data_{year}.csv")
        df_table.to_csv(save_file, index=False)











   