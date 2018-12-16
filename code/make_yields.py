from unidecode import unidecode

import sys
import os
import pandas as pd

CLEAN_NAME = lambda s: unidecode(s).lower().translate(None, "'()/&-").strip()

REGION_1 = 'Region1'
REGION_2 = 'Region2'
CROP = 'Crop'
YEAR = 'Year'
AREA_PLANTED = 'Area planted'
AREA_HARVESTED = 'Area harvested'
PRODUCTION = 'Production'
YIELD = 'Yield'

def process_brazil():
    CROP_TRANSLATIONS = {
        "Milho (em grao)": "corn",
        "Soja (em grao)": "soybeans",
        "Trigo (em grao)": "wheat",
        "Arroz (em casca)": "rice",
        "Cana-de-acucar": "sugarcane"
    }

    def read_sheet(crop, sheet_num, field_name):
        sheet_df = pd.read_excel('./static_data_files/brazil_yields_raw/Tabela 1612 - Producao de {}.xlsx'.format(crop),
                                 sheet_name=sheet_num, header=3)
        sheet_df = sheet_df.reset_index().drop([0, 730])
        sheet_df[REGION_2] = sheet_df['index'].map(lambda s: s.split(' (')[0]).map(CLEAN_NAME)
        sheet_df = sheet_df.iloc[34:171].copy()
        sheet_df[REGION_1] = sheet_df['index'].map(lambda s: s.split(' (')[1][:-1]).map(CLEAN_NAME)
        sheet_df = sheet_df.drop('index', axis=1)
        sheet_df = sheet_df.reset_index(drop=True)
        sheet_df = sheet_df.melt(id_vars=[REGION_1, REGION_2])  # now has two columns "variable" and "value"
        sheet_df = sheet_df.rename({'variable': YEAR}, axis=1)  # keep "value" for now
        sheet_df = sheet_df[(sheet_df["value"] != '...') & (sheet_df["value"] != '-')]
        sheet_df[CROP] = CROP_TRANSLATIONS[crop]
        sheet_df[REGION_1] = 'brasil'

        sheet_df = sheet_df.rename({"value": field_name}, axis=1)

        return sheet_df

    result_df = None
    for crop in CROP_TRANSLATIONS:
        area_planted = read_sheet(crop, 0, AREA_PLANTED)
        area_harvested = read_sheet(crop, 1, AREA_HARVESTED)
        production = read_sheet(crop, 2, PRODUCTION)
        yields = read_sheet(crop, 3, YIELD)

        to_add = pd.merge(area_planted, area_harvested, on=[REGION_1, REGION_2, YEAR, CROP]) \
                   .merge(production, on=[REGION_1, REGION_2, YEAR, CROP]) \
                   .merge(yields, on=[REGION_1, REGION_2, YEAR, CROP])
        to_add[YIELD] = to_add[YIELD].map(float) / 1000.

        result_df = pd.concat([result_df, to_add], ignore_index=True)

    result_df.to_csv('./static_data_files/brazil_yields_standardized.csv', index=False)

def process_usa():
    USA_FIPS_CODES = {
        "29": u"MO", "20": u"KS", "31": u"NE", "19": u"IA", "38": u"ND", "46": u"SD",
        "27": u"MN", "5": u"AR", "17": u"IL", "18": u"IN", "39": u"OH"
    }

    CROP_TRANSLATIONS = {
        "SOYBEANS": "soybeans"
    }

    BUSHELS_IN_ONE_TON = {
    #    "soybeans": 36.74 # One metric ton of soybeans is 36.74 bushels
    }

    #ACRES_IN_ONE_HECTARE = 2.4711

    usa = pd.read_csv('./static_data_files/usa_yields_raw.csv', encoding='utf-8')
    usa = usa[['Year', 'State ANSI', 'County', 'Commodity', 'Data Item', 'Value']].copy()
    usa['Variable'] = usa['Data Item'].map(lambda s: s.split(' - ')[1])
    usa['State'] = usa['State ANSI'].map(lambda i: USA_FIPS_CODES[str(i)]).map(CLEAN_NAME)
    usa = usa[usa['County'] != 'OTHER (COMBINED) COUNTIES'].copy()
    usa['County'] = usa['County'].map(CLEAN_NAME)
    usa = usa.drop(['Data Item', 'State ANSI'], axis=1)
    usa['Value'] = usa['Value'].map(lambda s: str(s).translate(None, ",")).map(float)
    usa['Commodity'] = usa['Commodity'].map(lambda s: CROP_TRANSLATIONS[s])
    usa = usa.pivot_table(values='Value', index=['Year','County','Commodity','State'], columns='Variable')
    usa = usa.reset_index()
    usa = usa.rename({'ACRES HARVESTED': AREA_HARVESTED, 'ACRES PLANTED': AREA_PLANTED,
                      'PRODUCTION, MEASURED IN BU': PRODUCTION, 'YIELD, MEASURED IN BU / ACRE': YIELD,
                      'State': REGION_1, 'County': REGION_2, 'Commodity': CROP}, axis=1)

    #usa[YIELD] = usa[YIELD] * (ACRES_IN_ONE_HECTARE / BUSHELS_IN_ONE_TON["soybeans"]) # Perform conversion
    usa.to_csv('./static_data_files/usa_yields_standardized.csv', index=False)

def process_argentina():
    HARVEST_YEARS = ['20{:02d}/{:02d}'.format(i, i + 1) for i in range(0, 16)]
    HARVEST_YEARS_MAP = dict(zip(HARVEST_YEARS, [2000+i+1 for i in range(0, 16)]))

    CROP_TRANSLATIONS = {
        "Maiz": "corn",
        "Soja": "soybeans",
        "Trigo": "wheat"
    }

    argentina = pd.read_csv('./static_data_files/argentina_yields_raw.csv', encoding='utf-8')
    argentina = argentina[['Provincia', 'Departamento', 'Cultivo', 'Campana', 'SuperficieSembrada (Ha)',
                           'SuperficieCosecha (Ha)', 'Produccion (Tn)',  'Rendimiento (Kg/Ha)']].copy()
    argentina['Provincia'] = argentina['Provincia'].map(CLEAN_NAME)
    argentina['Departamento'] = argentina['Departamento'].map(CLEAN_NAME)
    argentina = argentina[argentina['Cultivo'].isin(CROP_TRANSLATIONS)].copy()
    argentina['Cultivo'] = argentina['Cultivo'].map(lambda s: CROP_TRANSLATIONS[s])
    argentina = argentina[argentina['Campana'].isin(HARVEST_YEARS)].copy()
    argentina['Rendimiento (Kg/Ha)'] = argentina['Rendimiento (Kg/Ha)'].map(float) / 1000
    argentina['Campana'] = argentina['Campana'].map(lambda s: HARVEST_YEARS_MAP[s])

    argentina = argentina.rename({'SuperficieCosecha (Ha)': AREA_HARVESTED, 'SuperficieSembrada (Ha)': AREA_PLANTED,
                                  'Produccion (Tn)': PRODUCTION, 'Rendimiento (Kg/Ha)': YIELD,
                                  'Provincia': REGION_1, 'Departamento': REGION_2, 'Cultivo': CROP,
                                  'Campana': YEAR}, axis=1)
    argentina.to_csv('./static_data_files/argentina_yields_standardized.csv', index=False)

def process_india():
    CROP_TRANSLATIONS = {
        'RICE': 'rice',
        'MAIZE': 'corn',
        #'SOYABEAN': 'soybeans',
        #'POTATO': 'potatoes',
        #'WHEAT': 'wheat'
    }

    india = pd.read_csv('./static_data_files/india_yields_raw.csv', encoding='utf-8')
    india['STATE'] = india['STATE'].map(CLEAN_NAME)
    india['DISTRICT'] = india['DISTRICT'].map(CLEAN_NAME)
    india = india[india['YIELD'].notna()].copy()
    india = india[india['CROP'].isin(CROP_TRANSLATIONS)]
    india['CROP'] = india['CROP'].map(lambda s: CROP_TRANSLATIONS[s])

    india = india.rename({'CROP': CROP, 'YIELD': YIELD, 'YEAR': YEAR, 'AREA': AREA_PLANTED,
                          'STATE': REGION_1, 'DISTRICT': REGION_2, 'PRODUCTION': PRODUCTION}, axis=1)
    india = india.drop(['YIELD.RM.OUT', 'Unnamed: 0'], axis=1)

    india_total = india[india['SEASON'] == 'TOTAL '].copy().drop('SEASON', axis=1)
    india_kharif = india[india['SEASON'] == 'KHARIF'].copy().drop('SEASON', axis=1)
    india_rabi = india[india['SEASON'] == 'RABI'].copy().drop('SEASON', axis=1)

    india_total.to_csv('./static_data_files/india_yields_standardized.csv', index=False)
    india_kharif.to_csv('./static_data_files/india_kharif_yields_standardized.csv', index=False)
    india_rabi.to_csv('./static_data_files/india_rabi_yields_standardized.csv', index=False)


if __name__ == '__main__':
    name = sys.argv[1]
    if (name == 'brazil'):
        process_brazil()
    elif (name == 'usa'):
        process_usa()
    elif (name == 'argentina'):
        process_argentina()
    elif (name == 'india'):
        process_india()
