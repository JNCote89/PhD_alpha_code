from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict, fields
from typing import Union

import pandas as pd


@dataclass(slots=True)
class Labels_Age:
    Age_Tot: Union[int, str] = 'Age_Tot'
    Age_0_4: Union[int, str] = 'Age_0_4'
    Age_5_9: Union[int, str] = 'Age_5_9'
    Age_10_14: Union[int, str] = 'Age_10_14'
    Age_15_19: Union[int, str] = 'Age_15_19'
    Age_20_24: Union[int, str] = 'Age_20_24'
    Age_25_29: Union[int, str] = 'Age_25_29'
    Age_30_34: Union[int, str] = 'Age_30_34'
    Age_35_39: Union[int, str] = 'Age_35_39'
    Age_40_44: Union[int, str] = 'Age_40_44'
    Age_45_49: Union[int, str] = 'Age_45_49'
    Age_50_54: Union[int, str] = 'Age_50_54'
    Age_55_59: Union[int, str] = 'Age_55_59'
    Age_60_64: Union[int, str] = 'Age_60_64'
    Age_65_69: Union[int, str] = 'Age_65_69'
    Age_70_74: Union[int, str] = 'Age_70_74'
    Age_75_79: Union[int, str] = 'Age_75_79'
    Age_80_84: Union[int, str] = 'Age_80_84'
    Age_85_over: Union[int, str] = 'Age_85_over'
    column_prefix: str = 'Age_'

    def get_labels(self, column_prefix: bool = True) -> list[str]:
        labels = []
        for key, value in asdict(self).items():
            if key != 'column_prefix':
                if column_prefix:
                    labels.append(key)
                else:
                    labels.append(key.strip(self.column_prefix))
        return labels


@dataclass(slots=True)
class Labels_IDs_Age_M:
    Age_M_tot: Union[int, str]
    Age_M_0_4: Union[int, str]
    Age_M_5_9: Union[int, str]
    Age_M_10_14: Union[int, str]
    Age_M_15_19: Union[int, str]
    Age_M_20_24: Union[int, str]
    Age_M_25_29: Union[int, str]
    Age_M_30_34: Union[int, str]
    Age_M_35_39: Union[int, str]
    Age_M_40_44: Union[int, str]
    Age_M_45_49: Union[int, str]
    Age_M_50_54: Union[int, str]
    Age_M_55_59: Union[int, str]
    Age_M_60_64: Union[int, str]
    Age_M_65_69: Union[int, str]
    Age_M_70_74: Union[int, str]
    Age_M_75_79: Union[int, str]
    Age_M_80_84: Union[int, str]
    Age_M_85_over: Union[int, str]
    column_prefix: str = 'Age_M_'


@dataclass(slots=True)
class Labels_IDs_Age_F:
    Age_F_tot: Union[int, str]
    Age_F_0_4: Union[int, str]
    Age_F_5_9: Union[int, str]
    Age_F_10_14: Union[int, str]
    Age_F_15_19: Union[int, str]
    Age_F_20_24: Union[int, str]
    Age_F_25_29: Union[int, str]
    Age_F_30_34: Union[int, str]
    Age_F_35_39: Union[int, str]
    Age_F_40_44: Union[int, str]
    Age_F_45_49: Union[int, str]
    Age_F_50_54: Union[int, str]
    Age_F_55_59: Union[int, str]
    Age_F_60_64: Union[int, str]
    Age_F_65_69: Union[int, str]
    Age_F_70_74: Union[int, str]
    Age_F_75_79: Union[int, str]
    Age_F_80_84: Union[int, str]
    Age_F_85_over: Union[int, str]
    column_prefix: str = 'Age_F_'


@dataclass(slots=True)
class Labels_IDs_Household:
    Household_Tot: Union[int, str, None]
    Household_One_person: Union[int, str, None]
    Household_Lone_parent: Union[int, str, None]
    Household_Renter: Union[int, str, None]
    column_prefix: str = 'Household_'


@dataclass(slots=True)
class Labels_IDs_Population:
    # The population is a precise count and the age has rounding to it.
    Pop_Tot: Union[int, str, None]
    Pop_Not_in_labour: Union[int, str, None]
    Pop_Unemployed: Union[int, str, None]
    Pop_Employed: Union[int, str, None]
    Pop_No_official_language: Union[int, str, None]
    Pop_Recent_immigrant: Union[int, str, None]
    Pop_Aboriginal: Union[int, str, None]
    column_prefix: str = 'Pop_'


@dataclass(slots=True)
class Labels_IDs_LabourIndustry:
    """
    Total Labour Force population aged 15 years and over by Industry - North American Industry Classification System
    (NAICS) 2012 - 25% sample data (226
    """
    Pop_Agriculture_11: Union[int, str, None]
    Pop_Mining_oil_21: Union[int, str, None]
    Pop_Utilities_22: Union[int, str, None]
    Pop_Construction_23: Union[int, str, None]
    Pop_Manufacturing_31_33: Union[int, str, None]
    Pop_Wholesale_41: Union[int, str, None]
    Pop_Retail_44_45: Union[int, str, None]
    Pop_Transport_warehousing_48_49: Union[int, str, None]
    Pop_Information_cultural_51: Union[int, str, None]
    Pop_Finance_52: Union[int, str, None]
    Pop_Real_estate_53: Union[int, str, None]
    Pop_Professional_scientific_54: Union[int, str, None]
    Pop_Management_55: Union[int, str, None]
    Pop_Waste_management_56: Union[int, str, None]
    Pop_Education_61: Union[int, str, None]
    Pop_Health_care_62: Union[int, str, None]
    Pop_Art_71: Union[int, str, None]
    Pop_Food_service_72: Union[int, str, None]
    Pop_Other_81: Union[int, str, None]
    Pop_Public_administration_91: Union[int, str, None]
    column_prefix: str = 'Pop_'


@dataclass(slots=True)
class Labels_IDs_Housing:
    Household_Detached: Union[int, str, None]
    Household_Apartment_5_storeys_over: Union[int, str, None]
    Household_Semi_detached: Union[int, str, None]
    Household_Row: Union[int, str, None]
    Household_Duplex: Union[int, str, None]
    Household_Apartment_4_storeys_less: Union[int, str, None]
    Household_Other: Union[int, str, None]
    Household_Movable: Union[int, str, None]
    Household_Major_repair: Union[int, str, None]
    column_prefix: str = 'Household_'


@dataclass(slots=True)
class Labels_IDs_HouseAge:
    Household_1960_before: Union[list[int], None]
    Household_1961_1980: Union[list[int], None]
    Household_1981_2000: Union[list[int], None]
    Household_2001_2005: Union[list[int], None]
    Household_2006_2010: Union[list[int], None]
    Household_2011_2015: Union[list[int], None]
    Household_2016_2020: Union[list[int], None]
    column_prefix: str = 'Household_'


@dataclass(slots=True)
class Labels_IDs_Socioeconomic:
    Pop_No_degree: Union[list[int], None]
    Pop_Lico_at: Union[list[int], None]
    Household_More_30_shelter_cost: Union[list[int], None]


class AbstractCensus_Labels_IDs(ABC):

    @property
    @abstractmethod
    def age_M(self) -> Labels_IDs_Age_M:
        raise NotImplementedError

    @property
    @abstractmethod
    def age_F(self) -> Labels_IDs_Age_F:
        raise NotImplementedError

    @property
    @abstractmethod
    def household(self) -> Labels_IDs_Household:
        raise NotImplementedError

    @property
    @abstractmethod
    def population(self) -> Labels_IDs_Population:
        raise NotImplementedError

    @property
    @abstractmethod
    def labour_industry(self) -> Labels_IDs_LabourIndustry:
        raise NotImplementedError

    @property
    @abstractmethod
    def housing(self) -> Labels_IDs_Housing:
        raise NotImplementedError

    @property
    @abstractmethod
    def house_age(self) -> Labels_IDs_HouseAge:
        raise NotImplementedError

    @property
    @abstractmethod
    def socioeconomic(self) -> Labels_IDs_Socioeconomic:
        raise NotImplementedError

    @property
    def column_age_tot_prefix(self) -> str:
        return Labels_Age().Age_Tot

    def get_age_M_labels_ids(self) -> tuple[list[str], list[int | str | None]]:
        labels = []
        ids = []
        for key, value in asdict(self.age_M).items():
            if key != 'column_prefix':
                labels.append(key)
                ids.append(value)

        return labels, ids

    def get_age_F_labels_ids(self) -> tuple[list[str], list[int | str | None]]:
        labels = []
        ids = []
        for key, value in asdict(self.age_F).items():
            if key != 'column_prefix':
                labels.append(key)
                ids.append(value)

        return labels, ids

    def get_one_to_one_labels_ids(self) -> tuple[list[str], list[int | str | None]]:
        concat_dict = (asdict(self.household) | asdict(self.population) | asdict(self.labour_industry) |
                       asdict(self.housing))
        labels = []
        ids = []
        for key, value in concat_dict.items():
            if key != 'column_prefix':
                labels.append(key)
                ids.append(value)

        return labels, ids

    def get_one_to_many_labels_ids(self) -> tuple[list[str], list[int | str | None]]:
        concat_dict = asdict(self.house_age) | asdict(self.socioeconomic)
        labels = []
        ids = []

        for key, value in concat_dict.items():
            if key != 'column_prefix':
                labels.append(key)
                ids.extend(value)

        return labels, ids

    def get_variable_name(self, variable: str) -> str:
        # There is no easy way to retrieve a variable name without iterating through fields or using ugly undocumented
        # hack meddling in internal state such as using __dataclass_fields__. This method make sure we call an
        # existing variable name and return its string representation to be used in dataframe
        # as a column selector for example.
        concat_dict = (asdict(self.age_M) | asdict(self.age_F) | asdict(self.household) | asdict(self.population) |
                       asdict(self.labour_industry) | asdict(self.housing) | asdict(self.house_age) |
                       asdict(self.socioeconomic))
        for key in concat_dict.keys():
            if key == variable:
                return key

    def df_age_threshold_tot(self, df_age_f: pd.DataFrame, df_age_m: pd.DataFrame) -> pd.DataFrame:
        df_age_F_raw = df_age_f.copy()
        df_age_M_raw = df_age_m.copy()

        df_age_F_raw.columns = [col_name.removeprefix(self.age_F.column_prefix) for col_name in
                                df_age_F_raw.columns]
        df_age_M_raw.columns = [col_name.removeprefix(self.age_M.column_prefix) for col_name in
                                df_age_M_raw.columns]
        df_age_tot = df_age_F_raw.add(df_age_M_raw)

        return df_age_tot.add_prefix(self.column_age_tot_prefix)


class Census_2001_Labels_IDs(AbstractCensus_Labels_IDs):
    """
    Column common labels and specific ids from the Statistics Canada (2001) 2001 Census Profile,
    95F0495XCB2001002.xml [XML] file.
    """

    @property
    def age_M(self) -> Labels_IDs_Age_M:
        return Labels_IDs_Age_M(Age_M_tot=3,
                                Age_M_0_4=4,
                                Age_M_5_9=5,
                                Age_M_10_14=6,
                                Age_M_15_19=7,
                                Age_M_20_24=8,
                                Age_M_25_29=9,
                                Age_M_30_34=10,
                                Age_M_35_39=11,
                                Age_M_40_44=12,
                                Age_M_45_49=13,
                                Age_M_50_54=14,
                                Age_M_55_59=15,
                                Age_M_60_64=16,
                                Age_M_65_69=17,
                                Age_M_70_74=18,
                                Age_M_75_79=19,
                                Age_M_80_84=20,
                                Age_M_85_over=21)

    @property
    def age_F(self) -> Labels_IDs_Age_F:
        return Labels_IDs_Age_F(Age_F_tot=22,
                                Age_F_0_4=23,
                                Age_F_5_9=24,
                                Age_F_10_14=25,
                                Age_F_15_19=26,
                                Age_F_20_24=27,
                                Age_F_25_29=28,
                                Age_F_30_34=29,
                                Age_F_35_39=30,
                                Age_F_40_44=31,
                                Age_F_45_49=32,
                                Age_F_50_54=33,
                                Age_F_55_59=34,
                                Age_F_60_64=35,
                                Age_F_65_69=36,
                                Age_F_70_74=37,
                                Age_F_75_79=38,
                                Age_F_80_84=39,
                                Age_F_85_over=40)

    @property
    def household(self) -> Labels_IDs_Household:
        return Labels_IDs_Household(Household_Tot=118,
                                    Household_One_person=119,
                                    Household_Lone_parent=64,
                                    Household_Renter=1447)

    @property
    def population(self) -> Labels_IDs_Population:
        return Labels_IDs_Population(Pop_Tot=1,
                                     Pop_Not_in_labour=736,
                                     Pop_Unemployed=735,
                                     Pop_Employed=734,
                                     Pop_No_official_language=214,
                                     Pop_Recent_immigrant=504,
                                     Pop_Aboriginal=715)

    @property
    def labour_industry(self) -> Labels_IDs_LabourIndustry:
        return Labels_IDs_LabourIndustry(Pop_Agriculture_11=1167,
                                         Pop_Mining_oil_21=1168,
                                         Pop_Utilities_22=1169,
                                         Pop_Construction_23=1170,
                                         Pop_Manufacturing_31_33=1171,
                                         Pop_Wholesale_41=1172,
                                         Pop_Retail_44_45=1173,
                                         Pop_Transport_warehousing_48_49=1174,
                                         Pop_Information_cultural_51=1175,
                                         Pop_Finance_52=1176,
                                         Pop_Real_estate_53=1177,
                                         Pop_Professional_scientific_54=1178,
                                         Pop_Management_55=1179,
                                         Pop_Waste_management_56=1180,
                                         Pop_Education_61=1181,
                                         Pop_Health_care_62=1183,
                                         Pop_Art_71=1183,
                                         Pop_Food_service_72=1184,
                                         Pop_Other_81=1185,
                                         Pop_Public_administration_91=1186)

    @property
    def housing(self) -> Labels_IDs_Housing:
        return Labels_IDs_Housing(Household_Detached=110,
                                  Household_Apartment_5_storeys_over=114,
                                  Household_Semi_detached=111,
                                  Household_Row=112,
                                  Household_Duplex=113,
                                  Household_Apartment_4_storeys_less=115,
                                  Household_Other=116,
                                  Household_Movable=117,
                                  Household_Major_repair=101)

    # 102 = Household before 1946, 103 = Household_46_60, 104 = Household_61_70, 105 = Household_71_80,
    # 106 = Household_81_90, 107 = Household_91_95, 108 = Household_96_00
    @property
    def house_age(self) -> Labels_IDs_HouseAge:
        return Labels_IDs_HouseAge(Household_1960_before=[102, 103],
                                   Household_1961_1980=[104, 105],
                                   Household_1981_2000=[106, 107, 108],
                                   Household_2001_2005=[],
                                   Household_2006_2010=[],
                                   Household_2011_2015=[],
                                   Household_2016_2020=[])

    # 1382 = Less than 9th grade, 1384 = without high school certificate (but it excludes those without a 9th grade)
    # 1449 = renter spending more than 30% of their income on dwelling, 1453 = owner spending more than 30% on dwelling
    @property
    def socioeconomic(self) -> Labels_IDs_Socioeconomic:
        return Labels_IDs_Socioeconomic(Pop_No_degree=[1382, 1384],
                                        Pop_Lico_at=[1444],
                                        Household_More_30_shelter_cost=[1449, 1453])


class Census_2006_Labels_IDs(AbstractCensus_Labels_IDs):
    """
    Column common labels and specific ids from the Statistics Canada (2006) 2006 Census Profile,
    94-581-XCB2006002.xml [XML] file.
    """

    @property
    def age_M(self) -> Labels_IDs_Age_M:
        return Labels_IDs_Age_M(Age_M_tot=3,
                                Age_M_0_4=4,
                                Age_M_5_9=5,
                                Age_M_10_14=6,
                                Age_M_15_19=7,
                                Age_M_20_24=8,
                                Age_M_25_29=9,
                                Age_M_30_34=10,
                                Age_M_35_39=11,
                                Age_M_40_44=12,
                                Age_M_45_49=13,
                                Age_M_50_54=14,
                                Age_M_55_59=15,
                                Age_M_60_64=16,
                                Age_M_65_69=17,
                                Age_M_70_74=18,
                                Age_M_75_79=19,
                                Age_M_80_84=20,
                                Age_M_85_over=21)

    @property
    def age_F(self) -> Labels_IDs_Age_F:
        return Labels_IDs_Age_F(Age_F_tot=22,
                                Age_F_0_4=23,
                                Age_F_5_9=24,
                                Age_F_10_14=25,
                                Age_F_15_19=26,
                                Age_F_20_24=27,
                                Age_F_25_29=28,
                                Age_F_30_34=29,
                                Age_F_35_39=30,
                                Age_F_40_44=31,
                                Age_F_45_49=32,
                                Age_F_50_54=33,
                                Age_F_55_59=34,
                                Age_F_60_64=35,
                                Age_F_65_69=36,
                                Age_F_70_74=37,
                                Age_F_75_79=38,
                                Age_F_80_84=39,
                                Age_F_85_over=40)

    @property
    def household(self) -> Labels_IDs_Household:
        return Labels_IDs_Household(Household_Tot=128,
                                    Household_One_person=129,
                                    Household_Lone_parent=69,
                                    Household_Renter=2049)

    @property
    def population(self) -> Labels_IDs_Population:
        return Labels_IDs_Population(Pop_Tot=1,
                                     Pop_Not_in_labour=579,
                                     Pop_Unemployed=578,
                                     Pop_Employed=577,
                                     Pop_No_official_language=247,
                                     Pop_Recent_immigrant=553,
                                     Pop_Aboriginal=565)

    @property
    def labour_industry(self) -> Labels_IDs_LabourIndustry:
        return Labels_IDs_LabourIndustry(Pop_Agriculture_11=1010,
                                         Pop_Mining_oil_21=1011,
                                         Pop_Utilities_22=1012,
                                         Pop_Construction_23=1013,
                                         Pop_Manufacturing_31_33=1014,
                                         Pop_Wholesale_41=1015,
                                         Pop_Retail_44_45=1016,
                                         Pop_Transport_warehousing_48_49=1017,
                                         Pop_Information_cultural_51=1018,
                                         Pop_Finance_52=1019,
                                         Pop_Real_estate_53=1020,
                                         Pop_Professional_scientific_54=1021,
                                         Pop_Management_55=1022,
                                         Pop_Waste_management_56=1023,
                                         Pop_Education_61=1024,
                                         Pop_Health_care_62=1025,
                                         Pop_Art_71=1026,
                                         Pop_Food_service_72=1027,
                                         Pop_Other_81=1028,
                                         Pop_Public_administration_91=1029)

    @property
    def housing(self) -> Labels_IDs_Housing:
        return Labels_IDs_Housing(Household_Detached=120,
                                  Household_Apartment_5_storeys_over=124,
                                  Household_Semi_detached=121,
                                  Household_Row=122,
                                  Household_Duplex=123,
                                  Household_Apartment_4_storeys_less=125,
                                  Household_Other=126,
                                  Household_Movable=127,
                                  Household_Major_repair=108)

    # 110 = Household before 1946, 111 = Household_46_60, 112 = Household_61_70, 113 = Household_71_80,
    # 114 = Household_81_84, 115 = Household_85_90, 116 = Household_91_95, 117 = Household_96_00, 118 = Household_01_05
    @property
    def house_age(self) -> Labels_IDs_HouseAge:
        return Labels_IDs_HouseAge(Household_1960_before=[110, 111],
                                   Household_1961_1980=[112, 113],
                                   Household_1981_2000=[114, 115, 116, 117],
                                   Household_2001_2005=[118],
                                   Household_2006_2010=[],
                                   Household_2011_2015=[],
                                   Household_2016_2020=[])

    # 1235 = No_degree_15_24, 1249 = No_degree_25_64, 1263 = No_degree_65_over,
    # For this census, the prevalence is computed for us, but to keep a consistency in the algorithms,
    # we convert it back to a number of people and will recalculate the percentage along with the other census.
    # 1981 = Prevalence_Lico_%, 1979 = Total_pop_sample_Lico
    # 2051 = renter spending more than 30% of their income on dwelling, 2056 = owner spending more than 30% on dwelling
    @property
    def socioeconomic(self) -> Labels_IDs_Socioeconomic:
        return Labels_IDs_Socioeconomic(Pop_No_degree=[1235, 1249, 1263],
                                        Pop_Lico_at=[1979, 1981],
                                        Household_More_30_shelter_cost=[2051, 2056])


class Census_2011_Labels_IDs(AbstractCensus_Labels_IDs):
    """
    Because of political reasons, the 2011 census is incomplete and not formatted in a friendly way...
    Column common labels and specific ids from the Statistics Canada (2011) 2011 Census Profile,
    98-316-XWF2011001-1501.csv [CSV] file.
    """

    @property
    def age_M(self) -> Labels_IDs_Age_M:
        return Labels_IDs_Age_M(Age_M_tot='Total population by age groups',
                                Age_M_0_4='   0 to 4 years',
                                Age_M_5_9='   5 to 9 years',
                                Age_M_10_14='   10 to 14 years',
                                Age_M_15_19='   15 to 19 years',
                                Age_M_20_24='   20 to 24 years',
                                Age_M_25_29='   25 to 29 years',
                                Age_M_30_34='   30 to 34 years',
                                Age_M_35_39='   35 to 39 years',
                                Age_M_40_44='   40 to 44 years',
                                Age_M_45_49='   45 to 49 years',
                                Age_M_50_54='   50 to 54 years',
                                Age_M_55_59='   55 to 59 years',
                                Age_M_60_64='   60 to 64 years',
                                Age_M_65_69='   65 to 69 years',
                                Age_M_70_74='   70 to 74 years',
                                Age_M_75_79='   75 to 79 years',
                                Age_M_80_84='   80 to 84 years',
                                Age_M_85_over='   85 years and over')

    @property
    def age_F(self) -> Labels_IDs_Age_F:
        return Labels_IDs_Age_F(Age_F_tot='Total population by age groups',
                                Age_F_0_4='   0 to 4 years',
                                Age_F_5_9='   5 to 9 years',
                                Age_F_10_14='   10 to 14 years',
                                Age_F_15_19='   15 to 19 years',
                                Age_F_20_24='   20 to 24 years',
                                Age_F_25_29='   25 to 29 years',
                                Age_F_30_34='   30 to 34 years',
                                Age_F_35_39='   35 to 39 years',
                                Age_F_40_44='   40 to 44 years',
                                Age_F_45_49='   45 to 49 years',
                                Age_F_50_54='   50 to 54 years',
                                Age_F_55_59='   55 to 59 years',
                                Age_F_60_64='   60 to 64 years',
                                Age_F_65_69='   65 to 69 years',
                                Age_F_70_74='   70 to 74 years',
                                Age_F_75_79='   75 to 79 years',
                                Age_F_80_84='   80 to 84 years',
                                Age_F_85_over='   85 years and over')

    @property
    def household(self) -> Labels_IDs_Household:
        return Labels_IDs_Household(Household_Tot='Total number of private households by household size',
                                    Household_One_person='   1 person',
                                    Household_Lone_parent='         Lone-parent-family households',
                                    Household_Renter=None)

    @property
    def population(self) -> Labels_IDs_Population:
        return Labels_IDs_Population(Pop_Tot='Population in 2011',
                                     Pop_Not_in_labour=None,
                                     Pop_Unemployed=None,
                                     Pop_Employed=None,
                                     Pop_No_official_language='  Neither English nor French',
                                     Pop_Recent_immigrant=None,
                                     Pop_Aboriginal=None)

    @property
    def labour_industry(self) -> Labels_IDs_LabourIndustry:
        return Labels_IDs_LabourIndustry(Pop_Agriculture_11=None,
                                         Pop_Mining_oil_21=None,
                                         Pop_Utilities_22=None,
                                         Pop_Construction_23=None,
                                         Pop_Manufacturing_31_33=None,
                                         Pop_Wholesale_41=None,
                                         Pop_Retail_44_45=None,
                                         Pop_Transport_warehousing_48_49=None,
                                         Pop_Information_cultural_51=None,
                                         Pop_Finance_52=None,
                                         Pop_Real_estate_53=None,
                                         Pop_Professional_scientific_54=None,
                                         Pop_Management_55=None,
                                         Pop_Waste_management_56=None,
                                         Pop_Education_61=None,
                                         Pop_Health_care_62=None,
                                         Pop_Art_71=None,
                                         Pop_Food_service_72=None,
                                         Pop_Other_81=None,
                                         Pop_Public_administration_91=None)

    @property
    def housing(self) -> Labels_IDs_Housing:
        return Labels_IDs_Housing(Household_Detached='   Single-detached house',
                                  Household_Apartment_5_storeys_over='   Apartment, building that has five or more '
                                                                     'storeys',
                                  Household_Semi_detached='      Semi-detached house',
                                  Household_Row='      Row house',
                                  Household_Duplex='      Apartment, duplex',
                                  Household_Apartment_4_storeys_less='      Apartment, building that has fewer than '
                                                                     'five storeys',
                                  Household_Other='      Other single-attached house',
                                  Household_Movable='   Movable dwelling',
                                  Household_Major_repair=None)

    @property
    def house_age(self) -> Labels_IDs_HouseAge:
        return Labels_IDs_HouseAge(Household_1960_before=None,
                                   Household_1961_1980=None,
                                   Household_1981_2000=None,
                                   Household_2001_2005=None,
                                   Household_2006_2010=None,
                                   Household_2011_2015=None,
                                   Household_2016_2020=None)

    @property
    def socioeconomic(self) -> Labels_IDs_Socioeconomic:
        return Labels_IDs_Socioeconomic(Pop_No_degree=None,
                                        Pop_Lico_at=None,
                                        Household_More_30_shelter_cost=None)


class Census_2016_Labels_IDs(AbstractCensus_Labels_IDs):
    """
    Column common labels and specific ids from the Statistics Canada (2016) 2016 Census Profile,
    98-401-X2016044_QUEBEC.csv [CSV] file.
    """

    @property
    def age_M(self) -> Labels_IDs_Age_M:
        return Labels_IDs_Age_M(Age_M_tot=8,
                                Age_M_0_4=10,
                                Age_M_5_9=11,
                                Age_M_10_14=12,
                                Age_M_15_19=14,
                                Age_M_20_24=15,
                                Age_M_25_29=16,
                                Age_M_30_34=17,
                                Age_M_35_39=18,
                                Age_M_40_44=19,
                                Age_M_45_49=20,
                                Age_M_50_54=21,
                                Age_M_55_59=22,
                                Age_M_60_64=23,
                                Age_M_65_69=25,
                                Age_M_70_74=26,
                                Age_M_75_79=27,
                                Age_M_80_84=28,
                                Age_M_85_over=29)

    @property
    def age_F(self) -> Labels_IDs_Age_F:
        return Labels_IDs_Age_F(Age_F_tot=8,
                                Age_F_0_4=10,
                                Age_F_5_9=11,
                                Age_F_10_14=12,
                                Age_F_15_19=14,
                                Age_F_20_24=15,
                                Age_F_25_29=16,
                                Age_F_30_34=17,
                                Age_F_35_39=18,
                                Age_F_40_44=19,
                                Age_F_45_49=20,
                                Age_F_50_54=21,
                                Age_F_55_59=22,
                                Age_F_60_64=23,
                                Age_F_65_69=25,
                                Age_F_70_74=26,
                                Age_F_75_79=27,
                                Age_F_80_84=28,
                                Age_F_85_over=29)

    @property
    def household(self) -> Labels_IDs_Household:
        return Labels_IDs_Household(Household_Tot=51,
                                    Household_One_person=52,
                                    Household_Lone_parent=78,
                                    Household_Renter=1619)

    @property
    def population(self) -> Labels_IDs_Population:
        return Labels_IDs_Population(Pop_Tot=1,
                                     Pop_Not_in_labour=1869,
                                     Pop_Unemployed=1868,
                                     Pop_Employed=1867,
                                     Pop_No_official_language=104,
                                     Pop_Recent_immigrant=1149,
                                     Pop_Aboriginal=1290)

    @property
    def labour_industry(self) -> Labels_IDs_LabourIndustry:
        return Labels_IDs_LabourIndustry(Pop_Agriculture_11=1900,
                                         Pop_Mining_oil_21=1901,
                                         Pop_Utilities_22=1902,
                                         Pop_Construction_23=1903,
                                         Pop_Manufacturing_31_33=1904,
                                         Pop_Wholesale_41=1905,
                                         Pop_Retail_44_45=1906,
                                         Pop_Transport_warehousing_48_49=1907,
                                         Pop_Information_cultural_51=1908,
                                         Pop_Finance_52=1909,
                                         Pop_Real_estate_53=1910,
                                         Pop_Professional_scientific_54=1911,
                                         Pop_Management_55=1912,
                                         Pop_Waste_management_56=1913,
                                         Pop_Education_61=1914,
                                         Pop_Health_care_62=1915,
                                         Pop_Art_71=1916,
                                         Pop_Food_service_72=1917,
                                         Pop_Other_81=1918,
                                         Pop_Public_administration_91=1919)

    @property
    def housing(self) -> Labels_IDs_Housing:
        return Labels_IDs_Housing(Household_Detached=42,
                                  Household_Apartment_5_storeys_over=43,
                                  Household_Semi_detached=45,
                                  Household_Row=46,
                                  Household_Duplex=47,
                                  Household_Apartment_4_storeys_less=48,
                                  Household_Other=49,
                                  Household_Movable=50,
                                  Household_Major_repair=1653)

    @property
    def house_age(self) -> Labels_IDs_HouseAge:
        return Labels_IDs_HouseAge(Household_1960_before=[1644],
                                   Household_1961_1980=[1645],
                                   Household_1981_2000=[1646, 1647],
                                   Household_2001_2005=[1648],
                                   Household_2006_2010=[1649],
                                   Household_2011_2015=[1650],
                                   Household_2016_2020=[])

    @property
    def socioeconomic(self) -> Labels_IDs_Socioeconomic:
        return Labels_IDs_Socioeconomic(Pop_No_degree=[1684],
                                        Pop_Lico_at=[862],
                                        Household_More_30_shelter_cost=[1669])


class Census_2021_Labels_IDs(AbstractCensus_Labels_IDs):
    """
    Column common labels and specific ids from the Statistics Canada (2021) 2021 Census Profile,
    98-401-X2021006_English_CSV_data_Quebec.csv [CSV] file.
    """

    @property
    def age_M(self) -> Labels_IDs_Age_M:
        return Labels_IDs_Age_M(Age_M_tot=8,
                                Age_M_0_4=10,
                                Age_M_5_9=11,
                                Age_M_10_14=12,
                                Age_M_15_19=14,
                                Age_M_20_24=15,
                                Age_M_25_29=16,
                                Age_M_30_34=17,
                                Age_M_35_39=18,
                                Age_M_40_44=19,
                                Age_M_45_49=20,
                                Age_M_50_54=21,
                                Age_M_55_59=22,
                                Age_M_60_64=23,
                                Age_M_65_69=25,
                                Age_M_70_74=26,
                                Age_M_75_79=27,
                                Age_M_80_84=28,
                                Age_M_85_over=29)

    @property
    def age_F(self) -> Labels_IDs_Age_F:
        return Labels_IDs_Age_F(Age_F_tot=8,
                                Age_F_0_4=10,
                                Age_F_5_9=11,
                                Age_F_10_14=12,
                                Age_F_15_19=14,
                                Age_F_20_24=15,
                                Age_F_25_29=16,
                                Age_F_30_34=17,
                                Age_F_35_39=18,
                                Age_F_40_44=19,
                                Age_F_45_49=20,
                                Age_F_50_54=21,
                                Age_F_55_59=22,
                                Age_F_60_64=23,
                                Age_F_65_69=25,
                                Age_F_70_74=26,
                                Age_F_75_79=27,
                                Age_F_80_84=28,
                                Age_F_85_over=29)

    @property
    def household(self) -> Labels_IDs_Household:
        return Labels_IDs_Household(Household_Tot=50,
                                    Household_One_person=51,
                                    Household_Lone_parent=86,
                                    Household_Renter=1416)

    @property
    def population(self) -> Labels_IDs_Population:
        return Labels_IDs_Population(Pop_Tot=1,
                                     Pop_Not_in_labour=2227,
                                     Pop_Unemployed=2226,
                                     Pop_Employed=2225,
                                     Pop_No_official_language=387,
                                     Pop_Recent_immigrant=1536,
                                     Pop_Aboriginal=1403)

    @property
    def labour_industry(self) -> Labels_IDs_LabourIndustry:
        return Labels_IDs_LabourIndustry(Pop_Agriculture_11=2262,
                                         Pop_Mining_oil_21=2263,
                                         Pop_Utilities_22=2264,
                                         Pop_Construction_23=2265,
                                         Pop_Manufacturing_31_33=2266,
                                         Pop_Wholesale_41=2267,
                                         Pop_Retail_44_45=2268,
                                         Pop_Transport_warehousing_48_49=2269,
                                         Pop_Information_cultural_51=2270,
                                         Pop_Finance_52=2271,
                                         Pop_Real_estate_53=2272,
                                         Pop_Professional_scientific_54=2273,
                                         Pop_Management_55=2274,
                                         Pop_Waste_management_56=2275,
                                         Pop_Education_61=2276,
                                         Pop_Health_care_62=2277,
                                         Pop_Art_71=2278,
                                         Pop_Food_service_72=2279,
                                         Pop_Other_81=2280,
                                         Pop_Public_administration_91=2281)

    @property
    def housing(self) -> Labels_IDs_Housing:
        return Labels_IDs_Housing(Household_Detached=42,
                                  Household_Apartment_5_storeys_over=47,
                                  Household_Semi_detached=43,
                                  Household_Row=44,
                                  Household_Duplex=45,
                                  Household_Apartment_4_storeys_less=46,
                                  Household_Other=48,
                                  Household_Movable=49,
                                  Household_Major_repair=1451)

    @property
    def house_age(self) -> Labels_IDs_HouseAge:
        return Labels_IDs_HouseAge(Household_1960_before=[1441],
                                   Household_1961_1980=[1442],
                                   Household_1981_2000=[1443, 1444],
                                   Household_2001_2005=[1445],
                                   Household_2006_2010=[1446],
                                   Household_2011_2015=[1447],
                                   Household_2016_2020=[1448])

    @property
    def socioeconomic(self) -> Labels_IDs_Socioeconomic:
        return Labels_IDs_Socioeconomic(Pop_No_degree=[1999],
                                        Pop_Lico_at=[355],
                                        Household_More_30_shelter_cost=[1467])
