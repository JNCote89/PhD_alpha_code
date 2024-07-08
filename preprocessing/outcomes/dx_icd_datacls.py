from dataclasses import dataclass, asdict

import pandas as pd


@dataclass(slots=True)
class DxCategories:
    infection: list[str]
    neoplasms: list[str]
    blood: list[str]
    endocrine: list[str]
    mental: list[str]
    nervous: list[str]
    eyes_ears: list[str]
    cardio: list[str]
    pneumo: list[str]
    digestive: list[str]
    skin: list[str]
    musculoskeletal: list[str]
    kidney: list[str]
    genitals: list[str]
    pregnancy: list[str]
    perinatal: list[str]
    congenital: list[str]
    symptoms: list[str]
    injury: list[str]
    uncertain: list[str]
    external: list[str]
    other: list[str]


@dataclass(slots=True)
class ICD10_dx:

    @property
    def get_dx_categories(self) -> DxCategories:
        # Range excludes the last number, so the dx code usually stops at 99
        return DxCategories(
            infection=[f"A{code:02d}" for code in range(0, 100)] + [f"B{code:02d}" for code in range(0, 100)],
            neoplasms=[f"C{code:02d}" for code in range(0, 100)] + [f"D{code:02d}" for code in range(0, 50)],
            blood=[f"D{code:02d}" for code in range(50, 100)],
            endocrine=[f"E{code:02d}" for code in range(0, 100)],
            mental=[f"F{code:02d}" for code in range(0, 100)],
            nervous=[f"G{code:02d}" for code in range(0, 100)],
            eyes_ears=[f"H{code:02d}" for code in range(0, 100)],
            cardio=[f"I{code:02d}" for code in range(0, 100)],
            pneumo=[f"J{code:02d}" for code in range(0, 100)],
            digestive=[f"K{code:02d}" for code in range(0, 100)],
            skin=[f"L{code:02d}" for code in range(0, 100)],
            musculoskeletal=[f"M{code:02d}" for code in range(0, 100)],
            kidney=[f"N{code:02d}" for code in range(0, 40)],
            genitals=[f"N{code:02d}" for code in range(40, 100)],
            pregnancy=[f"O{code:02d}" for code in range(0, 100)],
            perinatal=[f"P{code:02d}" for code in range(0, 100)],
            congenital=[f"Q{code:02d}" for code in range(0, 100)],
            symptoms=[f"R{code:02d}" for code in range(0, 100)],
            injury=[f"S{code:02d}" for code in range(0, 100)] + [f"T{code:02d}" for code in range(0, 100)],
            uncertain=[f"U{code:02d}" for code in range(0, 100)],
            external=[f"V{code:02d}" for code in range(0, 100)] + [f"W{code:02d}" for code in range(0, 100)
                                                                   ] + [f"X{code:02d}" for code in range(0, 100)],
            other=[f"Z{code:02d}" for code in range(0, 100)])

    def classified_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df_copy = df.copy()
        for cls, code in asdict(self.get_dx_categories).items():
            df_copy[cls] = 0
            df_copy.loc[df_copy['dx'].isin(code), cls] = 1

        return df_copy


@dataclass(slots=True)
class ConversionTableICD9_10_ColumnNames:
    """
    Internal file from Health Canada mapping every ICD9 code to its ICD10 equivalent with a concordance rating. However,
    there are some one-to-many relationships making it hard to get an exact mapping. Therefore, to try to get a more
    accurate mapping, the prevalence of each ICD9 code is calculated and the ICD10 code with the highest prevalence is
    chosen.
    """
    ICD10: str = 'ICD10'
    label: str = 'label'
    ICD9: str = 'ICD9'
    ICD9_rating: str = 'ICD9_rating'
    ICD10_frequency: str = 'ICD10_frequency'

    def add_frequency(self, df_in: pd.DataFrame, dx_column_name: str) -> pd.DataFrame:
        df_copy = df_in.copy()
        return df_copy[dx_column_name].value_counts().rename_axis(dx_column_name
                                                                  ).to_frame(self.ICD10_frequency)
