import pandas as pd
import numpy as np
import wbgapi as wb
import requests
import io
import country_converter as coco
import logging
from typing import List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ConflictDownloader:
    """
    Downloads and processes Correlates of War (COW) data.
    """

    COW_URLS = {
        'inter_state': 'https://correlatesofwar.org/wp-content/uploads/Inter-StateWarData_v4.0.csv',
        'intra_state': 'https://correlatesofwar.org/wp-content/uploads/Intra-StateWarData_v4.1.csv',
        'non_state': 'https://correlatesofwar.org/wp-content/uploads/Non-StateWarData_v4.0.csv',
        'extra_state': 'https://correlatesofwar.org/wp-content/uploads/Extra-StateWarData_v4.0.csv'
    }

    def __init__(self):
        self.cc = coco.CountryConverter()

    def download_csv(self, url: str) -> pd.DataFrame:
        """Downloads a CSV from a URL and returns a DataFrame."""
        try:
            logger.info(f"Downloading data from {url}")
            response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
            response.raise_for_status()
            content = response.content
            # Try decoding with utf-8, fallback to latin1 if needed
            try:
                df = pd.read_csv(io.BytesIO(content), encoding='utf-8')
            except UnicodeDecodeError:
                df = pd.read_csv(io.BytesIO(content), encoding='latin1')
            return df
        except Exception as e:
            logger.error(f"Failed to download {url}: {e}")
            raise

    def process_cow_data(self) -> pd.DataFrame:
        """
        Downloads all COW datasets, cleans them, and aggregates to a Country-Year panel.
        Returns a DataFrame with columns: ['ISO3', 'Year', 'War_Binary', 'War_Intensity', 'Battle_Deaths']
        """
        all_wars = []

        # Process Inter-State
        df_inter = self.download_csv(self.COW_URLS['inter_state'])
        # Columns typically: WarNum, WarName, WarType, Ccode, StateName, StartYear1, EndYear1, ... BatDeath
        # We need to expand this to country-years.
        all_wars.append(self._expand_to_country_year(df_inter, 'inter'))

        # Process Intra-State
        df_intra = self.download_csv(self.COW_URLS['intra_state'])
        all_wars.append(self._expand_to_country_year(df_intra, 'intra'))

        # Process Extra-State
        df_extra = self.download_csv(self.COW_URLS['extra_state'])
        all_wars.append(self._expand_to_country_year(df_extra, 'extra'))

        # Non-state wars might be harder to map to a state, but we will try if there is state involvement info.
        # Often Non-state wars are between non-state actors. If they take place in a territory, we might assign it.
        # For this rigorous analysis, we'll focus on wars where the state is a participant (Inter, Intra, Extra).
        # We can try to include Non-State if they have location codes.

        # Concatenate all
        full_df = pd.concat(all_wars, ignore_index=True)

        # Aggregate by ISO3 and Year
        # We want to know if a country was in a war in a given year.
        # Max of War_Binary, Sum of Battle_Deaths

        aggregated = full_df.groupby(['ISO3', 'Year']).agg({
            'War_Binary': 'max',
            'Battle_Deaths': 'sum',
            'War_Type': lambda x: ','.join(set(x))
        }).reset_index()

        aggregated['War_Intensity'] = np.log1p(aggregated['Battle_Deaths'])

        return aggregated

    def _expand_to_country_year(self, df: pd.DataFrame, war_type: str) -> pd.DataFrame:
        """
        Expands a COW war entry (StartYear, EndYear) to a series of rows for each year.
        """
        expanded_rows = []

        # Standardize column names roughly
        # different datasets have different column names for start/end dates
        # Inter-state: StartYear1, EndYear1, EndYear2...
        # We will simplify by taking the min start and max end for the record.

        # Helper to find columns
        def get_col(candidates):
            for c in candidates:
                if c in df.columns:
                    return c
            return None

        ccode_col = get_col(['ccode', 'Ccode', 'StateNum'])
        statename_col = get_col(['StateName', 'Name', 'SideA']) # Use name for conversion
        start_year_col = get_col(['StartYear1', 'StartYear', 'YrBeg1'])
        end_year_col = get_col(['EndYear1', 'EndYear', 'YrEnd1'])
        deaths_col = get_col(['BatDeath', 'BatDeaths', 'SideDeaths'])

        if not (start_year_col and end_year_col):
             logger.warning(f"Could not find date columns for {war_type} dataset. Skipping.")
             return pd.DataFrame()

        # Use StateName for conversion if available, else skip or rely on ccode if I could
        # But since cown is not working, we rely on names.
        if not statename_col:
             logger.warning(f"Could not find StateName column for {war_type}. Skipping.")
             return pd.DataFrame()

        # Pre-convert names to ISO3
        unique_names = df[statename_col].unique()
        unique_names = [n for n in unique_names if isinstance(n, str)]

        iso3_map = self.cc.convert(names=unique_names, to='ISO3', not_found='None')

        if isinstance(iso3_map, list):
             name_map = dict(zip(unique_names, iso3_map))
        else:
             name_map = {unique_names[0]: iso3_map}

        for idx, row in df.iterrows():
            name = row[statename_col]
            if name not in name_map or name_map[name] == 'None':
                continue

            iso3 = name_map[name]
            if isinstance(iso3, list):
                iso3 = iso3[0]

            # Handle -9 or missing dates
            start_yr = row[start_year_col]
            end_yr = row[end_year_col]

            # Helper to safely check validity
            def is_invalid(val):
                return val is None or pd.isna(val) or val == -9

            if is_invalid(start_yr):
                continue

            # Treat ongoing (-7) or missing end as start_yr (duration 1) or strictly limit to available data
            if is_invalid(end_yr) or end_yr < 0:
                end_yr = start_yr

            # Cast to int safe
            try:
                s_y = int(float(start_yr))
                e_y = int(float(end_yr))
            except ValueError:
                continue

            # Loop through years
            # Battle deaths are often total for the war. We distribute them evenly or just assign to all years.
            # Distributing evenly is better.

            duration = e_y - s_y + 1
            if duration < 1: duration = 1

            # Handle deaths
            deaths = row[deaths_col] if deaths_col else 0
            if deaths == -9 or deaths is None or deaths < 0:
                deaths = 0

            annual_deaths = deaths / duration

            for y in range(s_y, e_y + 1):
                expanded_rows.append({
                    'ISO3': iso3,
                    'Year': y,
                    'War_Binary': 1,
                    'Battle_Deaths': annual_deaths,
                    'War_Type': war_type
                })

        return pd.DataFrame(expanded_rows)


class WorldBankFetcher:
    """
    Fetches economic data from World Bank API.
    """

    INDICATORS = {
        'NY.GDP.PCAP.KD': 'GDP_Per_Capita_Constant',
        'NY.GDP.MKTP.KD.ZG': 'GDP_Growth',
        'FP.CPI.TOTL.ZG': 'Inflation',
        'SL.UEM.TOTL.ZS': 'Unemployment',
        'GC.XPN.TOTL.GD.ZS': 'Govt_Expenditure_GDP',
        'NE.TRD.GNFS.ZS': 'Trade_Openness',
        'BX.KLT.DINV.WD.GD.ZS': 'FDI_Inflows_GDP',
        'NV.AGR.TOTL.ZS': 'Agri_Value_Added',
        'NV.IND.TOTL.ZS': 'Ind_Value_Added',
        'NV.SRV.TOTL.ZS': 'Serv_Value_Added',
        'SE.SEC.ENRR': 'School_Enrollment'
    }

    def fetch_metadata(self) -> pd.DataFrame:
        """Fetches country metadata (Income Groups)."""
        logger.info("Fetching World Bank country metadata...")
        # wbgapi economy.info returns a generator of dicts or similar.
        # We use wb.economy.DataFrame() for easier handling
        meta = wb.economy.DataFrame()
        # meta index is the ID (ISO3)
        # We need 'incomeLevel'
        return meta[['incomeLevel', 'region', 'name']].reset_index().rename(columns={'id': 'ISO3', 'incomeLevel': 'Income_Group'})

    def fetch_indicators(self, start_year=1990, end_year=2024) -> pd.DataFrame:
        """Fetches all defined indicators for all countries."""
        logger.info("Fetching World Bank economic indicators...")

        # Fetching all at once might be large, but wbgapi handles it well.
        # data.DataFrame returns index as (economy, time) (or similar depending on params)

        try:
            df = wb.data.DataFrame(
                list(self.INDICATORS.keys()),
                economy='all',
                time=range(start_year, end_year + 1),
                labels=False,
                skipAggs=True, # Skip aggregates like 'WLD', 'AFR'
                numericTimeKeys=True
            )

            # df has MultiIndex (economy) and columns are years, or flat?
            # wbgapi default: Index=economy, Columns=series+Year? Or if multiple series, it might be tricky.
            # Best is to fetch tidy format or melt.
            # wb.data.DataFrame with multiple indicators usually puts indicators in columns if time is index, or vice versa?
            # Actually, with multiple indicators and time range, it usually returns a MultiIndex or simple wide format.
            # Let's check documentation memory...
            # "d = wb.data.DataFrame(ind_code, COUNTRIES, time=...)" in memory returns economy index, year columns.
            # With multiple indicators, it might stack them or use multi-level columns.

            # Let's use `wb.data.fetch` which returns a generator of dicts, cleaner for converting to DataFrame
            data_iter = wb.data.fetch(
                list(self.INDICATORS.keys()),
                economy='all',
                time=range(start_year, end_year + 1),
                skipAggs=True
            )

            records = []
            for d in data_iter:
                # d is like {'value': ..., 'series': 'NY.GDP...', 'economy': 'ZWE', 'time': '2020'}
                records.append(d)

            df_tidy = pd.DataFrame(records)

            # Pivot to wide format: Index=[Country, Year], Columns=[Indicators]
            df_pivot = df_tidy.pivot_table(index=['economy', 'time'], columns='series', values='value').reset_index()

            # Rename columns
            df_pivot.rename(columns=self.INDICATORS, inplace=True)
            df_pivot.rename(columns={'economy': 'ISO3', 'time': 'Year'}, inplace=True)

            # Convert Year to int
            df_pivot['Year'] = pd.to_numeric(df_pivot['Year'].astype(str).str.replace('YR', ''), errors='coerce')

            return df_pivot

        except Exception as e:
            logger.error(f"Error fetching WB data: {e}")
            raise

if __name__ == "__main__":
    # Test execution
    try:
        cd = ConflictDownloader()
        conflict_data = cd.process_cow_data()
        print("Conflict Data Sample:")
        print(conflict_data.head())

        wb_fetcher = WorldBankFetcher()
        meta = wb_fetcher.fetch_metadata()
        print("Metadata Sample:")
        print(meta.head())

        # Test fetch small range
        econ = wb_fetcher.fetch_indicators(start_year=2020, end_year=2022)
        print("Econ Data Sample:")
        print(econ.head())

    except Exception as e:
        print(f"Error: {e}")
