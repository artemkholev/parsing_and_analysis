import re
import pandas as pd
import numpy as np
from typing import List, Tuple
from sklearn.preprocessing import StandardScaler

from pipeline.base_handler import BaseHandler


class DataLoaderHandler(BaseHandler):
    """Loads CSV data into DataFrame"""

    def __init__(self, file_path: str):
        super().__init__()
        self.file_path = file_path

    def _process(self, input_data: None) -> pd.DataFrame:
        """
        Load CSV file into pandas DataFrame

        Args:
            input_data: Not used (entry point)

        Returns:
            Loaded DataFrame
        """
        print(f"Loading from {self.file_path}...")
        dataframe = pd.read_csv(self.file_path, index_col=0, engine='python', on_bad_lines='warn')
        print(f"Loaded {len(dataframe)} rows")
        return dataframe


class DataCleaningHandler(BaseHandler):
    """Cleans data by removing duplicates and special characters"""

    def _process(self, input_data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean DataFrame by removing duplicates and normalizing text

        Args:
            input_data: Input DataFrame

        Returns:
            Cleaned DataFrame
        """
        print("Cleaning data...")
        output = input_data.copy()

        # Remove duplicates
        initial_count = len(output)
        output = output.drop_duplicates()
        print(f"Removed {initial_count - len(output)} duplicates")

        # Clean text columns
        text_columns = output.select_dtypes(include=['object']).columns
        for column in text_columns:
            if column in output.columns:
                output[column] = output[column].astype(str)
                output[column] = output[column].str.replace('\ufeff', '', regex=False)
                output[column] = output[column].str.replace('\xa0', ' ', regex=False)
                output[column] = output[column].str.replace(r'[\t\n\r]', ' ', regex=True)
                output[column] = output[column].str.replace(r'\s+', ' ', regex=True).str.strip()

        return output


class FeatureExtractionHandler(BaseHandler):
    """Extracts structured features from raw text data"""

    def _process(self, input_data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract structured features from text columns

        Args:
            input_data: Input DataFrame with raw text data

        Returns:
            DataFrame with extracted features
        """
        print("Extracting features...")
        output = input_data.copy()

        output['Salary'] = self._extract_salary(output['ЗП'])
        output['Age'] = self._extract_age(output['Пол, возраст'])
        output['Gender'] = self._extract_gender(output['Пол, возраст'])
        output['City'] = self._extract_city(output['Город'])
        output['Employment'] = self._extract_employment(output['Занятость'])
        output['Schedule'] = self._extract_schedule(output['График'])
        output['Experience_Years'] = self._extract_experience(output['Опыт (двойное нажатие для полной версии)'])
        output['Education'] = self._extract_education(output['Образование и ВУЗ'])
        output['Has_Car'] = self._extract_car(output['Авто'])

        selected_cols = ['Salary', 'Age', 'Gender', 'City', 'Employment',
                        'Schedule', 'Experience_Years', 'Education', 'Has_Car']
        return output[selected_cols]

    @staticmethod
    def _extract_salary(series: pd.Series) -> pd.Series:
        def parse(value):
            if pd.isna(value) or value == 'nan':
                return np.nan
            value = str(value).replace('\xa0', ' ')
            value = re.sub(r'[^\d\s]', '', value)
            numbers = re.findall(r'\d+', value)
            return float(numbers[0]) if numbers else np.nan
        return series.apply(parse)

    @staticmethod
    def _extract_age(series: pd.Series) -> pd.Series:
        def parse(value):
            if pd.isna(value) or value == 'nan':
                return np.nan
            match = re.search(r'(\d+)\s*(?:год|лет|года)', str(value))
            return int(match.group(1)) if match else np.nan
        return series.apply(parse)

    @staticmethod
    def _extract_gender(series: pd.Series) -> pd.Series:
        def parse(value):
            value = str(value).lower()
            if 'мужчина' in value or 'male' in value:
                return 'male'
            elif 'женщина' in value or 'female' in value:
                return 'female'
            return 'unknown'
        return series.apply(parse)

    @staticmethod
    def _extract_city(series: pd.Series) -> pd.Series:
        def parse(value):
            if pd.isna(value) or value == 'nan':
                return 'unknown'
            parts = str(value).split(',')
            return parts[0].strip().lower() if parts else 'unknown'
        return series.apply(parse)

    @staticmethod
    def _extract_employment(series: pd.Series) -> pd.Series:
        def parse(value):
            value = str(value).lower()
            if 'полная' in value or 'full' in value:
                return 'full'
            elif 'частичная' in value or 'part' in value:
                return 'part'
            return 'unknown'
        return series.apply(parse)

    @staticmethod
    def _extract_schedule(series: pd.Series) -> pd.Series:
        def parse(value):
            value = str(value).lower()
            if 'полный день' in value:
                return 'full_day'
            elif 'гибкий' in value:
                return 'flexible'
            elif 'удаленная' in value:
                return 'remote'
            elif 'сменный' in value:
                return 'shift'
            return 'unknown'
        return series.apply(parse)

    @staticmethod
    def _extract_experience(series: pd.Series) -> pd.Series:
        def parse(value):
            if pd.isna(value) or value == 'nan':
                return 0.0
            value = str(value)
            years_match = re.search(r'(\d+)\s*(?:год|лет|года)', value)
            months_match = re.search(r'(\d+)\s*месяц', value)
            years = float(years_match.group(1)) if years_match else 0.0
            months = float(months_match.group(1)) if months_match else 0.0
            return years + months / 12.0
        return series.apply(parse)

    @staticmethod
    def _extract_education(series: pd.Series) -> pd.Series:
        def parse(value):
            value = str(value).lower()
            if 'высшее' in value or 'bachelor' in value:
                return 'higher'
            elif 'среднее специальное' in value:
                return 'vocational'
            elif 'среднее' in value:
                return 'secondary'
            elif 'неоконченное' in value:
                return 'incomplete_higher'
            return 'unknown'
        return series.apply(parse)

    @staticmethod
    def _extract_car(series: pd.Series) -> pd.Series:
        def parse(value):
            value = str(value).lower()
            return 1 if any(keyword in value for keyword in ['имеется', 'есть', 'собственный']) else 0
        return series.apply(parse)


class MissingDataHandler(BaseHandler):
    """Handles missing values in the dataset"""

    def _process(self, input_data: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the DataFrame

        For numeric columns, fills with median
        For categorical columns, fills with 'unknown'

        Args:
            input_data: Input DataFrame

        Returns:
            DataFrame with missing values handled
        """
        print("Handling missing values...")
        output = input_data.copy()

        # Fill numeric columns with median
        numeric_cols = output.select_dtypes(include=[np.number]).columns
        for column in numeric_cols:
            if output[column].isna().any():
                median_value = output[column].median()
                output[column] = output[column].fillna(median_value)
                print(f"Filled {column} missing values with median: {median_value:.2f}")

        # Fill categorical columns with 'unknown'
        categorical_cols = output.select_dtypes(include=['object']).columns
        for column in categorical_cols:
            if output[column].isna().any():
                output[column] = output[column].fillna('unknown')

        return output


class OutlierRemovalHandler(BaseHandler):
    """Removes outliers using the IQR method"""

    def __init__(self, columns: List[str] = None, iqr_multiplier: float = 1.5):
        """
        Initialize outlier removal handler

        Args:
            columns: List of column names to check for outliers (default: all numeric)
            iqr_multiplier: IQR multiplier for outlier detection (default: 1.5)
        """
        super().__init__()
        self.columns = columns
        self.iqr_multiplier = iqr_multiplier

    def _process(self, input_data: pd.DataFrame) -> pd.DataFrame:
        """
        Remove outliers using IQR method

        Args:
            input_data: Input DataFrame

        Returns:
            DataFrame with outliers removed
        """
        print("Removing outliers...")
        output = input_data.copy()
        initial_count = len(output)

        # Select columns to check
        cols_to_check = self.columns if self.columns else output.select_dtypes(include=[np.number]).columns

        for column in cols_to_check:
            if column in output.columns and column != 'Salary':
                Q1 = output[column].quantile(0.25)
                Q3 = output[column].quantile(0.75)
                IQR = Q3 - Q1

                lower_bound = Q1 - self.iqr_multiplier * IQR
                upper_bound = Q3 + self.iqr_multiplier * IQR

                output = output[(output[column] >= lower_bound) & (output[column] <= upper_bound)]

        print(f"Removed {initial_count - len(output)} outlier rows")
        return output


class CategoryGroupingHandler(BaseHandler):
    """Groups rare categories to ensure balance"""

    def __init__(self, min_frequency: int = 100):
        """
        Initialize category grouping handler

        Args:
            min_frequency: Minimum frequency for a category to remain separate
        """
        super().__init__()
        self.min_frequency = min_frequency

    def _process(self, input_data: pd.DataFrame) -> pd.DataFrame:
        """
        Group rare categories into 'other'

        Args:
            input_data: Input DataFrame

        Returns:
            DataFrame with grouped categories
        """
        print("Grouping rare categories...")
        output = input_data.copy()

        # Group rare cities
        if 'City' in output.columns:
            city_counts = output['City'].value_counts()
            rare_cities = city_counts[city_counts < self.min_frequency].index
            output.loc[output['City'].isin(rare_cities), 'City'] = 'other'
            print(f"Grouped {len(rare_cities)} rare cities into 'other'")

        # Map education levels to broader categories
        if 'Education' in output.columns:
            education_mapping = {
                'higher': 'higher',
                'incomplete_higher': 'higher',
                'vocational': 'vocational',
                'secondary': 'secondary',
                'unknown': 'other'
            }
            output['Education'] = output['Education'].map(lambda x: education_mapping.get(x, 'other'))

        return output


class EncodingHandler(BaseHandler):
    """Encodes categorical variables using one-hot encoding"""

    def _process(self, input_data: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical variables using one-hot encoding

        Args:
            input_data: Input DataFrame

        Returns:
            DataFrame with encoded categorical variables
        """
        print("Encoding categorical variables...")
        output = input_data.copy()

        # Apply one-hot encoding
        categorical_cols = output.select_dtypes(include=['object']).columns.tolist()

        if categorical_cols:
            output = pd.get_dummies(output, columns=categorical_cols, prefix=categorical_cols, drop_first=True)
            print(f"Encoded {len(categorical_cols)} categorical columns")

        # Convert boolean to int
        for column in output.columns:
            if output[column].dtype == 'bool':
                output[column] = output[column].astype(int)

        return output


class NormalizationHandler(BaseHandler):
    """Normalizes numeric features using StandardScaler"""

    def _process(self, input_data: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize numeric features using StandardScaler

        Args:
            input_data: Input DataFrame

        Returns:
            DataFrame with normalized numeric features
        """
        print("Normalizing numeric features...")
        output = input_data.copy()

        # Separate target variable from features
        if 'Salary' in output.columns:
            target = output['Salary'].copy()
            features = output.drop('Salary', axis=1)
        else:
            target = None
            features = output

        # Normalize numeric columns
        numeric_cols = features.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) > 0:
            scaler = StandardScaler()
            features[numeric_cols] = scaler.fit_transform(features[numeric_cols])
            print(f"Normalized {len(numeric_cols)} numeric columns")

        # Recombine with target
        if target is not None:
            output = features.copy()
            output['Salary'] = target

        return output


class ArrayConversionHandler(BaseHandler):
    """Converts DataFrame to numpy arrays for X and y"""

    def _process(self, input_data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert DataFrame to numpy arrays

        Args:
            input_data: Input DataFrame

        Returns:
            Tuple of (X_array, y_array) as numpy arrays
        """
        print("Converting to numpy arrays...")

        if 'Salary' in input_data.columns:
            # Remove rows where salary is missing
            clean_data = input_data[input_data['Salary'].notna()].copy()

            # Separate features and target
            y_array = clean_data['Salary'].values
            X_array = clean_data.drop('Salary', axis=1).values

            print(f"X shape: {X_array.shape}, y shape: {y_array.shape}")
            return X_array, y_array
        else:
            raise ValueError("Salary column not found in data")
