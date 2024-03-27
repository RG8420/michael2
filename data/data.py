import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from data.data_utils import load_sheet_names, load_csv_data, load_excel_data, missing_values_per_column, get_info, \
    convert_datetime, handle_missing_values, encode_categorical


class VietnamDataset:
    def __init__(self, paths):
        self.paths = paths

    def _extract_paths(self):
        self.main_data_path = self.paths["main_data_path"]
        self.coordinate_path = self.paths["coordinate_path"]
        self.criterion_path = self.paths["criterion_path"]

    def _extract_files(self):
        # print("Starting Extraction")
        # print("_" * 50)
        #
        # print("Extracting main data for Vietnam dataset...")
        # print("_" * 20)
        load_sheet_names(self.main_data_path)
        self.df_vietnam = load_excel_data(self.main_data_path,
                                          sheet_name="Sheet")

        # print("Extracting Coordinate for Vietnam dataset...")
        # print("_" * 20)
        load_sheet_names(self.coordinate_path)
        self.df_vietnam_coordinate = load_excel_data(self.coordinate_path,
                                                     sheet_name="Coordinate")

        # print("Extracting Criterion for Vietnam dataset...")
        # print("_" * 20)
        load_sheet_names(self.criterion_path)
        self.df_vietnam_criterion = load_excel_data(self.criterion_path,
                                                    sheet_name="Sheet")

        # print("Extraction Done.")
        # print("_" * 50)

    def _explore_files(self):
        print("Starting Exploration...")
        print("_" * 50)

        print("Exploring main data for Vietnam dataset...")
        print("_" * 20)
        print("Missing Values per Column")
        print("-" * 15)
        missing_values_per_column(self.df_vietnam)
        print("Information")
        print("-" * 15)
        get_info(self.df_vietnam)

        print("Exploring Coordinate for Vietnam dataset...")
        print("_" * 20)
        print("Missing Values per Column")
        print("-" * 15)
        missing_values_per_column(self.df_vietnam_coordinate)
        print("Information")
        print("-" * 15)
        get_info(self.df_vietnam_coordinate)

        print("Exploring Criterion for Vietnam dataset...")
        print("_" * 20)
        print("Missing Values per Column")
        print("-" * 15)
        missing_values_per_column(self.df_vietnam_criterion)
        print("Information")
        print("-" * 15)
        get_info(self.df_vietnam_criterion)

        print("Exploration Done.")
        print("_" * 50)

    def _preprocess_dataset(self):
        # print("Starting Preprocessing...")
        # print("_" * 50)

        self.df_vietnam_timestamp = convert_datetime(self.df_vietnam,
                                                     column="date_sampling")
        self.df_vietnam_timestamp = convert_datetime(self.df_vietnam_timestamp,
                                                     column="date_analyzing")

        self.df_vietnam_encoded, self.df_vietnam_categorical_columns = \
            encode_categorical(self.df_vietnam_timestamp)

        self.df_vietnam_missing_handled = handle_missing_values(self.df_vietnam_encoded)

    def _define_wqi(self):
        #  standard value recommended for parameter
        si = 8.5 + 1000 + 500 + 300 + 200 + 75 + 30 + 200 + 12 + 350 + 250 + 200 + 1.5
        k = 1 / si

        values = [8.5, 1500, 50, 200, 12, 350, 250, 200, 200, 200]
        wi = []
        for i in values:
            wi.append(k / i)

        pHn = []
        for i in self.df_vietnam_missing_handled['ph']:
            pHn.append(100 * ((i - 7.0) / (8.5 - 7.0)))

        TDSn = []
        for i in self.df_vietnam_missing_handled['tds105']:
            TDSn.append(100 * (i / 1500))

        Magnesiumn = []
        for i in self.df_vietnam_missing_handled['mg2']:
            Magnesiumn.append(100 * (i / 50))

        Sodiumn = []
        for i in self.df_vietnam_missing_handled['na']:
            Sodiumn.append(100 * (i / 200))

        Potassiumn = []
        for i in self.df_vietnam_missing_handled['k']:
            Potassiumn.append(100 * (i / 12))

        Bicarbonaten = []
        for i in self.df_vietnam_missing_handled['hco3']:
            Bicarbonaten.append(100 * (i / 350))

        Chloriden = []
        for i in self.df_vietnam_missing_handled['cl']:
            Chloriden.append(100 * (i / 250))

        Sulphaten = []
        for i in self.df_vietnam_missing_handled['so4']:
            Sulphaten.append(100 * (i / 200))

        Nitritn = []
        for i in self.df_vietnam_missing_handled['no2']:
            Nitritn.append(100 * (i / 200))

        Nitratn = []
        for i in self.df_vietnam_missing_handled['no3']:
            Nitratn.append(100 * (i / 200))

        wqi = []
        for i in range(len(self.df_vietnam_missing_handled)):
            wqi.append(((pHn[i] * wi[0]) + (TDSn[i] * wi[1]) + (Magnesiumn[i] * wi[2])
                        + (Sodiumn[i] * wi[3]) + (Potassiumn[i] * wi[4]) + (Bicarbonaten[i] * wi[5])
                        + (Chloriden[i] * wi[6]) + (Sulphaten[i] * wi[7]) + (Nitritn[i] * wi[8])
                        + (Nitratn[i] * wi[9])) /
                       (wi[0] + wi[1] + wi[2] + wi[3] + wi[4] + wi[5] + wi[6] + wi[7] + wi[8] + wi[9]))

        self.df_vietnam_missing_handled['WQI'] = wqi

    def _define_wqc(self):
        wqc = []
        for i in range(len(self.df_vietnam_missing_handled)):
            res = ""
            if 0 <= self.df_vietnam_missing_handled["WQI"][i] <= 25:
                res = "Excellent"
            elif 25 < self.df_vietnam_missing_handled["WQI"][i] <= 50:
                res = "Good"
            elif 50 < self.df_vietnam_missing_handled["WQI"][i] <= 75:
                res = "Poor"
            elif self.df_vietnam_missing_handled["WQI"][i] > 75:
                res = "Very Poor"
            wqc.append(res)
        self.df_vietnam_missing_handled['WQC'] = wqc

    def _process_data(self):
        self.output_label_encoder_vietnam = LabelEncoder()
        self.df_vietnam_missing_handled_label_encoded = self.df_vietnam_missing_handled.copy()
        self.df_vietnam_missing_handled_label_encoded["WQC"] = self.output_label_encoder_vietnam.fit_transform(
            self.df_vietnam_missing_handled["WQC"].astype(str))

        # Get column names with integer, float, and datetime data types
        self.integer_columns_vietnam = self.df_vietnam_missing_handled.select_dtypes(include=['int']).columns.tolist()
        self.float_columns_vietnam = self.df_vietnam_missing_handled.select_dtypes(include=['float']).columns.tolist()
        self.datetime_columns_vietnam = self.df_vietnam_missing_handled.select_dtypes(
            include=['datetime']).columns.tolist()

        self.scalable_features = self.float_columns_vietnam + \
                                 self.datetime_columns_vietnam

        # Convert lists to sets
        set_A = set(self.scalable_features)
        set_B = set(self.df_vietnam_categorical_columns)

        # Find the intersection
        intersection = set_A.intersection(set_B)

        # Remove elements from list A that are in the intersection
        self.scalable_features_revised_vietnam = [item for item in self.scalable_features if item not in intersection]

        # Initialize MinMaxScaler
        scaler_vietnam = MinMaxScaler()

        self.df_vietnam_missing_handled_label_encoded_feat_scaled = self.df_vietnam_missing_handled_label_encoded.copy()
        self.scaled_features_vietnam = scaler_vietnam.fit_transform(
            self.df_vietnam_missing_handled_label_encoded[self.scalable_features_revised_vietnam])
        self.df_vietnam_missing_handled_label_encoded_feat_scaled[
            self.scalable_features_revised_vietnam] = self.scaled_features_vietnam

        self.y_vietnam_clf = self.df_vietnam_missing_handled_label_encoded["WQC"].values
        self.y_vietnam_reg = self.df_vietnam_missing_handled_label_encoded["WQI"].values
        self.feat_cols_vietnam = ['well_code', 'date_sampling', 'quarter', 'type_analyzing',
                                  'date_analyzing', 'laboratory', 'number_analyzing', 'na', 'k', 'ca2',
                                  'mg2', 'fe3', 'fe2', 'al3', 'cl', 'so4', 'hco3', 'co3', 'no2',
                                  'hardness_general', 'no3', 'hardness_temporal', 'hardness_permanent',
                                  'ph', 'co2_free', 'co2_depend', 'co2_infiltrate', 'sio2', 'color',
                                  'smell', 'tatse', 'tds105']

        self.x_vietnam = self.df_vietnam_missing_handled_label_encoded_feat_scaled[self.feat_cols_vietnam].values

        self.y_vietnam_clf = self.y_vietnam_clf.reshape(-1, 1)
        self.y_vietnam_reg = self.y_vietnam_reg.reshape(-1, 1)
        self.feat_cols_vietnam_arr = np.array(self.feat_cols_vietnam)

        # Save arrays to a .npz file
        np.savez("dataset/data_vietnam.npz",
                 x=self.x_vietnam,
                 y_classification=self.y_vietnam_clf,
                 y_regression=self.y_vietnam_reg,
                 columns=self.feat_cols_vietnam_arr)

    def run(self):
        self._extract_paths()
        self._extract_files()
        self._explore_files()
        self._preprocess_dataset()
        self._define_wqi()
        self._define_wqc()
        self._process_data()


class IndianDataset:
    def __init__(self, paths):
        self.paths = paths

    def _extract_paths(self):
        self.main_data_path = self.paths["main_data_path"]

    def _extract_files(self):
        # print("Starting Extraction")
        # print("_" * 50)
        #
        # print("Extracting main data for Indian dataset...")
        # print("_" * 20)
        self.df_indian = load_csv_data(self.main_data_path)

        # print("Extraction Done.")
        # print("_" * 50)

    def _explore_files(self):
        print("Starting Exploration...")
        print("_" * 50)

        print("Exploring main data for Indian dataset...")
        print("_" * 20)
        print("Missing Values per Column")
        print("-" * 15)
        missing_values_per_column(self.df_indian)
        print("Information")
        print("-" * 15)
        get_info(self.df_indian)

        print("Exploration Done.")
        print("_" * 50)

    def _preprocess_dataset(self):
        # self.df_indian_new = self.df_indian.drop(columns=self.df_indian.columns[0])
        self.df_indian_new = self.df_indian
        self.df_indian_encoded, self.df_indian_categorical_columns = encode_categorical(self.df_indian_new)
        self.df_indian_missing_handled = handle_missing_values(self.df_indian_encoded)

    def _define_wqi(self):
        #  standard value recommended for parameter
        si_indian = 8.5 + 1000 + 500 + 300 + 200 + 75 + 30 + 200 + 12 + 350 + 250 + 200 + 1.5
        k_indian = 1 / si_indian

        values_indian = [8.5, 1000, 500, 300, 200, 75, 30, 200, 12, 350, 250, 200, 1.5]
        wi_indian = []
        for i_indian in values_indian:
            wi_indian.append(k_indian / i_indian)

        pHn_indian = []
        for i in self.df_indian_missing_handled['pH']:
            pHn_indian.append(100 * ((i - 7.0) / (8.5 - 7.0)))

        ECn_indian = []
        for i in self.df_indian_missing_handled['EC']:
            ECn_indian.append(100 * (i / 1000))

        TDSn_indian = []
        for i in self.df_indian_missing_handled['TDS']:
            TDSn_indian.append(100 * (i / 500))

        THn_indian = []
        for i in self.df_indian_missing_handled['TH']:
            THn_indian.append(100 * (i / 300))

        Alkalinityn_indian = []
        for i in self.df_indian_missing_handled['Alkalinity']:
            Alkalinityn_indian.append(100 * (i / 200))

        Calciumn_indian = []
        for i in self.df_indian_missing_handled['Calcium']:
            Calciumn_indian.append(100 * (i / 75))

        Magnesiumn_indian = []
        for i in self.df_indian_missing_handled['Magnesium']:
            Magnesiumn_indian.append(100 * (i / 50))

        Sodiumn_indian = []
        for i in self.df_indian_missing_handled['Sodium']:
            Sodiumn_indian.append(100 * (i / 200))

        Potassiumn_indian = []
        for i in self.df_indian_missing_handled['Potassium']:
            Potassiumn_indian.append(100 * (i / 12))

        Bicarbonaten_indian = []
        for i in self.df_indian_missing_handled['Bicarbonate']:
            Bicarbonaten_indian.append(100 * (i / 350))

        Chloriden_indian = []
        for i in self.df_indian_missing_handled['Chloride']:
            Chloriden_indian.append(100 * (i / 250))

        Sulphaten_indian = []
        for i in self.df_indian_missing_handled['Sulphate']:
            Sulphaten_indian.append(100 * (i / 200))

        Fluoriden_indian = []
        for i in self.df_indian_missing_handled['Fluoride']:
            Fluoriden_indian.append(100 * (i / 1.5))

        wqi_indian = []
        for i in range(len(self.df_indian_missing_handled)):
            wqi_indian.append(((pHn_indian[i] * wi_indian[0]) + (ECn_indian[i] * wi_indian[1]) + (
                    TDSn_indian[i] * wi_indian[2]) + (THn_indian[i] * wi_indian[3]) + (
                                       Alkalinityn_indian[i] * wi_indian[4]) + (
                                       Calciumn_indian[i] * wi_indian[5]) + (
                                       Magnesiumn_indian[i] * wi_indian[6]) + (Sodiumn_indian[i] * wi_indian[7]) + (
                                       Potassiumn_indian[i] * wi_indian[8]) + (
                                       Bicarbonaten_indian[i] * wi_indian[9]) + (
                                       Chloriden_indian[i] * wi_indian[10]) + (
                                       Sulphaten_indian[i] * wi_indian[11]) + (
                                       Fluoriden_indian[i] * wi_indian[12])) / (
                                      wi_indian[0] + wi_indian[1] + wi_indian[2] + wi_indian[3] + wi_indian[4] +
                                      wi_indian[5] + wi_indian[6] + wi_indian[7] + wi_indian[8] + wi_indian[9] +
                                      wi_indian[10] + wi_indian[11] + wi_indian[12]))
        self.df_indian_missing_handled['WQI'] = wqi_indian

    def _define_wqc(self):
        # Defining Water quality classes
        wqc_indian = []
        for i in range(len(self.df_indian_missing_handled)):
            res = ""
            if 0 <= self.df_indian_missing_handled["WQI"][i] <= 25:
                res = "Excellent"
            elif 25 < self.df_indian_missing_handled["WQI"][i] <= 50:
                res = "Good"
            elif 50 < self.df_indian_missing_handled["WQI"][i] <= 75:
                res = "Poor"
            elif self.df_indian_missing_handled["WQI"][i] > 75:
                res = "Very Poor"
            wqc_indian.append(res)
        self.df_indian_missing_handled['WQC'] = wqc_indian

    def _process_data(self):
        self.output_label_encoder_indian = LabelEncoder()

        self.df_indian_missing_handled_label_encoded = self.df_indian_missing_handled.copy()
        self.df_indian_missing_handled_label_encoded["WQC"] = self.output_label_encoder_indian.fit_transform(
            self.df_indian_missing_handled["WQC"].astype(str))

        # Get column names with integer, float, and datetime data types
        self.integer_columns_indian = self.df_indian_missing_handled.select_dtypes(include=['int']).columns.tolist()
        self.float_columns_indian = self.df_indian_missing_handled.select_dtypes(include=['float']).columns.tolist()
        self.datetime_columns_indian = self.df_indian_missing_handled.select_dtypes(
            include=['datetime']).columns.tolist()

        self.scalable_features_indian = self.float_columns_indian + self.datetime_columns_indian

        # Convert lists to sets
        set_A = set(self.scalable_features_indian)
        set_B = set(self.df_indian_categorical_columns)

        # Find the intersection
        intersection = set_A.intersection(set_B)

        # Remove elements from list A that are in the intersection
        self.scalable_features_revised_indian = [item for item in self.scalable_features_indian
                                                 if item not in intersection]

        # Initialize MinMaxScaler
        scaler_indian = MinMaxScaler()

        self.df_indian_missing_handled_label_encoded_feat_scaled = self.df_indian_missing_handled_label_encoded.copy()
        self.scaled_features_indian = scaler_indian.fit_transform(
            self.df_indian_missing_handled_label_encoded[self.scalable_features_revised_indian])
        self.df_indian_missing_handled_label_encoded_feat_scaled[
            self.scalable_features_revised_indian] = self.scaled_features_indian

        self.y_indian_clf = self.df_indian_missing_handled_label_encoded["WQC"].values
        self.y_indian_reg = self.df_indian_missing_handled_label_encoded["WQI"].values
        self.feat_cols_indian = ['District', 'Village', 'pH', 'EC', 'TDS', 'TH', 'Alkalinity', 'Calcium',
                                 'Magnesium', 'Sodium', 'Potassium', 'Bicarbonate', 'Chloride',
                                 'Sulphate', 'Fluoride']

        self.x_indian = self.df_indian_missing_handled_label_encoded_feat_scaled[self.feat_cols_indian].values

        self.y_indian_clf = self.y_indian_clf.reshape(-1, 1)
        self.y_indian_reg = self.y_indian_reg.reshape(-1, 1)
        self.feat_cols_indian_arr = np.array(self.feat_cols_indian)

        # Save arrays to a .npz file
        np.savez("dataset/data_indian.npz",
                 x=self.x_indian,
                 y_classification=self.y_indian_clf,
                 y_regression=self.y_indian_reg,
                 columns=self.feat_cols_indian_arr)

    def run(self):
        self._extract_paths()
        self._extract_files()
        self._explore_files()
        self._preprocess_dataset()
        self._define_wqi()
        self._define_wqc()
        self._process_data()
