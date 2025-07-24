import pandas as pd

class FeatureEngineer:
    def __init__(self, df, ip_df=None):
        """
        Feature engineering for fraud detection dataset.
        :param df: main DataFrame (fraud data)
        :param ip_df: IP to country mapping DataFrame (optional)
        """
        self.df = df.copy()
        self.ip_df = ip_df.copy() if ip_df is not None else None

    def add_time_features(self):
        """Add hour_of_day, day_of_week, and time_since_signup."""
        if 'purchase_time' in self.df.columns and 'signup_time' in self.df.columns:
            self.df['hour_of_day'] = self.df['purchase_time'].dt.hour
            self.df['day_of_week'] = self.df['purchase_time'].dt.dayofweek
            self.df['time_since_signup'] = (
                self.df['purchase_time'] - self.df['signup_time']
            ).dt.total_seconds()
        else:
            print("Time columns not found. Skipping time features.")

    def add_transaction_velocity(self):
        """Add user and device transaction frequency features."""
        if 'user_id' in self.df.columns:
            user_freq = self.df['user_id'].value_counts()
            self.df['user_transaction_count'] = self.df['user_id'].map(user_freq)
        if 'device_id' in self.df.columns:
            device_freq = self.df['device_id'].value_counts()
            self.df['device_transaction_count'] = self.df['device_id'].map(device_freq)

    def map_ip_to_country(self):
        # """Map numeric IPs to country using IP range lookup."""
        # if self.ip_df is None:
        #     print("No IP mapping file provided. Skipping.")
        #     self.df['country'] = 'Unknown'
        #     return

        # # Convert IPs to integers
        # print("Converting IP addresses to integers...")
        # print(self.df["ip_address"].dtype)
        # self.df['ip_int'] = pd.to_numeric(self.df['ip_address'], errors='coerce').astype('Int64')
        # self.ip_df['lower_bound_int'] = pd.to_numeric(self.ip_df['lower_bound_ip_address'], errors='coerce').astype('Int64')
        # self.ip_df['upper_bound_int'] = pd.to_numeric(self.ip_df['upper_bound_ip_address'], errors='coerce').astype('Int64')

        # # Drop nulls from IP columns
        # self.df.dropna(subset=['ip_int'], inplace=True)
        # self.ip_df.dropna(subset=['lower_bound_int', 'upper_bound_int'], inplace=True)

        # # Sort for merge_asof
        # self.df.sort_values('ip_int', inplace=True)
        # self.ip_df.sort_values('lower_bound_int', inplace=True)

        # # Merge using merge_asof
        # print("Merging IP ranges...")
        # merged = pd.merge_asof(
        #     self.df,
        #     self.ip_df,
        #     left_on='ip_int',
        #     right_on='lower_bound_int',
        #     direction='backward'
        # )

        # # Keep only valid IPs
        # merged = merged[merged['ip_int'] <= merged['upper_bound_int']]

        # if merged.empty:
        #     print("Warning: No IPs matched any country range. Setting 'Unknown'.")
        #     self.df['country'] = 'Unknown'
        # else:
        #     self.df['country'] = merged['country'].values

        # self.df.drop(columns=['ip_int'], inplace=True)
        pass
    def get_engineered_data(self):
        """Return engineered dataset."""
        self.add_time_features()
        self.add_transaction_velocity()
        self.map_ip_to_country()

        return self.df


# testing 
# df = pd.read_csv('data/Fraud_Data.csv', parse_dates=['purchase_time', "signup_time"])
# ip = pd.read_csv('data/IpAddress_to_Country.csv')

# print("Testing")
# f = FeatureEngineer(df,ip)
# print(df.head())
# df = f.get_engineered_data()
# print(df.head())


