import pandas as pd
class DataLoader:
    def __init__(self, fraud_path, ip_path, credit_path):
        self.fraud_path = fraud_path
        self.ip_path = ip_path
        self.credit_path = credit_path

    def load_fraud_data(self):
        """
        Load fraud data from CSV and cache it in memory.
        """
        try:

            return pd.read_csv(self.fraud_path, parse_dates=['signup_time', 'purchase_time'])

        except Exception as e:
            print(f"File failed to load:{e}")


    def load_creditcard_data(self):
        """
        Load credit data from CSV and cache it in memory.
        """
        try:

            return pd.read_csv(self.credit_path)

        except Exception as e:
            print(f"File failed to load:{e}")

    def load_ipaddress_data(self):
        """
        Load ipaddress data from CSV and cache it in memory.
        """
        try:

            return pd.read_csv(self.ip_path)

        except Exception as e:
            print(f"File failed to load:{e}")
