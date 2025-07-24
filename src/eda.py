import seaborn as sns
import matplotlib.pyplot as plt
class EDAExplorer:
    def __init__(self, df, target_column):
        self.df = df
        self.target = target_column

    def univariate_summary(self):
        print(self.df.describe())

    def plot_class_distribution(self):
        sns.countplot(x=self.target, data=self.df)
        plt.title("Class Distribution")


    def bivariate_plots(self,feature):
        sns.boxplot(x=self.target, y=feature, data=self.df)
