import openai
from openai import OpenAI
from itertools import combinations
import random
from text_generation import AI
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
client = OpenAI(api_key="#############################")
import importlib
def get_ai_instance():
    from text_generation import AI
    return AI()

ai = get_ai_instance()
class BivariateAnalyzer1:
    def __init__(self, df, dataset_name):
        self.df = df
        self.dataset_name = dataset_name
    def analyze(self):
        analysis_results = {}
        for column1 in self.df.columns:
            for column2 in self.df.columns:
                if column1 != column2:
                    result = self.analyze_columns(column1, column2)
                    if result is not None:
                        analysis_results[(column1, column2)] = result
        return analysis_results

    def analyze_columns(self, column1, column2):
        series1 = self.df[column1]
        series2 = self.df[column2]
        if pd.api.types.is_numeric_dtype(series1) and pd.api.types.is_numeric_dtype(series2):
            correlation = series1.corr(series2)
            return {'correlation': correlation}
        elif pd.api.types.is_numeric_dtype(series1) and pd.api.types.is_categorical_dtype(series2):
            grouped_mean = series1.groupby(series2).mean()
            return {'grouped_mean': grouped_mean}
        elif pd.api.types.is_categorical_dtype(series1) and pd.api.types.is_numeric_dtype(series2):
            grouped_mean = series2.groupby(series1).mean()
            return {'grouped_mean': grouped_mean}
        elif pd.api.types.is_categorical_dtype(series1) and pd.api.types.is_categorical_dtype(series2):
            unique_combinations = self.df.groupby([series1.name, series2.name]).size()
            return {'unique_combinations': unique_combinations}

class BivariateAnalyzer:
    def __init__(self, df, bi_gpt, bi_columns):
        self.df = df
        self.bi_gpt = bi_gpt
        self.bi_columns = bi_columns

    def visualize(self):
        with PdfPages('Bi_variate_output.pdf') as pdf:
            for column_pair_dict in self.bi_columns:
                for column_pair in column_pair_dict.items():
                    fig, axs = plt.subplots(2, 1, figsize=(6, 4))
                    column1, column2 = column_pair
                    print(f"Processing columns: {column1}, {column2}")  # Debug print
                    if pd.api.types.is_numeric_dtype(self.df[column1]) and pd.api.types.is_numeric_dtype(self.df[column2]):
                        print(f"Creating scatterplot for {column1} and {column2}")  # Debug print
                        sns.scatterplot(data=self.df, x=column1, y=column2, ax=axs[0])
                        axs[0].set_title(f"Relationship between {column1} and {column2}")
                    elif pd.api.types.is_numeric_dtype(self.df[column1]) and pd.api.types.is_categorical_dtype(self.df[column2]):
                        print(f"Creating boxplot for {column1} (numeric) and {column2} (categorical)")  # Debug print
                        sns.boxplot(x=column2, y=column1, data=self.df, ax=axs[0])
                        axs[0].set_title(f"Relationship between {column1} (numeric) and {column2} (categorical)")
                    elif pd.api.types.is_categorical_dtype(self.df[column1]) and pd.api.types.is_numeric_dtype(self.df[column2]):
                        print(f"Creating boxplot for {column1} (categorical) and {column2} (numeric)")  # Debug print
                        sns.boxplot(x=column1, y=column2, data=self.df, ax=axs[0])
                        axs[0].set_title(f"Relationship between {column1} (categorical) and {column2} (numeric)")
                    else:
                        print(f"Creating countplot for {column1} and {column2} (both categorical)")  # Debug print
                        sns.countplot(x=column1, hue=column2, data=self.df, ax=axs[0])
                        axs[0].set_title(f"Relationship between {column1} and {column2} (both categorical)")

                    axs[1].text(0.5, 0.5, self.bi_gpt[(column1, column2)], wrap=True, horizontalalignment='center', verticalalignment='center', fontsize=8)
                    axs[1].axis('off')  # Hide the axes

                    pdf.savefig(fig)  # saves the current figure into a pdf page
                    plt.close()
                    
def bi_analyze(df, dataset_name, bi_columns, analysis_results):
    bi_descriptions = {} 
    max_unique_values = 10
    object_columns={}
    for col in df.columns:
                unique_values = df[col].unique().tolist()
                if len(unique_values) > max_unique_values:
                    unique_values = random.sample(unique_values, max_unique_values)
                    object_columns[col] = unique_values
    for bi_column in bi_columns:
        for column1, column2 in bi_column.items():
            uni1 = object_columns.get(column1)
            uni2 = object_columns.get(column2)
            stats = analysis_results.get((column1, column2), {})
            prompt = f"these are unique values in {column1} {uni1} and {column2} {uni2} in the dataset. Please generate a description only about the relationship (for bivariate analysis) between {column1} and {column2}  by using {stats} in simple words or in natural language for my Graph. Start with explaining their relationship and how to they are related only and don't describe about dataset starts with the relationship between these columns are tend to be like that. Don't return all the unique values of columns that are given thats are only for reference. "
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "system", "content": prompt}],
                max_tokens=300,
                temperature=0.2,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0
            )
            
            bi_descriptions[(column1, column2)] = response.choices[0].message.content
            

    return bi_descriptions
def bi_visualize_analyze(df, dataset_name, target_variable):
    ai_instance = AI()
    uni_poss_corr = ai_instance.uni_poss_corr
    
    bi_columns = ai.bi_poss_corr
    analyzer = BivariateAnalyzer1(df, dataset_name)
    analysis_results = analyzer.analyze()
    bi_descriptions = bi_analyze(df, dataset_name, bi_columns, analysis_results)
    visualizer = BivariateAnalyzer(df, bi_descriptions, bi_columns)
    visualizer.visualize()
def get_analysis_results(df, dataset_name):
    analyzer = BivariateAnalyzer1(df, dataset_name)
    return analyzer.analyze()