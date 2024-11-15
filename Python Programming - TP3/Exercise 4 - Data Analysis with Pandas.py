#Exercise 4

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
from dataclasses import dataclass

@dataclass
class StudentAnalysis:
    """Class to hold student analysis results"""
    avg_scores: pd.Series
    high_performers: pd.DataFrame
    sorted_students: pd.DataFrame
    subject_stats: pd.DataFrame
    correlation_matrix: pd.DataFrame

class StudentPerformanceAnalyzer:
    def __init__(self, data: Dict[str, List]):
        """Initialize the analyzer with student data."""
        self.df = pd.DataFrame(data)
        self.df['average_score'] = self.df[['math_score', 'science_score']].mean(axis=1)
        
    def perform_analysis(self) -> StudentAnalysis:
        """Perform all analyses and return results."""
        return StudentAnalysis(
            avg_scores=self._calculate_average_scores(),
            high_performers=self._find_high_performers(),
            sorted_students=self._sort_by_average(),
            subject_stats=self._calculate_subject_stats(),
            correlation_matrix=self._calculate_correlations()
        )
    
    def _calculate_average_scores(self) -> pd.Series:
        """Calculate average score for each student."""
        return self.df['average_score']
    
    def _find_high_performers(self, threshold: float = 85) -> pd.DataFrame:
        """Find students with average score above threshold."""
        return self.df[self.df['average_score'] > threshold]
    
    def _sort_by_average(self) -> pd.DataFrame:
        """Sort students by average score in descending order."""
        return self.df.sort_values('average_score', ascending=False)
    
    def _calculate_subject_stats(self) -> pd.DataFrame:
        """Calculate basic statistics for each subject."""
        return self.df[['math_score', 'science_score']].describe()
    
    def _calculate_correlations(self) -> pd.DataFrame:
        """Calculate correlation matrix for numerical columns."""
        return self.df[['age', 'math_score', 'science_score', 'average_score']].corr()

    def create_visualizations(self, save_path: str = 'student_analysis_plots.pdf'):
        """Create and save all visualizations."""
        plt.style.use('seaborn')
        fig = plt.figure(figsize=(15, 10))
        
        ''' 1. Score Distribution '''
        plt.subplot(2, 2, 1)
        self._plot_score_distribution()
        
        ''' 2. Math vs Science Comparison '''
        plt.subplot(2, 2, 2)
        self._plot_score_comparison()
        
        ''' 3. Student Performance Overview '''
        plt.subplot(2, 2, 3)
        self._plot_student_overview()
        
        ''' 4. Correlation Heatmap '''
        plt.subplot(2, 2, 4)
        self._plot_correlation_heatmap()
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        
    def _plot_score_distribution(self):
        """Plot score distribution for each subject."""
        scores = pd.melt(self.df[['math_score', 'science_score']], 
                        var_name='Subject', value_name='Score')
        sns.boxplot(x='Subject', y='Score', data=scores)
        plt.title('Score Distribution by Subject')
        
    def _plot_score_comparison(self):
        """Create scatter plot comparing math and science scores."""
        plt.scatter(self.df['math_score'], self.df['science_score'])
        plt.xlabel('Math Score')
        plt.ylabel('Science Score')
        plt.title('Math vs Science Scores')

        min_score = min(self.df['math_score'].min(), self.df['science_score'].min())
        max_score = max(self.df['math_score'].max(), self.df['science_score'].max())
        plt.plot([min_score, max_score], [min_score, max_score], 'r--', alpha=0.5)
        
    def _plot_student_overview(self):
        """Create bar plot showing each student's performance."""
        self.df.plot(kind='bar', x='name', y=['math_score', 'science_score'], 
                    ax=plt.gca())
        plt.title('Student Performance Overview')
        plt.xticks(rotation=45)
        
    def _plot_correlation_heatmap(self):
        """Create correlation heatmap."""
        sns.heatmap(self._calculate_correlations(), annot=True, cmap='coolwarm')
        plt.title('Correlation Matrix')

def main():
    student_data = {
        'name': ['John', 'Alice', 'Bob', 'Sarah', 'Mike'],
        'age': [20, 21, 19, 22, 20],
        'math_score': [85, 90, 75, 95, 80],
        'science_score': [75, 95, 80, 85, 90],
        'passed': [True, True, False, True, True]
    }
    
    analyzer = StudentPerformanceAnalyzer(student_data)
    
    analysis = analyzer.perform_analysis()
    
    print("\n=== Student Performance Analysis Report ===\n")
    
    print("Average Scores:")
    for name, avg in zip(analyzer.df['name'], analysis.avg_scores):
        print(f"{name}: {avg:.2f}")
    
    print("\nHigh Performers (>85 average):")
    print(analysis.high_performers[['name', 'average_score']])
    
    print("\nStudents Sorted by Average Score:")
    print(analysis.sorted_students[['name', 'average_score']])
    
    print("\nSubject Statistics:")
    print(analysis.subject_stats)
    
    print("\nCorrelation Matrix:")
    print(analysis.correlation_matrix)
    
    analyzer.create_visualizations()
    print("\nVisualizations saved to 'student_analysis_plots.pdf'")
    
    with pd.ExcelWriter('student_analysis_results.xlsx') as writer:
        analyzer.df.to_excel(writer, sheet_name='Raw Data', index=False)
        analysis.high_performers.to_excel(writer, sheet_name='High Performers', index=False)
        analysis.subject_stats.to_excel(writer, sheet_name='Statistics')
        analysis.correlation_matrix.to_excel(writer, sheet_name='Correlations')
    
    print("Analysis results exported to 'student_analysis_results.xlsx'")

if __name__ == "__main__":
    main()

'''
getting some help from AI to complete it ... not fully working though
'''