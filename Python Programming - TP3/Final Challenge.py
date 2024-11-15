#Challenge

import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json
import os

@dataclass
class SalesAnalysis:
    """Class to hold sales analysis results"""
    daily_revenue: pd.Series
    product_performance: pd.DataFrame
    best_seller: str
    total_revenue: float
    revenue_by_product: pd.Series

class SalesAnalysisSystem:
    def __init__(self, db_name: str = "sales.db"):
        """Initialize the sales analysis system."""
        self.db_name = db_name
        self.initialize_database()
        
    def initialize_database(self):
        """Create the sales database and table."""
        try:
            with sqlite3.connect(self.db_name) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS sales (
                        id INTEGER PRIMARY KEY,
                        date DATE NOT NULL,
                        product TEXT NOT NULL,
                        quantity INTEGER NOT NULL,
                        price DECIMAL(10,2) NOT NULL,
                        revenue DECIMAL(10,2) NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                conn.commit()
        except sqlite3.Error as e:
            print(f"Database initialization error: {str(e)}")

    def load_sales_data(self, sales_data: Dict[str, List]) -> pd.DataFrame:
        """Load sales data into DataFrame and database."""
        try:
            df = pd.DataFrame(sales_data)
            df['date'] = pd.to_datetime(df['date'])
            df['revenue'] = df['quantity'] * df['price']

            with sqlite3.connect(self.db_name) as conn:
                df.to_sql('sales', conn, if_exists='append', index=False)
            
            return df
        except Exception as e:
            print(f"Error loading sales data: {str(e)}")
            return pd.DataFrame()

    def analyze_sales(self, df: pd.DataFrame) -> SalesAnalysis:
        """Perform comprehensive sales analysis."""

        daily_revenue = df.groupby('date')['revenue'].sum()

        product_performance = df.groupby('product').agg({
            'quantity': 'sum',
            'revenue': 'sum'
        }).sort_values('revenue', ascending=False)

        best_seller = product_performance['quantity'].idxmax()

        total_revenue = df['revenue'].sum()

        revenue_by_product = df.groupby('product')['revenue'].sum()
        
        return SalesAnalysis(
            daily_revenue=daily_revenue,
            product_performance=product_performance,
            best_seller=best_seller,
            total_revenue=total_revenue,
            revenue_by_product=revenue_by_product
        )

    def create_visualizations(self, analysis: SalesAnalysis, df: pd.DataFrame, save_path: str = 'sales_analysis_plots.pdf'):
        """Create and save sales visualizations."""
        plt.style.use('seaborn')
        fig = plt.figure(figsize=(15, 10))
        
        ''' 1. Daily Revenue Trend '''
        plt.subplot(2, 2, 1)
        self._plot_revenue_trend(analysis.daily_revenue)
        
        ''' 2. Product Revenue Distribution '''
        plt.subplot(2, 2, 2)
        self._plot_product_revenue(analysis.revenue_by_product)
        
        ''' 3. Quantity vs Revenue Scatter '''
        plt.subplot(2, 2, 3)
        self._plot_quantity_revenue_scatter(df)
        
        ''' 4. Product Performance Comparison '''
        plt.subplot(2, 2, 4)
        self._plot_product_comparison(analysis.product_performance)
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def _plot_revenue_trend(self, daily_revenue: pd.Series):
        """Plot daily revenue trend."""
        daily_revenue.plot(kind='line', marker='o')
        plt.title('Daily Revenue Trend')
        plt.xlabel('Date')
        plt.ylabel('Revenue')
        plt.xticks(rotation=45)
        
    def _plot_product_revenue(self, revenue_by_product: pd.Series):
        """Plot revenue distribution by product."""
        revenue_by_product.plot(kind='pie', autopct='%1.1f%%')
        plt.title('Revenue Distribution by Product')
        
    def _plot_quantity_revenue_scatter(self, df: pd.DataFrame):
        """Create quantity vs revenue scatter plot."""
        plt.scatter(df['quantity'], df['revenue'])
        plt.xlabel('Quantity')
        plt.ylabel('Revenue')
        plt.title('Quantity vs Revenue')
        
    def _plot_product_comparison(self, product_performance: pd.DataFrame):
        """Plot product performance comparison."""
        product_performance.plot(kind='bar')
        plt.title('Product Performance Comparison')
        plt.xlabel('Product')
        plt.xticks(rotation=45)

    def export_results(self, df: pd.DataFrame, analysis: SalesAnalysis, filename: str = 'sales_analysis_results.xlsx'):
        """Export analysis results to Excel."""
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:

            df.to_excel(writer, sheet_name='Raw Data', index=False)

            analysis.daily_revenue.to_excel(writer, sheet_name='Daily Revenue')

            analysis.product_performance.to_excel(writer, sheet_name='Product Performance')

            summary = pd.DataFrame({
                'Metric': ['Total Revenue', 'Best Selling Product', 'Number of Products'],
                'Value': [
                    f"${analysis.total_revenue:,.2f}",
                    analysis.best_seller,
                    len(analysis.revenue_by_product)
                ]
            })
            summary.to_excel(writer, sheet_name='Summary', index=False)

def main():
    
    sales_data = {
        'date': ['2024-01-01', '2024-01-02', '2024-01-03'],
        'product': ['A', 'B', 'A'],
        'quantity': [10, 15, 12],
        'price': [100, 150, 100]
    }
    

    system = SalesAnalysisSystem()

    print("Loading sales data...")
    df = system.load_sales_data(sales_data)
    
    print("\nAnalyzing sales data...")
    analysis = system.analyze_sales(df)

    print("\n=== Sales Analysis Report ===")
    print(f"\nTotal Revenue: ${analysis.total_revenue:,.2f}")
    print(f"Best Selling Product: {analysis.best_seller}")
    
    print("\nDaily Revenue:")
    print(analysis.daily_revenue)
    
    print("\nProduct Performance:")
    print(analysis.product_performance)

    print("\nGenerating visualizations...")
    system.create_visualizations(analysis, df)
    print("Visualizations saved to 'sales_analysis_plots.pdf'")

    print("\nExporting results...")
    system.export_results(df, analysis)
    print("Analysis results exported to 'sales_analysis_results.xlsx'")

if __name__ == "__main__":
    main()

'''
getting some help from AI to complete it ... not fully working though
'''