#Exercise 3
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import os
from dotenv import load_dotenv

class TMDBAnalyzer:
    def __init__(self):
        '''it is actually a proprietary api so we need an api key on env file but we don't have it so we'll just pretend for now...'''
        load_dotenv()
        self.api_key = os.getenv('TMDB_API_KEY')
        if not self.api_key:
            raise ValueError("TMDB API key not found. Please set TMDB_API_KEY environment variable.")
        
        self.base_url = "https://api.themoviedb.org/3"
        self.endpoint = "/movie/top_rated"
        
    def fetch_movies(self, num_movies: int = 20) -> List[Dict]:
        """Fetch top rated movies from TMDB API."""
        movies = []
        page = 1
        
        try:
            while len(movies) < num_movies:
                '''get the infos from this link : https://developer.themoviedb.org/docs/search-and-query-for-details''' 
                params = {
                    'api_key': self.api_key,
                    'language': 'en-US',
                    'page': page
                }
                
                response = requests.get(f"{self.base_url}{self.endpoint}", params=params)
                response.raise_for_status()
                
                data = response.json()
                movies.extend(data['results'])
                page += 1
                
            return movies[:num_movies]
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data from TMDB: {str(e)}")
            return []

    def process_movies(self, movies: List[Dict]) -> pd.DataFrame:
        """Extract relevant information and create DataFrame."""
        processed_data = []
        
        for movie in movies:
            processed_data.append({
                'title': movie['title'],
                'release_date': movie['release_date'],
                'vote_average': movie['vote_average'],
                'popularity': movie['popularity'],
                'overview': movie['overview']
            })
            
        return pd.DataFrame(processed_data)

    def analyze_movies(self, df: pd.DataFrame) -> Dict:
        """Perform various analyses on the movie data."""
        current_year = datetime.now().year
        df['release_year'] = pd.to_datetime(df['release_date']).dt.year
        
        analysis = {
            'average_rating': df['vote_average'].mean(),
            'highest_rated': df.loc[df['vote_average'].idxmax()],
            'lowest_rated': df.loc[df['vote_average'].idxmin()],
            'recent_movies': df[df['release_year'] >= (current_year - 5)],
            'total_movies': len(df),
            'avg_popularity': df['popularity'].mean()
        }
        
        return analysis

    def format_movie_info(self, movie: pd.Series) -> str:
        """Format movie information for display."""
        return (f"Title: {movie['title']}\n"
                f"Release Date: {movie['release_date']}\n"
                f"Rating: {movie['vote_average']}\n"
                f"Popularity: {movie['popularity']:.2f}\n")

    def generate_report(self, analysis: Dict) -> str:
        """Generate a formatted report of the analysis."""
        report = [
            "=== TMDB Movie Analysis Report ===\n",
            f"Total Movies Analyzed: {analysis['total_movies']}\n",
            f"\nAverage Rating: {analysis['average_rating']:.2f}\n",
            "\nHighest Rated Movie:",
            self.format_movie_info(analysis['highest_rated']),
            "\nLowest Rated Movie:",
            self.format_movie_info(analysis['lowest_rated']),
            f"\nRecent Movies (Last 5 Years): {len(analysis['recent_movies'])}",
            f"\nAverage Popularity Score: {analysis['avg_popularity']:.2f}\n",
            "\nRecent Movies Details:"
        ]
        
        for _, movie in analysis['recent_movies'].iterrows():
            report.append(self.format_movie_info(movie))
        
        return "\n".join(report)

def main():
    try:
        analyzer = TMDBAnalyzer()
        
        print("Fetching top rated movies...")
        movies = analyzer.fetch_movies()
        
        if not movies:
            print("No movies fetched. Exiting.")
            return
        
        print("Processing movie data...")
        df = analyzer.process_movies(movies)
        
        csv_filename = 'top_movies.csv'
        df.to_csv(csv_filename, index=False)
        print(f"Saved movie data to {csv_filename}")
        
        print("\nAnalyzing movie data...")
        analysis = analyzer.analyze_movies(df)
        
        report = analyzer.generate_report(analysis)
        print(report)
        
        report_filename = 'movie_analysis_report.txt'
        with open(report_filename, 'w') as f:
            f.write(report)
        print(f"\nSaved detailed report to {report_filename}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()


'''
this code is not tested, and it's built based on similar code from google and stackoverflow, I don't know if it's working or not
'''