from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import os

app = Flask(__name__)

# Global variable to store dataframe
df = None

def load_data():
    """Load GitHub projects data from CSV file"""
    global df
    try:
        # Try multiple possible filenames
        possible_files = [
            'Github_data.csv',
            'github_data.csv',
            'Github_data',
            'github_dataset.csv'
        ]
        
        csv_path = None
        for filename in possible_files:
            if os.path.exists(filename):
                csv_path = filename
                break
            # Also try with .csv extension if not present
            if not filename.endswith('.csv') and os.path.exists(filename + '.csv'):
                csv_path = filename + '.csv'
                break
        
        if csv_path is None:
            print("ERROR: Dataset file not found!")
            print(f"Looking for files: {possible_files}")
            print(f"Current directory: {os.getcwd()}")
            print(f"Files in current directory: {os.listdir('.')}")
            return None
        
        print(f"Found dataset at: {csv_path}")
        
        # Read the CSV file with error handling
        try:
            df = pd.read_csv(csv_path, encoding='utf-8')
        except UnicodeDecodeError:
            print("UTF-8 failed, trying latin-1 encoding...")
            df = pd.read_csv(csv_path, encoding='latin-1')
        
        # Print column names for debugging
        print("Available columns:", df.columns.tolist())
        print(f"Dataset shape: {df.shape}")
        
        # Handle missing values
        df = df.fillna('')
        
        # Check for required columns with flexible naming
        name_columns = ['name', 'repo_name', 'repository_name', 'project_name']
        desc_columns = ['description', 'desc', 'repo_description']
        
        name_col = None
        desc_col = None
        
        for col in name_columns:
            if col in df.columns:
                name_col = col
                break
        
        for col in desc_columns:
            if col in df.columns:
                desc_col = col
                break
        
        if name_col is None:
            print(f"WARNING: No name column found. Searched for: {name_columns}")
            print(f"Available columns: {df.columns.tolist()}")
            # Use first column as name if no name column found
            if len(df.columns) > 0:
                name_col = df.columns[0]
                print(f"Using '{name_col}' as name column")
        
        if desc_col is None:
            print(f"WARNING: No description column found. Searched for: {desc_columns}")
        
        # Standardize column names
        if name_col and name_col != 'name':
            df['name'] = df[name_col]
        
        if desc_col and desc_col != 'description':
            df['description'] = df[desc_col]
        elif 'description' not in df.columns:
            df['description'] = ''
        
        # Convert numeric columns if they exist
        numeric_mappings = {
            'stars': ['stars', 'stargazers_count', 'star_count', 'stargazers'],
            'forks': ['forks', 'forks_count', 'fork_count'],
            'watchers': ['watchers', 'watchers_count'],
            'open_issues': ['open_issues', 'open_issues_count', 'issues']
        }
        
        for target_col, possible_names in numeric_mappings.items():
            for col_name in possible_names:
                if col_name in df.columns:
                    df[target_col] = pd.to_numeric(df[col_name], errors='coerce').fillna(0).astype(int)
                    break
            if target_col not in df.columns:
                df[target_col] = 0
        
        print(f"Successfully loaded {len(df)} projects from {csv_path}")
        return df
        
    except Exception as e:
        print(f"Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return None

# Load data at startup
try:
    df = load_data()
    if df is None:
        print("=" * 50)
        print("CRITICAL ERROR: Failed to load dataset!")
        print("=" * 50)
except Exception as e:
    print(f"Failed to load dataset: {e}")
    df = None

def preprocess_text(text):
    """Clean and preprocess text"""
    if pd.isna(text) or text == '':
        return ''
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return text

def get_recommendations(query, top_n=5):
    """Get project recommendations based on query"""
    
    if df is None or df.empty:
        return []
    
    # Create a copy to avoid modifying original
    df_work = df.copy()
    
    # Combine relevant text fields for matching
    text_fields = []
    
    # Check for various column names
    if 'description' in df_work.columns:
        text_fields.append(df_work['description'].fillna(''))
    
    if 'topics' in df_work.columns:
        text_fields.append(df_work['topics'].fillna(''))
    
    if 'language' in df_work.columns:
        text_fields.append(df_work['language'].fillna(''))
    
    if 'name' in df_work.columns:
        text_fields.append(df_work['name'].fillna(''))
    
    # If no text fields found, use all string columns
    if not text_fields:
        for col in df_work.columns:
            if df_work[col].dtype == 'object':
                text_fields.append(df_work[col].fillna(''))
    
    # Combine all text fields
    if text_fields:
        df_work['combined_text'] = pd.concat(text_fields, axis=1).apply(lambda x: ' '.join(x.astype(str)), axis=1)
    else:
        return []
    
    df_work['combined_text'] = df_work['combined_text'].apply(preprocess_text)
    
    # Remove empty entries
    df_work = df_work[df_work['combined_text'].str.len() > 0]
    
    if df_work.empty:
        return []
    
    # Preprocess query
    query_processed = preprocess_text(query)
    
    if not query_processed:
        return []
    
    try:
        # Create TF-IDF vectors
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english', min_df=1)
        tfidf_matrix = vectorizer.fit_transform(df_work['combined_text'])
        
        # Transform query
        query_vector = vectorizer.transform([query_processed])
        
        # Calculate similarity scores
        similarity_scores = cosine_similarity(query_vector, tfidf_matrix)[0]
        
        # Get top N recommendations
        top_indices = similarity_scores.argsort()[-top_n:][::-1]
        
        recommendations = []
        for idx, i in enumerate(top_indices):
            if similarity_scores[i] > 0:  # Only include if there's some similarity
                row = df_work.iloc[i]
                
                # Extract topics
                topics = []
                if 'topics' in row and pd.notna(row['topics']) and row['topics']:
                    topics_str = str(row['topics'])
                    # Handle both comma-separated and JSON-like formats
                    topics = re.findall(r'[\w-]+', topics_str)[:5]
                    topics = [t.strip() for t in topics if t.strip()]
                
                # Get owner and create initial
                owner = row.get('owner', row.get('owner_login', row.get('user', 'unknown')))
                if pd.isna(owner) or owner == '':
                    owner = 'unknown'
                owner = str(owner)
                owner_initial = owner[:2].upper() if owner else 'UN'
                
                # Get repository URL
                url = row.get('html_url', row.get('url', row.get('repo_url', '#')))
                if pd.isna(url) or url == '':
                    # Try to construct URL from name and owner
                    name = row.get('name', '')
                    if name and owner != 'unknown':
                        url = f"https://github.com/{owner}/{name}"
                    else:
                        url = '#'
                
                # Get language
                language = row.get('language', row.get('primary_language', 'Unknown'))
                if pd.isna(language) or language == '':
                    language = 'Unknown'
                
                # Get stars
                stars = int(row.get('stars', 0))
                
                # Get forks
                forks = int(row.get('forks', 0))
                
                # Get description
                description = row.get('description', 'No description available')
                if pd.isna(description) or description == '':
                    description = 'No description available'
                
                # Calculate days since update
                updated_days_ago = 0
                if 'updated_at' in row and pd.notna(row['updated_at']):
                    try:
                        updated_date = pd.to_datetime(row['updated_at'])
                        updated_days_ago = (pd.Timestamp.now() - updated_date).days
                    except:
                        updated_days_ago = 0
                elif 'updated_days_ago' in row:
                    updated_days_ago = int(row.get('updated_days_ago', 0))
                
                project = {
                    'rank': idx + 1,
                    'name': str(row.get('name', 'Unknown')),
                    'owner': owner,
                    'description': str(description)[:300],  # Limit description length
                    'topics': topics,
                    'stars': stars,
                    'forks': forks,
                    'language': str(language),
                    'updated_days_ago': updated_days_ago,
                    'url': str(url),
                    'match_percentage': int(similarity_scores[i] * 100),
                    'owner_initial': owner_initial
                }
                recommendations.append(project)
        
        return recommendations
    
    except Exception as e:
        print(f"Error in get_recommendations: {e}")
        import traceback
        traceback.print_exc()
        return []

def determine_badges(stars, updated_days_ago):
    """Determine which badges to show"""
    badges = []
    if stars > 1000:
        badges.append('popular')
    if updated_days_ago <= 7:
        badges.append('active')
    return badges

def format_time_ago(days):
    """Format time ago string"""
    if days == 0:
        return "Updated today"
    elif days == 1:
        return "Updated yesterday"
    elif days < 7:
        return f"Updated {days} days ago"
    elif days < 14:
        return "Updated 1 week ago"
    elif days < 21:
        return "Updated 2 weeks ago"
    elif days < 30:
        return "Updated 3 weeks ago"
    else:
        months = days // 30
        return f"Updated {months} month{'s' if months > 1 else ''} ago"

def get_language_color(language):
    """Get color for programming language"""
    colors = {
        'Python': '#3572A5',
        'JavaScript': '#f1e05a',
        'Java': '#b07219',
        'C++': '#f34b7d',
        'C': '#555555',
        'Ruby': '#701516',
        'Go': '#00ADD8',
        'Rust': '#dea584',
        'TypeScript': '#2b7489',
        'PHP': '#4F5D95',
        'Swift': '#ffac45',
        'Kotlin': '#F18E33',
        'C#': '#178600',
        'HTML': '#e34c26',
        'CSS': '#563d7c',
        'Shell': '#89e051',
        'Jupyter Notebook': '#DA5B0B',
        'R': '#198CE7',
        'Dart': '#00B4AB',
        'Scala': '#c22d40'
    }
    return colors.get(language, '#6b7280')

@app.route('/')
def search_page():
    """Landing page with search form"""
    if df is None:
        error_msg = 'Dataset not loaded. Please ensure Github_data.csv exists in the same directory as this script.'
        return render_template('search.html', error=error_msg)
    return render_template('search.html', dataset_loaded=True, total_projects=len(df))

@app.route('/recommend', methods=['POST'])
def recommend():
    """Process search and show recommendations"""
    if df is None:
        error_msg = 'Dataset not loaded. Please ensure Github_data.csv exists in the same directory as this script.'
        return render_template('search.html', error=error_msg)
    
    query = request.form.get('fname', '').strip()
    
    if not query:
        return render_template('search.html', error='Please enter a search query', dataset_loaded=True, total_projects=len(df))
    
    # Get recommendations
    recommendations = get_recommendations(query, top_n=5)
    
    if not recommendations:
        return render_template('search.html', 
                             error=f'No recommendations found for "{query}". Try different keywords.',
                             dataset_loaded=True, 
                             total_projects=len(df))
    
    # Add additional data for each recommendation
    for rec in recommendations:
        rec['badges'] = determine_badges(rec['stars'], rec['updated_days_ago'])
        rec['time_ago'] = format_time_ago(rec['updated_days_ago'])
        rec['language_color'] = get_language_color(rec['language'])
    
    return render_template('index.html', 
                         query=query,
                         recommendations=recommendations,
                         total_results=len(recommendations))

@app.template_filter('format_number')
def format_number(value):
    """Format numbers with k suffix"""
    try:
        value = int(value)
        if value >= 1000:
            return f"{value / 1000:.1f}k"
        return str(value)
    except:
        return "0"

if __name__ == '__main__':
    if df is None:
        print("\n" + "=" * 60)
        print("ERROR: Cannot start application - dataset not loaded!")
        print("Please ensure 'Github_data.csv' is in the same directory")
        print("=" * 60 + "\n")
    else:
        print("\n" + "=" * 60)
        print(f"✓ Dataset loaded successfully: {len(df)} projects")
        print("✓ Starting Flask application...")
        print("=" * 60 + "\n")
        app.run(debug=True, host='0.0.0.0', port=5000)