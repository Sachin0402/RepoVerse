# RepoVerse
RepoVerse is an AI-powered web application that recommends GitHub repositories based on user queries.
Built using Flask, Pandas, and Scikit-learn, it analyzes repository metadata such as names, descriptions, topics, and programming languages to generate highly relevant repository suggestions.

The system combines TF-IDF vectorization and cosine similarity techniques to understand the semantic relationship between user input and repository content. This allows developers to quickly discover similar or trending projects that match their areas of interest.

# GitHub Project Recommender - Setup Guide

# GitHub Project Recommender - Setup Guide

## 📁 Project Structure

```
github-recommender/
│
├── flask_recommendation.py      # Flask backend application
├── requirements.txt             # Python dependencies
├── README.md                    # This file
│
└── templates/                   # HTML templates
    ├── search.html             # Landing/search page
    └── index.html              # Results page
```

## 🔧 Installation Steps

### Step 1: Install Python Dependencies

Create a `requirements.txt` file with the following content:

```
Flask==3.0.0
pandas==2.1.3
numpy==1.26.2
scikit-learn==1.3.2
```

Then install:

```bash
pip install -r requirements.txt
```

### Step 2: Create Directory Structure

```bash
# Create the templates folder
mkdir templates

# Move HTML files to templates folder
# - Move search.html to templates/search.html
# - Move index.html to templates/index.html
```

### Step 3: File Placement

1. **flask_recommendation.py** → Root directory
2. **search.html** → `templates/search.html`
3. **index.html** → `templates/index.html`

## 🚀 Running the Application

### Method 1: Using Python directly

```bash
python flask_recommendation.py
```

### Method 2: Using Flask CLI

```bash
export FLASK_APP=flask_recommendation.py
export FLASK_ENV=development
flask run
```

The application will start on: **http://127.0.0.1:5000**

## 🌐 Accessing the Application

1. **Landing Page:** http://127.0.0.1:5000/
2. **Search:** Enter your query and click "Find Recommendations"
3. **Results:** View AI-powered recommendations

## 📊 How It Works

### Search Flow:

1. User visits landing page (`/`)
2. User enters search query (e.g., "recommendation system")
3. Form submits to `/recommend` endpoint
4. Backend processes query using TF-IDF and cosine similarity
5. Top 5 most relevant projects are returned
6. Results page displays with match percentages and details

### Features:

✅ **Content-Based Filtering** - Uses TF-IDF vectorization  
✅ **Cosine Similarity** - Calculates relevance scores  
✅ **Dynamic Match Percentages** - Shows 0-100% match  
✅ **Badge System** - Popular (1000+ stars) & Active (updated within 7 days)  
✅ **Beautiful UI** - Animated gradient backgrounds with particles  
✅ **Responsive Design** - Works on all devices  
✅ **Share Functionality** - Share projects via native share or clipboard  

## 🎨 Customization

### Adding Your Own Dataset:

Edit the `load_data()` function in `flask_recommendation.py`:

```python
def load_data():
    # Option 1: Load from CSV
    df = pd.read_csv('your_projects.csv')
    
    # Option 2: Load from database
    # df = pd.read_sql('SELECT * FROM projects', conn)
    
    # Option 3: Use the sample data (current implementation)
    return df
```

### Required DataFrame Columns:

- `name` - Project name
- `owner` - Project owner username
- `description` - Project description
- `topics` - Comma-separated topics
- `stars` - Number of stars
- `forks` - Number of forks
- `language` - Programming language
- `updated_days_ago` - Days since last update
- `url` - GitHub project URL

### Changing Number of Results:

In `flask_recommendation.py`, modify:

```python
recommendations = get_recommendations(query, top_n=10)  # Change 5 to 10
```

### Customizing Colors:

Edit CSS variables in both HTML files:

```css
:root {
    --primary: #2563eb;        /* Main blue color */
    --purple: #8b5cf6;         /* Purple accent */
    --pink: #ec4899;           /* Pink accent */
    --success: #10b981;        /* Green for success */
}
```

## 🐛 Troubleshooting

### Issue: "Template not found"
**Solution:** Ensure HTML files are in the `templates/` folder

### Issue: "Module not found"
**Solution:** Install all requirements: `pip install -r requirements.txt`

### Issue: "Port already in use"
**Solution:** Change port in flask_recommendation.py:
```python
app.run(debug=True, host='0.0.0.0', port=5001)  # Use different port
```

### Issue: No results found
**Solution:** The query might be too specific. Try broader terms like:
- "recommendation" instead of "advanced recommendation system"
- "machine learning" instead of "deep neural networks"

## 📝 Development Tips

### Enable Debug Mode:

```python
app.run(debug=True)  # Auto-reloads on code changes
```

### View Logs:

The console will show:
- Incoming requests
- Search queries
- Number of recommendations found
- Any errors

### Testing Different Queries:

Try these example queries:
- "recommendation system"
- "machine learning"
- "web scraper"
- "chatbot"
- "data science"
- "neural network"

## 🔐 Production Deployment

### For Production Use:

1. **Disable Debug Mode:**
```python
app.run(debug=False)
```

2. **Use Production Server:**
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 flask_recommendation:app
```

3. **Add Environment Variables:**
```bash
export FLASK_ENV=production
export SECRET_KEY=your-secret-key-here
```

## 📚 Additional Features to Add

### Future Enhancements:

- [ ] Add search filters (by language or stars)
- [ ] Display live GitHub repo stats
- [ ] Search history
- [ ] Improve UI and responsiveness


## 🤝 Contributing

Feel free to customize and extend this project! Some ideas:

1. Add more recommendation algorithms
2. Integrate with GitHub API for real-time data
3. Add user preferences and personalization
4. Implement collaborative filtering
5. Add project comparison features

## 📄 License

This project is open-source and available for educational purposes.

## 🆘 Support

If you encounter any issues:

1. Check the troubleshooting section
2. Verify all files are in correct locations
3. Ensure all dependencies are installed
4. Check Python version (3.8+ recommended)

---

**Enjoy discovering amazing GitHub projects! 🚀**
