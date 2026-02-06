import pandas as pd

df = pd.read_csv('data_movies_clean.csv')

# Find all Warner movies
warner = df[df['production_company_names'].str.contains('Warner', case=False, na=False)]

print(f"Total Warner movies in dataset: {len(warner)}")
print("\n" + "="*80)

# Check how many match each filter
print("\nFiltering step by step:")
print(f"1. Starting with: {len(warner)} Warner movies")

# Release date
warner_date = warner[(warner['release_date'] >= '2000-01-01') & (warner['release_date'] <= '2020-12-31')]
print(f"2. After release_date filter (2000-2020): {len(warner_date)}")

# Popularity
warner_pop = warner_date[(warner_date['popularity'] >= 3) & (warner_date['popularity'] <= 6)]
print(f"3. After popularity filter (3-6): {len(warner_pop)}")

# Vote average
warner_vote = warner_pop[(warner_pop['vote_average'] >= 3) & (warner_pop['vote_average'] <= 5)]
print(f"4. After vote_average filter (3-5): {len(warner_vote)}")

# Runtime
warner_runtime = warner_vote[(warner_vote['runtime'] >= 30) & (warner_vote['runtime'] <= 60)]
print(f"5. After runtime filter (30-60): {len(warner_runtime)}")

# Origin country
warner_country = warner_runtime[warner_runtime['origin_country'].isin(['US', 'GB'])]
print(f"6. After origin_country filter (US/GB): {len(warner_country)}")

# Language
warner_lang = warner_country[warner_country['original_language'] == 'en']
print(f"7. After original_language filter (en): {len(warner_lang)}")

print("\n" + "="*80)
if len(warner_lang) > 0:
    print("\nSample movies that match ALL filters:")
    print(warner_lang[['title', 'production_company_names', 'release_date', 'runtime', 'popularity', 'vote_average']].head(10))
else:
    print("\n❌ NO Warner movies match all those filters!")
    print("\nLet's check some Warner movies without filters:")
    print(warner[['title', 'production_company_names', 'release_date', 'runtime', 'popularity', 'vote_average']].head(10))