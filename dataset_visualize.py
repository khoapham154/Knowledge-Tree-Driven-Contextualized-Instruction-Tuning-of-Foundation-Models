import json
import spacy
from collections import Counter
import pandas as pd
import plotly.express as px

# Load the JSON file
json_path = 'your_json_file_path.json'
with open(json_path, 'r') as f:
    data = json.load(f)

# Extract all "text" fields (diagnostic questions)
texts = [item['text'] for item in data]

# Load SpaCy model for English
nlp = spacy.load('en_core_web_sm')

# Process all texts with SpaCy
docs = [nlp(text) for text in texts]

# --- Lexical Composition Table ---

# Define mapping from SpaCy POS tags to lexical categories
category_map = {
    'NOUN': 'noun',
    'PROPN': 'noun',    # Proper nouns treated as nouns
    'ADJ': 'adjective',
    'ADV': 'adverb',
    'VERB': 'verb',
    'AUX': 'verb',      # Auxiliary verbs treated as verbs
    'NUM': 'numeral',
    'ADP': 'preposition'
}

# Count occurrences of each lexical category
category_counts = Counter()
total_words = 0

for doc in docs:
    for token in doc:
        # Exclude punctuation from word count
        if not token.is_punct:
            total_words += 1
            pos = token.pos_
            if pos in category_map:
                category_counts[category_map[pos]] += 1

# Calculate percentages for each category
percentages = {cat: (count / total_words) * 100 for cat, count in category_counts.items()}

# Create a DataFrame for the table
df_table = pd.DataFrame({
    'Lexical': ['Instruction Dataset'],
    'n. (noun)': [round(percentages.get('noun', 0), 1)],
    'adj. (adjective)': [round(percentages.get('adjective', 0), 1)],
    'adv. (adverb)': [round(percentages.get('adverb', 0), 1)],
    'v. (verb)': [round(percentages.get('verb', 0), 1)],
    'num. (numeral)': [round(percentages.get('numeral', 0), 1)],
    'prep. (preposition)': [round(percentages.get('preposition', 0), 1)]
})

# Display the table
print("Lexical Composition Table:")
print(df_table.to_string(index=False))

# --- Sunburst Diagram for Verb-Noun Pairs with Threshold ---

# Extract verb-noun pairs where noun is the direct object (dobj) of the verb
pair_counts = Counter()

for doc in docs:
    for token in doc:
        # Check if token is a direct object and its head is a verb
        if token.dep_ == 'dobj' and token.head.pos_ == 'VERB':
            verb = token.head.lemma_  # Use lemma to normalize verb form
            noun = token.lemma_       # Use lemma to normalize noun form
            pair_counts[(verb, noun)] += 1

# Calculate total frequency of all pairs
total_pairs = sum(pair_counts.values())

# Set threshold (e.g., pairs that make up more than 1% of the total)
threshold_percentage = 0.5  # Adjust this value as needed (e.g., 5.0 for 5%)

# Calculate the minimum frequency required to meet the threshold
min_frequency = (threshold_percentage / 100) * total_pairs

# Filter pairs that meet or exceed the minimum frequency
filtered_pairs = {pair: freq for pair, freq in pair_counts.items() if freq >= min_frequency}

# Convert filtered pair counts to a DataFrame for visualization
pairs_df = pd.DataFrame(
    [(verb, noun, freq) for (verb, noun), freq in filtered_pairs.items()],
    columns=['verb', 'noun', 'frequency']
)

# Create the sunburst diagram with filtered data
fig = px.sunburst(
    pairs_df,
    path=['verb', 'noun'],    # Hierarchy: verb -> noun
    values='frequency',       # Size of wedges based on pair frequency
    title=f'Distribution of Root Noun-Verb Pairs (>{threshold_percentage}%) in Instruction Dataset'
)

# Display the sunburst diagram
fig.show()