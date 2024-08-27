import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from datetime import datetime, timedelta
import anthropic
from dateutil import parser

# Initialize Anthropic client
claude_client = anthropic.Anthropic(api_key="your_api_key_here")

def load_data():
    rfc_data = pd.read_csv('rfc_data.csv')
    ci_data = pd.read_csv('ci_data.csv')
    events_data = pd.read_csv('events_data.csv')
    critical_systems_data = pd.read_csv('critical_systems.csv')  # New dataset for critical systems
    return rfc_data, ci_data, events_data, critical_systems_data

def preprocess_text(text):
    if pd.isna(text):
        return ""
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())
    return text

def claude_analyze(text, task):
    prompt = f"Task: {task}\n\nText: {text}\n\nAnalysis:"
    response = claude_client.completions.create(
        model="claude-3-opus-20240229",
        prompt=prompt,
        max_tokens_to_sample=300
    )
    return response.completion

def extract_key_terms(text):
    key_terms = claude_analyze(
        text, 
        "Extract key terms such as service names, applications, and problem descriptions. Return as a comma-separated list."
    )
    return [term.strip().lower() for term in key_terms.split(',')]

def calculate_similarity(text1, text2):
    vectorizer = TfidfVectorizer(preprocessor=preprocess_text)
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

def keyword_search(text, dataset, column):
    key_terms = extract_key_terms(text)
    return dataset[dataset[column].apply(lambda x: any(term in preprocess_text(str(x)) for term in key_terms))]

def semantic_search(text, dataset, column, threshold=0.3):
    return dataset[dataset[column].apply(lambda x: calculate_similarity(text, str(x)) > threshold)]

def check_correlation(new_item, dataset, column, threshold=0.3):
    correlations = []
    for _, item in dataset.iterrows():
        similarity = calculate_similarity(new_item, str(item[column]))
        if similarity > threshold:
            correlations.append({
                'type': type(item).name,
                'id': item.get('id') or item.index,
                'description': item[column],
                'similarity': similarity
            })
    return correlations

def check_recent_events(new_item, events_data):
    recent_events = []
    current_time = datetime.now()
    for _, event in events_data.iterrows():
        try:
            event_time = datetime.strptime(event['created_ts'], '%Y-%m-%d %H:%M:%S.%f')
        except ValueError:
            try:
                event_time = datetime.strptime(event['created_ts'], '%Y-%m-%d %H:%M:%S')
            except ValueError:
                event_time = parser.parse(event['created_ts'])
        if current_time - event_time <= timedelta(days=7):
            similarity = calculate_similarity(new_item, event['description'])
            if similarity > 0.3:
                recent_events.append({
                    'type': 'Event',
                    'id': event['event_id'],
                    'description': event['description'],
                    'similarity': similarity,
                    'status': event['event_status'],
                    'severity': event['severity']
                })
    return recent_events

def extract_metadata(new_item):
    metadata = claude_analyze(
        new_item, 
        "Extract the responsible_service_area and site_location if present. Return as JSON: {\"responsible_service_area\": \"...\", \"site_location\": \"...\"}"
    )
    try:
        return eval(metadata)
    except:
        return {"responsible_service_area": None, "site_location": None}

def check_critical_system_correlation(new_item, critical_systems_data):
    correlations = []
    for _, system in critical_systems_data.iterrows():
        similarity = calculate_similarity(new_item, system['description'])
        if similarity > 0.3:
            correlations.append({
                'type': 'Critical System',
                'id': system['system_id'],
                'name': system['system_name'],
                'description': system['description'],
                'similarity': similarity,
                'impact_level': system['impact_level']
            })
    return correlations

def triage_new_item(new_item, rfc_data, ci_data, events_data, critical_systems_data, alert_history):
    print(f"Triaging new item: {new_item}")
    
    # Extract metadata
    metadata = extract_metadata(new_item)
    responsible_service_area = metadata['responsible_service_area']
    site_location = metadata['site_location']
    
    # Check for recent similar alerts if metadata is available
    if responsible_service_area and site_location:
        for recent_alert in alert_history:
            if (recent_alert['responsible_service_area'] == responsible_service_area and
                recent_alert['site_location'] == site_location):
                similarity = calculate_similarity(new_item, recent_alert['description'])
                if similarity > 0.8:  # High similarity threshold
                    print("Similar alert recently processed. Skipping triage.")
                    return

    # Perform keyword and semantic search across all datasets
    keyword_results = {
        'RFC': keyword_search(new_item, rfc_data, 'BRIEF_DESCRIPTION'),
        'CI': keyword_search(new_item, ci_data, 'DESCRIPTION'),
        'Events': keyword_search(new_item, events_data, 'description'),
        'Critical Systems': keyword_search(new_item, critical_systems_data, 'description')
    }
    
    semantic_results = {
        'RFC': semantic_search(new_item, rfc_data, 'BRIEF_DESCRIPTION'),
        'CI': semantic_search(new_item, ci_data, 'DESCRIPTION'),
        'Events': semantic_search(new_item, events_data, 'description'),
        'Critical Systems': semantic_search(new_item, critical_systems_data, 'description')
    }

    # Check correlations
    rfc_correlations = check_correlation(new_item, rfc_data, 'BRIEF_DESCRIPTION')
    ci_correlations = check_correlation(new_item, ci_data, 'DESCRIPTION')
    recent_events = check_recent_events(new_item, events_data)
    critical_system_correlations = check_critical_system_correlation(new_item, critical_systems_data)

    # Print results
    print("\nKeyword Search Results:")
    for data_type, results in keyword_results.items():
        print(f"{data_type}: {len(results)} matches")
    
    print("\nSemantic Search Results:")
    for data_type, results in semantic_results.items():
        print(f"{data_type}: {len(results)} matches")
    
    print("\nCorrelations:")
    print(f"RFC Correlations: {len(rfc_correlations)}")
    print(f"CI Correlations: {len(ci_correlations)}")
    print(f"Recent Events: {len(recent_events)}")
    print(f"Critical System Correlations: {len(critical_system_correlations)}")

    # Perform additional analysis using Claude
    impact_analysis = claude_analyze(
        new_item,
        "Analyze the potential business impact of this issue and suggest initial steps for resolution."
    )
    print(f"\nImpact Analysis and Initial Steps:\n{impact_analysis}")

    # Add this alert to the history
    alert_history.append({
        'description': new_item,
        'responsible_service_area': responsible_service_area,
        'site_location': site_location,
        'timestamp': datetime.now()
    })

def main():
    rfc_data, ci_data, events_data, critical_systems_data = load_data()
    alert_history = []
    
    while True:
        new_item = input("\nEnter the description of the new item (or 'quit' to exit): ")
        if new_item.lower() == 'quit':
            break
        triage_new_item(new_item, rfc_data, ci_data, events_data, critical_systems_data, alert_history)

if __name__ == "__main__":
    main()
