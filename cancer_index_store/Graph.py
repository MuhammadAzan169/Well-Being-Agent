import json
from collections import Counter, defaultdict

def create_text_based_relationship_graph():
    """Create a text-based relationship graph between topics"""
    
    JSON_PATH = r"C:\Users\muham\OneDrive\Desktop\Cancer Json\breast_cancer.json"
    
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Extract all topics
    topics = [entry.get('topic', 'Unknown') for entry in data]
    topic_counts = Counter(topics)
    
    print(f"📋 Found {len(topic_counts)} unique topics")
    print("=" * 60)
    
    # Show topic frequencies
    print(f"\n📊 Topic Frequency Distribution:")
    for topic, count in topic_counts.most_common():
        percentage = (count / len(data)) * 100
        print(f"  {topic}: {count} entries ({percentage:.1f}%)")
    
    # Create relationships based on shared medical keywords
    medical_keywords = ['treatment', 'symptoms', 'diagnosis', 'therapy', 'cancer', 'breast', 
                       'pain', 'surgery', 'radiation', 'chemotherapy', 'health', 'care']
    
    print(f"\n🔗 Topic Relationships Based on Shared Keywords:")
    print("=" * 60)
    
    relationships = []
    
    # Find relationships between topics
    topic_list = list(topic_counts.keys())
    for i, topic1 in enumerate(topic_list):
        for j, topic2 in enumerate(topic_list):
            if i < j:  # Avoid duplicate pairs
                words1 = set(topic1.lower().split())
                words2 = set(topic2.lower().split())
                common_words = words1.intersection(words2).intersection(set(medical_keywords))
                
                if common_words:
                    relationships.append((topic1, topic2, len(common_words), common_words))
    
    # Sort by relationship strength
    relationships.sort(key=lambda x: x[2], reverse=True)
    
    # Show top relationships
    print(f"\n🏆 Top 20 Strongest Topic Relationships:")
    for i, (topic1, topic2, strength, common_words) in enumerate(relationships[:20]):
        print(f"\n{i+1}. {topic1} ↔ {topic2}")
        print(f"   Strength: {strength} | Common words: {', '.join(common_words)}")
    
    # Create topic clusters
    print(f"\n🎯 Topic Clusters (Groups of Related Topics):")
    print("=" * 60)
    
    # Group topics by primary medical categories
    medical_categories = {
        'Treatment & Therapy': ['treatment', 'therapy', 'chemotherapy', 'radiation', 'surgery'],
        'Symptoms & Diagnosis': ['symptoms', 'diagnosis', 'pain', 'detection'],
        'Reproduction & Fertility': ['pregnancy', 'fertility', 'birth', 'reproduction'],
        'Mental Health & Support': ['support', 'mental', 'emotional', 'health', 'care'],
        'General Cancer': ['cancer', 'breast', 'disease', 'medical']
    }
    
    topic_clusters = defaultdict(list)
    
    for topic in topic_list:
        topic_lower = topic.lower()
        assigned = False
        
        for category, keywords in medical_categories.items():
            if any(keyword in topic_lower for keyword in keywords):
                topic_clusters[category].append((topic, topic_counts[topic]))
                assigned = True
                break
        
        if not assigned:
            topic_clusters['Other Topics'].append((topic, topic_counts[topic]))
    
    # Display clusters
    for category, topics_in_cluster in topic_clusters.items():
        if topics_in_cluster:
            total_entries = sum(count for _, count in topics_in_cluster)
            print(f"\n📁 {category} ({total_entries} total entries):")
            for topic, count in sorted(topics_in_cluster, key=lambda x: x[1], reverse=True):
                print(f"   • {topic}: {count} entries")

def create_ascii_relationship_map():
    """Create a simple ASCII art relationship map"""
    
    JSON_PATH = r"C:\Users\muham\OneDrive\Desktop\Cancer Json\breast_cancer.json"
    
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    topics = [entry.get('topic', 'Unknown') for entry in data]
    topic_counts = Counter(topics)
    
    print(f"\n🎨 ASCII Relationship Map")
    print("=" * 60)
    
    # Get top 15 topics by frequency
    top_topics = [topic for topic, _ in topic_counts.most_common(15)]
    
    print(f"\nTop 15 Topics (by frequency):")
    print("─" * 40)
    
    for i, topic in enumerate(top_topics, 1):
        count = topic_counts[topic]
        bar = "█" * (count // 2)  # Simple visual representation
        print(f"{i:2d}. {topic:<40} {bar} ({count})")
    
    # Show connections between top topics
    print(f"\nKey Relationships Between Top Topics:")
    print("─" * 50)
    
    medical_keywords = ['treatment', 'symptoms', 'diagnosis', 'therapy', 'cancer', 'breast']
    
    for i in range(len(top_topics)):
        for j in range(i + 1, len(top_topics)):
            topic1, topic2 = top_topics[i], top_topics[j]
            words1 = set(topic1.lower().split())
            words2 = set(topic2.lower().split())
            common_words = words1.intersection(words2).intersection(set(medical_keywords))
            
            if common_words:
                strength = len(common_words)
                connection_char = "─" * strength + "┼" if strength > 1 else "─┼"
                print(f"{topic1:<25} {connection_char} {topic2}")

if __name__ == "__main__":
    create_text_based_relationship_graph()
    create_ascii_relationship_map()