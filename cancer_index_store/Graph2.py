import json
from collections import defaultdict, Counter

def analyze_topic_network():
    """Analyze the network of topics without graph libraries"""
    
    JSON_PATH = r"C:\Users\muham\OneDrive\Desktop\Cancer Json\breast_cancer.json"
    
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    topics = [entry.get('topic', 'Unknown') for entry in data]
    topic_counts = Counter(topics)
    
    print("🌐 Topic Network Analysis")
    print("=" * 60)
    
    # Build adjacency list manually
    adjacency = defaultdict(list)
    medical_keywords = ['treatment', 'symptoms', 'diagnosis', 'therapy', 'cancer', 'breast']
    
    topic_list = list(topic_counts.keys())
    
    for i, topic1 in enumerate(topic_list):
        for j, topic2 in enumerate(topic_list):
            if i != j:
                words1 = set(topic1.lower().split())
                words2 = set(topic2.lower().split())
                common_words = words1.intersection(words2).intersection(set(medical_keywords))
                
                if common_words:
                    adjacency[topic1].append((topic2, len(common_words)))
    
    # Calculate network metrics
    print(f"\n📈 Network Statistics:")
    print(f"   Total nodes (topics): {len(adjacency)}")
    
    total_connections = sum(len(connections) for connections in adjacency.values())
    print(f"   Total connections: {total_connections}")
    
    # Find most connected topics (hubs)
    connection_counts = {topic: len(connections) for topic, connections in adjacency.items()}
    hubs = sorted(connection_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    
    print(f"\n🏆 Most Connected Topics (Network Hubs):")
    for topic, connections in hubs:
        print(f"   {topic}: {connections} connections")
    
    # Find strongly connected pairs
    strong_connections = []
    for topic, connections in adjacency.items():
        for other_topic, strength in connections:
            if strength >= 2:  # At least 2 common keywords
                strong_connections.append((topic, other_topic, strength))
    
    print(f"\n💪 Strongly Connected Topic Pairs:")
    for topic1, topic2, strength in sorted(strong_connections, key=lambda x: x[2], reverse=True)[:15]:
        print(f"   {topic1} ↔ {topic2} (strength: {strength})")
    
    # Find isolated topics (no connections)
    isolated_topics = [topic for topic in topic_list if topic not in adjacency or not adjacency[topic]]
    print(f"\n🏝️  Isolated Topics (no connections):")
    for topic in isolated_topics[:10]:
        print(f"   • {topic}")

def export_relationship_data():
    """Export relationship data for external visualization"""
    
    JSON_PATH = r"C:\Users\muham\OneDrive\Desktop\Cancer Json\breast_cancer.json"
    
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    topics = [entry.get('topic', 'Unknown') for entry in data]
    topic_counts = Counter(topics)
    
    # Build relationships
    relationships = []
    medical_keywords = ['treatment', 'symptoms', 'diagnosis', 'therapy', 'cancer', 'breast']
    
    topic_list = list(topic_counts.keys())
    
    for i, topic1 in enumerate(topic_list):
        for j, topic2 in enumerate(topic_list):
            if i < j:
                words1 = set(topic1.lower().split())
                words2 = set(topic2.lower().split())
                common_words = words1.intersection(words2).intersection(set(medical_keywords))
                
                if common_words:
                    relationships.append({
                        'source': topic1,
                        'target': topic2,
                        'weight': len(common_words),
                        'common_words': list(common_words),
                        'source_count': topic_counts[topic1],
                        'target_count': topic_counts[topic2]
                    })
    
    # Export to JSON
    export_data = {
        'nodes': [{'id': topic, 'size': count} for topic, count in topic_counts.items()],
        'links': relationships
    }
    
    with open('topic_relationships.json', 'w', encoding='utf-8') as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Relationship data exported to 'topic_relationships.json'")
    print(f"   Nodes: {len(export_data['nodes'])}")
    print(f"   Links: {len(export_data['links'])}")
    print(f"\n📊 You can import this file into:")
    print(f"   • https://observablehq.com/@d3/force-directed-graph")
    print(f"   • https://gephi.org/")
    print(f"   • Any D3.js force-directed graph visualization")

if __name__ == "__main__":
    analyze_topic_network()
    print("\n" + "="*60 + "\n")
    export_relationship_data()