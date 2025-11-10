import json
from pathlib import Path

def analyze_vector_store_simple():
    """Analyze the vector store without numpy dependency"""
    
    vector_file = Path("cancer_index_store/default__vector_store.json")
    
    if not vector_file.exists():
        print("❌ Vector store file not found!")
        return
    
    # Load the vector store data
    with open(vector_file, 'r', encoding='utf-8') as f:
        vector_data = json.load(f)
    
    print("🔍 Vector Store Analysis")
    print("=" * 50)
    
    # Check the structure
    if 'embedding_dict' in vector_data:
        embeddings = vector_data['embedding_dict']
        print(f"📊 Total embeddings: {len(embeddings)}")
        
        # Show sample embeddings
        print(f"\n📋 Sample Document IDs:")
        doc_ids = list(embeddings.keys())[:5]
        for i, doc_id in enumerate(doc_ids):
            vector = embeddings[doc_id]
            print(f"  {i+1}. {doc_id}")
            print(f"     Vector dimensions: {len(vector)}")
            print(f"     First 5 values: {vector[:5]}")
        
        # Basic statistics without numpy
        if embeddings:
            first_vector = list(embeddings.values())[0]
            all_values = [val for vec in embeddings.values() for val in vec]
            
            print(f"\n📈 Basic Statistics:")
            print(f"   Vector dimensions: {len(first_vector)}")
            print(f"   Min value: {min(all_values):.6f}")
            print(f"   Max value: {max(all_values):.6f}")
            print(f"   Avg value: {sum(all_values)/len(all_values):.6f}")
    
    if 'text_id_to_ref_doc_id' in vector_data:
        mapping = vector_data['text_id_to_ref_doc_id']
        print(f"\n🔗 Text to Document Mapping: {len(mapping)} entries")
        print("   Sample mappings:")
        for i, (text_id, doc_id) in enumerate(list(mapping.items())[:3]):
            print(f"     {text_id} → {doc_id}")
    
    # Show what semantic search does
    print(f"\n🎯 How Semantic Search Works:")
    print(f"   1. Your question gets converted to a {len(first_vector) if 'embedding_dict' in vector_data else '384'}-dimensional vector")
    print(f"   2. System finds the most similar vectors in this file")
    print(f"   3. Returns documents with highest similarity scores")
    print(f"   4. Similar medical content = Closer vectors in this space")

def show_search_example():
    """Show a concrete example of how search works"""
    
    print(f"\n🔍 Concrete Search Example:")
    print(f"   Query: 'What are breast cancer symptoms?'")
    print(f"   → Converted to vector: [0.123, -0.456, 0.789, ...] (384 numbers)")
    print(f"   → Compared against 320 document vectors in default__vector_store.json")
    print(f"   → Finds vectors for documents about:")
    print(f"      • 'Common symptoms of breast cancer'")
    print(f"      • 'Early warning signs and detection'")
    print(f"      • 'Patient symptom reporting guidelines'")

if __name__ == "__main__":
    analyze_vector_store_simple()
    show_search_example()