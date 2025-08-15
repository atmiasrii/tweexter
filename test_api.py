#!/usr/bin/env python3
"""Quick API test script."""

from api import app
from fastapi.testclient import TestClient

def test_api():
    client = TestClient(app)
    
    # Test health endpoint
    health = client.get('/healthz')
    print(f"Health check: {health.status_code} - {health.json()}")
    
    # Test root endpoint
    root = client.get('/')
    print(f"Root endpoint: {root.status_code} - {root.json()}")
    
    # Test prediction endpoint
    test_data = {
        'text': 'This is a great day for coding!',
        'followers': 1000,
        'return_details': False
    }
    
    pred = client.post('/predict', json=test_data)
    print(f"Prediction test: {pred.status_code}")
    
    if pred.status_code == 200:
        result = pred.json()
        print(f"  Likes: {result['likes']}")
        print(f"  Retweets: {result['retweets']}")
        print(f"  Replies: {result['replies']}")
        print(f"  Ranges available: {'ranges' in result}")
        if 'ranges' in result:
            print(f"  Likes range: {result['ranges']['likes']}")
    else:
        print(f"  Error: {pred.text}")

if __name__ == "__main__":
    test_api()
