"""Smoke test: python test_app.py"""
from app import app

c = app.test_client()

assert c.get('/').status_code == 200

spam = c.post('/predict', json={'message': 'WINNER! You have won a FREE prize. Call 09061701461 to claim now'}).get_json()
ham = c.post('/predict', json={'message': 'Hey, are we still meeting for lunch tomorrow?'}).get_json()
assert spam['is_spam'] and spam['label'] == 'SPAM' and 50 <= spam['confidence'] <= 100, spam
assert not ham['is_spam'] and ham['label'] == 'HAM', ham

# Test input validation
for bad in [dict(json={'message': '   '}), dict(json={'message': None}), dict(json={'message': 123}),
            dict(json=['hi']), dict(data='message=hi'), dict()]:
    r = c.post('/predict', **bad)
    assert r.status_code == 400 and r.get_json() == {'error': 'No message provided'}, (bad, r.status_code)

assert c.post('/predict', json={'message': 'x' * 100_000}).status_code == 413

print('All checks passed.')
