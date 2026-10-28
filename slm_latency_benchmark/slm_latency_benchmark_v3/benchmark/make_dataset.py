#!/usr/bin/env python3
"""Validate the immutable, English-only benchmark dataset and print its contents."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
qa=json.loads((ROOT/'data/questions.json').read_text(encoding='utf-8'))
gen=json.loads((ROOT/'data/generation_prompts.json').read_text(encoding='utf-8'))
assert len(qa)==60 and len(gen)==12
assert all(x['language']=='en' for x in qa+gen)
assert all(x['options'][ord(x['expected'])-65]==x['answer_text'] for x in qa)
print(f'English-only dataset validated: {len(qa)} multiple-choice questions and {len(gen)} generation prompts.')
