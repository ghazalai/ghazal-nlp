import io
import json
import struct

import pytest

from ghazal_nlp import vowels as v
from ghazal_nlp._vowel_tables import PackedMap, load_context, load_letters


def test_packed_map_grows_preserves_order_and_replaces_values():
    table = PackedMap()
    expected = {f'گل{i}': str(i).encode() for i in range(100)}
    for key, value in expected.items():
        table.add(key, value)
    table.add('گل0', b'new')
    expected['گل0'] = b'new'
    assert table == expected
    assert list(table) == list(expected)
    assert table.get('absent') is None and None not in table
    with pytest.raises(TypeError):
        table[[]]


def test_streaming_json_across_every_character_boundary():
    class SmallReads(io.StringIO):
        def read(self, n=-1):
            return super().read(min(n, 1))

    weights = {f'گل{i}': [-0.0, 1.2345678901234567, 1e-100] for i in range(20)}
    expected = {'words': {'گل': {'forms': ['گُل', 'گِل'], 'weights': weights}},
                'temperature': 0.7, 'escaped': '"\\\nگل', 'empty': {}}
    loaded = load_context(SmallReads(json.dumps(expected, ensure_ascii=False)))
    assert loaded == expected
    assert struct.pack('<3d', *loaded['words']['گل']['weights']['گل0']) == struct.pack('<3d', *weights['گل0'])
    letters = {'contexts': {'^گل': {'ُ': 12, 'ِ': 3}}, 'support': 5}
    assert load_letters(SmallReads(json.dumps(letters))) == letters


def test_streaming_json_rejects_truncated_records():
    with pytest.raises((ValueError, json.JSONDecodeError)):
        load_context(io.StringIO('{"words": {"گل": {'))


def test_lazy_lexicon_keeps_nonadjacent_readings_and_last_totals(tmp_path):
    path = tmp_path / 'lexicon.tsv'
    path.write_text('# sample\nگل\tگُل\t2\t1\t2\t3\t0\t{"a":2}\n'
                    'دل\tدِل\t1\t1\t1\t1\t0\n'
                    'گل\tگِل\t1\t1\t3\t5\t1\t{"b":1}\n', encoding='utf-8')
    lex = v.Lexicon.load(path)
    assert list(lex) == ['گل', 'دل']
    assert lex['گل'] == v.Entry('گل', (v.Reading('گُل', 2, 1, {'a': 2}), v.Reading('گِل', 1, 1, {'b': 1})), 3, 5, 1)
    assert lex.lookup('گُل') is lex.lookup('گِل')
    assert lex.get(None) is None
    entry = lex['دل']
    lex['new'] = entry
    del lex['دل']
    assert lex['new'] is entry and 'دل' not in lex
    assert lex.copy() == dict(lex)


def test_lookup_fast_path_keeps_encoding_normalization():
    entries = {key: v.Entry(key, (), 0, 0, 0) for key in ['گل', 'کتاب', 'می\u200cرود', 'أمر', 'آرد', 'خانه']}
    lex = v.Lexicon(entries)
    for word in ['گل', 'گُل', 'كِتاب', 'کتاب', 'می\u200cرود', 'می\u200dرود', 'أمر', 'أمر', 'آرد', 'خانۀ', '\u200cگل']:
        key = v.skeleton(v.split_terminal(v.canonicalize(v.fold(word)))[0])
        assert lex.lookup(word) == entries.get(key)
