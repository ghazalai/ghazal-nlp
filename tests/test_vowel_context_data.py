import gzip
import json

from ghazal_nlp import vowels as v, vowel_context as c


def test_compressed_model_and_licence_provenance():
    path=v.DATA/'vowels-context.json.gz'
    packed=path.read_bytes()
    assert packed[:2]==b'\x1f\x8b' and packed[4:8]==b'\0'*4
    data=json.loads(gzip.decompress(packed))
    assert data==c.model()
    assert not (v.DATA/'vowels-context.json').exists()
    assert data['provenance']['licence'].startswith('CC BY-SA 4.0')
    assert {'homorich','ganjoor-poems','fawiki','fawikisource'}<=set(data['provenance']['sources'])
    assert 'books' in data['provenance']['sources']
    assert data['provenance']['dictionary_sources']['sources']
    assert 'vowels-context.json.gz is also CC BY-SA 4.0' in (v.DATA/'LICENSE').read_text()


def test_common_reading_and_no_added_mark_evidence_survive_training():
    record=c.model()['words']['این']
    assert 'اِین' in record['forms']
    choice=v.choose('این کتاب خوب است.')[0]
    assert choice.form=='این' and 0<choice.confidence<1
    assert 'آب' in c.model()['words']['آب']['forms']
    assert v.mark('آب',mode='full',ezafe=False)=='آب'

