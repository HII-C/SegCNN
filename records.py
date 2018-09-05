'''
For organizing the i2b2 data.
Uses static type checking (PEP526)
'''
import os
import re
from enum import Enum
from typing import List, Tuple, T
from dataclasses import dataclass
import json
# from SegDataObject import SegData
# from keras.preprocessing.text import text_to_word_sequence


class RelationLabel(Enum):
    TRCP = "TrCP"
    TRAP = "TrAP"
    TRWP = "TrWP"
    TRIP = "TrIP"
    TRNAP = "TrNAP"
    TERP = "TeRP"
    TECP = "TeCP"
    PIP = "PIP"

    @property
    def encoded(self):
        encoder = {"TrCP": 0, "TrAP": 1, "TrWP": 2, "TrIP": 3,
                   "TrNAP": 4, "TeRP": 5, "TeCP": 6, "PIP": 7}
        return encoder[self.value]


@dataclass
class Relation:
    sentence_id: int
    c1_start: int
    c1_end: int
    c2_start: int
    c2_end: int
    relation: RelationLabel

    def __init__(self, line):
        note = re.search(
            r'c=".*?" (\d+):(\d+) \d+:(\d+)\|\|r="(.*?)"\|\|c=".*?" (\d+):(\d+) \d+:(\d+)', line)
        self.sentence_id = int(note.group(1))-1
        self.c1_start = int(note.group(2))
        self.c1_end = int(note.group(3))+1
        self.c2_start = int(note.group(6))
        self.c2_end = int(note.group(7))+1
        self.relation = RelationLabel(note.group(4))

        if self.c1_start > self.c2_start:
            t1_s = self.c1_start
            t1_e = self.c1_end
            self.c1_start = self.c2_start
            self.c1_end = self.c2_end
            self.c2_start = t1_s
            self.c2_end = t1_e


class Sentence:
    sentence_id: int = 0
    text: List[str] = list()

    def __init__(self, index, data):
        self.sentence_id = index
        # Test if there is a diff between top & bottom for data acc
        # If no difference, filtering useless info should help (right?)
        self.text = data.lower().split()
        # self.text = text_to_word_sequence(data)


class Relations:
    parsed_rels: List[Relation] = list()
    sentences_w_rel: List[int] = list()

    def __init__(self, path):
        with open(path, 'r') as file_:
            lines = file_.readlines()
            for line in lines:
                self.parsed_rels.append(Relation(line))
                self.sentences_w_rel.append(self.parsed_rels[-1].sentence_id)


class Note:
    sentences: List[str] = list()  # Access by index

    def __init__(self, path):
        with open(path, 'r') as file_:
            lines = file_.readlines()
            for idx, line in enumerate(lines):
                self.sentences.append(Sentence(idx, line))


class LabeledSegment:
    prec: List[str] = list()
    c1: List[str] = list()
    mid: List[str] = list()
    c2: List[str] = list()
    succ: List[str] = list()
    label = RelationLabel
    encoded_label = int

    def __init__(self, sentence: Sentence, rel: Relation):
        self.prec = sentence.text[0: rel.c1_start]
        self.c1 = sentence.text[rel.c1_start: rel.c1_end]
        self.mid = sentence.text[rel.c1_end: rel.c2_start]
        self.c2 = sentence.text[rel.c2_start: rel.c2_end]
        self.succ = sentence.text[rel.c2_end: -1]
        self.label = rel.relation.value
        self.encoded_label = rel.relation.encoded


@dataclass
class RecordMeta:
    id_: str = None
    note_path: str = None
    relation_path: str = None

    def __init__(self, note_path, relation_path):
        tail = (os.path.split(note_path)[-1])
        self.id_ = tail.split('.')[0]
        self.note_path = note_path
        self.relation_path = relation_path


class Record:
    id_: str = None
    note: Note = None
    relations: Relations = None
    segmented_record: List[LabeledSegment] = list()

    def __init__(self, record_meta: RecordMeta):
        self.id_ = record_meta.id_
        self.note = Note(record_meta.note_path)
        self.relations = Relations(record_meta.relation_path)

    def to_segment_fmt(self, filter_rels=False, rels_to_find: List[LabeledSegment] = None):
        '''
        Form segmented record, with labels, from the text.
        Does not encode the text, nor form it into a matrix.
        Note: rels_to_find must be a list of RelationLabel
        '''
        rels = self.relations.parsed_rels
        if filter_rels:
            if rels_to_find is None:
                raise AttributeError("rels_to_find must not be None")
            rels[:] = [x for x in rels if x.relation in rels_to_find]

        for rela in rels:
            self.segmented_record.append(LabeledSegment(
                self.note.sentences[rela.sentence_id], rela))

    def to_str(self) -> Tuple[List[List[List[str]]], List[int], List[RelationLabel]]:
        outer_list: List[List[str]] = list()
        encoded_labels: List[int] = list()
        rel_labels: List[RelationLabel] = list()
        for entry in self.segmented_record:
            inner = [entry.prec, entry.c1, entry.mid, entry.c2, entry.succ]
            outer_list.append(inner)
            encoded_labels.append(entry.encoded_label)
            rel_labels.append(entry.label)
        return tuple([outer_list, encoded_labels, rel_labels])


class RecordsUtil:
    ''' Loading, accessing, and necessary utilities for Notes '''
    # Dict, str keys for id_, values of class -> Note
    records: List[Record] = list()
    labeled_segs: List[List[LabeledSegment]] = list()

    def load_records(self, list_rec_meta: List[RecordMeta]):
        for rec_meta in list_rec_meta:
            self.records.append(Record(rec_meta))
            self.records[-1].to_segment_fmt()
            self.labeled_segs.append(self.records[-1].segmented_record)

    def to_matrix(self, seg_rec):
        print()

    def to_str(self, encoded=False) -> \
            Tuple[List[List[List[List[str]]]], (List[RelationLabel] | List[int])]:
        '''
        End result, semantically, for the first entry is:
            List of Records -> list of sentences -> list of segments -> list of strings

        '''
        rec_list_ = list()
        label_list_ = list()
        for record_ in self.records:
            str_tuple = record_.to_str()
            rec_list_.append(str_tuple[0])
            if encoded:
                label_list_.append(str_tuple[1])
            else:
                label_list_.append(str_tuple[2])
        return tuple([rec_list_, label_list_])


if __name__ == '__main__':
    base_str = 'data/i2b2/concept_assertion_relation_training_data/beth'
    raw_list = [f.split('.')[0] for f in os.listdir(f'{base_str}/txt')]
    raw_list[:] = [_ for _ in raw_list if _ is not '']
    print(raw_list)

    txt_list = [f'{base_str}/txt/{file_}.txt' for file_ in raw_list]
    rel_list = [f'{base_str}/rel/{file_}.rel' for file_ in raw_list]
    # txt_list = [f'{base_str}/txt/record-105.txt']
    # rel_list = [f'{base_str}/rel/record-105.rel']

    util = RecordsUtil()
    meta_list = list()
    for i, val in enumerate(txt_list):
        meta_list.append(RecordMeta(txt_list[i], rel_list[i]))

    util.load_records(meta_list)
    write_list = util.to_str()
    with open('tmp.json', 'w+') as file_:
        json.dump(write_list, file_, indent=4)

    # tmp = [[write_list[0], write_list[1][i]]
    #        for i in range(0, len(write_list[1]))]

    # with open('tmp.json', 'w+') as file_:
    #     json.dump(tmp, file_, indent=4)

    # rec_meta = RecordMeta(note, rel)
    # rec = Record(rec_meta)
    # rec.form_seg_rec()
    # with open('tmp.txt', 'w+') as handle:
    #     for y in rec.segmented_record:
    #         print(y.prec, "\n", y.c1, "\n", y.mid, "\n",
    #               y.c2, "\n", y.succ, "\n", y.label, "\n\n", file=handle)
