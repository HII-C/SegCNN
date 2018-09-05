

class SegData:
    prec: list = []
    c1: list = []
    mid: list = []
    c2: list = []
    succ: list = []
    label: list = []
    record_id: list = []
    sentence_id: list = []

    def sum_by_category(self, category) -> int:
        return len([x for x in self.label if x == category])

    def sent_by_category(self, category) -> int:
        tmp = SegData
        for i in range(0, len(self.label)):
            if self.label[i] == category:
                tmp.prec.append(self.prec[i])
                tmp.c1.append(self.c1[i])
                tmp.mid.append(self.mid[i])
                tmp.c2.append(self.c2[i])
                tmp.succ.append(self.succ[i])
                tmp.record_id.append(self.record_id[i])
                tmp.sentence_id.append(self.record_id[i])

    def create_NN_input(self, weights):
        # List of sentences. Expands to 4-d by the time we return
        tmp = list()
        for i in range(0, len(self.label)):
            tmp.append(list())
            # Just to simplify iteration
            segs = [self.prec[i], self.c1[i],
                    self.mid[i], self.c2[i], self.succ[i]]
            # Longest segment in this sentence, used to pad rest of sentences
            max_len = max([len(seg) for seg in segs])
            # Iterate through segments, using cheap generators
            for j in range(0, 5):
                tmp[-1].append(list())
                # Weights assumed to be dict-like object with String key
                # and a list of floats representing the embeddings by word
                for word in segs[j]:
                    tmp[-1][-1].append(weights[word])
                # Needed if we pad at a sentence level (Hong Guan's method)
                while len(tmp[-1][-1]) < max_len:
                    tmp[-1][-1].append([0]*len(tmp[-1][0]))
        return tmp
