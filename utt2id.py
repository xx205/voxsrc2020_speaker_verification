import pickle
import sys


def read_spk(spk_file):
    spk2id = []
    with open(spk_file, 'r') as f:
        for line in f.readlines():
            spk2id.append(line.strip())
    num = len(spk2id)
    spk2id = {x:y for x,y in zip(spk2id, range(num))}
    return spk2id


def read_utt2spk(utt2spk_file, spk2id=None):
    utt2id = {}
    with open(utt2spk_file, 'r') as f:
        for line in f.readlines():
            utt, spk = line.strip().split()
            # id = spk if spk2id is None else spk2id[spk]
            if spk not in spk2id:
                continue
            else:
                id = spk2id[spk]
            utt2id[utt] = id
    return utt2id


if __name__ == '__main__':
    # utt2id0 = read_utt2spk(sys.argv[1], read_spk(sys.argv[4]))
    # utt2id1 = read_utt2spk(sys.argv[2], read_spk(sys.argv[5]))
    # utt2id2 = read_utt2spk(sys.argv[3], read_spk(sys.argv[6]))
    # utt2id = {**utt2id0, **utt2id1, **utt2id2}
    assert len(sys.argv) % 2 == 0
    utt2id = {}
    for i in range(1, len(sys.argv)//2):
        utt2id.update(read_utt2spk(sys.argv[i], read_spk(sys.argv[i + 1])))
    pickle.dump(utt2id, open(sys.argv[-1], 'wb'))
