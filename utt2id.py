# Copyright (c) 2020 Xu Xiang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


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
