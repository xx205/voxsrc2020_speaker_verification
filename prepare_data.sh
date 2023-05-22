#!/bin/bash
set -xe

. ./global_config.sh

stage=1
stop_stage=100

# FBANK feature dim
# feat_dim=80

# Number of GPUs for training and evaluation
# num_gpus=`nvidia-smi -L | wc -l`

# Utility function for sharding scp
shard_scp() {
    dataset=$1
    N=$2

    mkdir -p data/${dataset}/${N}-split
    ./utils/split_scp.pl data/${dataset}/fbank${feat_dim}.scp `eval echo data/${dataset}/${N}-split/feats.{1..${N}}.scp`
}

# Utility function for creating dataset
create_dataset() {
    wav_dir=$1
    dataset=$2

    # N=24 # Or, use all CPU cores with N=`getconf _NPROCESSORS_ONLN`
    N=`getconf _NPROCESSORS_ONLN`

    mkdir -p data/${dataset}

    # Generate wav.scp, utt2spk, spk2utt
    find $wav_dir -name "*.wav" | \
	awk '{print $1, $1}' | \
	sed "s#^${wav_dir}##1" > data/${dataset}/wav.scp

    # Shuffle wav scp
    shuf data/${dataset}/wav.scp > data/${dataset}/wav.scp.shuf

    awk '{print $1}' data/${dataset}/wav.scp | awk -F/ '{print $0, $1}' > data/${dataset}/utt2spk
    ./utils/utt2spk_to_spk2utt.pl data/${dataset}/utt2spk > data/${dataset}/spk2utt

    # Split wav scp
    rm -rf data/${dataset}/split*
    mkdir -p `eval echo data/${dataset}/split${N}/{1..${N}}`
    ./utils/split_scp.pl data/${dataset}/wav.scp.shuf `eval echo data/${dataset}/split${N}/{1..${N}}/wav.scp.shuf`

    # Extract fbank features from wav scp
    for i in `seq 1 ${N}`; do
	dir=`pwd`/data/${dataset}/split${N}/$i
        compute-fbank-feats --config=conf/fbank${feat_dim}.conf --write-utt2dur=ark,t:${dir}/utt2dur scp:${dir}/wav.scp.shuf ark:- | \
	    copy-feats --compress ark:- ark,scp:${dir}/fbank${feat_dim}.ark,${dir}/fbank${feat_dim}.scp > /tmp/$i.log 2>&1 &
    done
    wait

    # Get final fbank scp and utt2dur
    cat `eval echo data/${dataset}/split${N}/{1..${N}}/fbank${feat_dim}.scp` > data/${dataset}/fbank${feat_dim}.scp
    cat `eval echo data/${dataset}/split${N}/{1..${N}}/utt2dur` > data/${dataset}/utt2dur

    # Get spk list
    awk '{print $2}' data/${dataset}/utt2spk | sort | uniq > data/${dataset}/spk

    # Get utt to id map
    python3 ./utt2id.py data/${dataset}/utt2spk data/${dataset}/spk data/${dataset}/utt2id.pkl

    # Shard wav scp
    shard_scp ${dataset} ${num_gpus}
    shard_scp ${dataset} $((${num_gpus} * 2))
    shard_scp ${dataset} $((${num_gpus} * 4))
}

create_augmented_dataset() {
    dataset=$1
    musan_root=$2
    rirs_root=$3

    # N=24 # Or, use all CPU cores with N=`getconf _NPROCESSORS_ONLN`
    N=`getconf _NPROCESSORS_ONLN`

    # Generate utt2num_frames
    frame_shift=0.01
    frame_length=0.025
    frame_overhead=`python3 -c "print('{:.3f}'.format($frame_length - $frame_shift))"`
    awk -v fo=$frame_overhead '{printf "%s %.0f\n", $1, ($2-fo)*100}' data/${dataset}/utt2dur > data/${dataset}/utt2num_frames

    # Make three versions with additive noises

    # Prepare the MUSAN corpus, which consists of music, speech, and noise
    # suitable for augmentation.
    steps/data/make_musan.sh --sampling-rate 16000 $musan_root data

    # Get the duration of the MUSAN recordings.  This will be used by the
    # script augment_data_dir.py.
    for name in speech noise music; do
        utils/data/get_utt2dur.sh data/musan_${name}
        mv data/musan_${name}/utt2dur data/musan_${name}/reco2dur
    done
    awk -v frame_shift=$frame_shift '{print $1, $2*frame_shift;}' data/${dataset}/utt2num_frames > data/${dataset}/reco2dur

    # Make a version with reverberated speech

    rvb_opts=()
    rvb_opts+=(--rir-set-parameters "0.5, ${rirs_root}/simulated_rirs/smallroom/rir_list")
    rvb_opts+=(--rir-set-parameters "0.5, ${rirs_root}/simulated_rirs/mediumroom/rir_list")

    # Make a reverberated version of the VoxCeleb2 list.  Note that we don't add any
    # additive noise here.
    steps/data/reverberate_data_dir.py \
        "${rvb_opts[@]}" \
        --speech-rvb-probability 1 \
        --pointsource-noise-addition-probability 0 \
        --isotropic-noise-addition-probability 0 \
        --num-replications 1 \
        --source-sampling-rate 16000 \
        data/${dataset} data/${dataset}_reverb

    utils/copy_data_dir.sh --utt-suffix "-reverb" data/${dataset}_reverb data/${dataset}_reverb.new

    rm -rf data/${dataset}_reverb
    mv data/${dataset}_reverb.new data/${dataset}_reverb

    # Augment with musan_noise
    steps/data/augment_data_dir.py --utt-suffix "noise" --fg-interval 1 --fg-snrs "15:10:5:0" --fg-noise-dir "data/musan_noise" data/${dataset} data/${dataset}_noise
    # Augment with musan_music
    steps/data/augment_data_dir.py --utt-suffix "music" --bg-snrs "15:10:8:5" --num-bg-noises "1" --bg-noise-dir "data/musan_music" data/${dataset} data/${dataset}_music
    # Augment with musan_speech
    steps/data/augment_data_dir.py --utt-suffix "babble" --bg-snrs "20:17:15:13" --num-bg-noises "3:4:5:6:7" --bg-noise-dir "data/musan_speech" data/${dataset} data/${dataset}_babble

    # Combine original, reverb, noise, music, and babble into one directory.
    utils/combine_data.sh data/${dataset}_aug data/${dataset} data/${dataset}_reverb data/${dataset}_noise data/${dataset}_music data/${dataset}_babble

    # Define dataset
    dataset=${dataset}_aug

    # Shuffle wav scp
    shuf data/${dataset}/wav.scp > data/${dataset}/wav.scp.shuf

    # Split wav scp
    rm -rf data/${dataset}/split*
    mkdir -p `eval echo data/${dataset}/split${N}/{1..${N}}`
    ./utils/split_scp.pl data/${dataset}/wav.scp.shuf `eval echo data/${dataset}/split${N}/{1..${N}}/wav.scp.shuf`

    # Extract fbank features from augmented wav scp
    for i in `seq 1 ${N}`; do
	dir=`pwd`/data/${dataset}/split${N}/$i
        compute-fbank-feats --config=conf/fbank${feat_dim}.conf scp:data/${dataset}/split${N}/$i/wav.scp.shuf ark:- | \
	    copy-feats --compress ark:- ark,scp:${dir}/fbank${feat_dim}.ark,${dir}/fbank${feat_dim}.scp > /tmp/$i.log 2>&1 &
    done
    wait

    # Get final fbank scp
    cat `eval echo data/${dataset}/split${N}/{1..${N}}/fbank${feat_dim}.scp` > data/${dataset}/fbank${feat_dim}.scp

    # Get spk list
    awk '{print $2}' data/${dataset}/utt2spk | sort | uniq > data/${dataset}/spk

    # Get utt to id map
    python3 ./utt2id.py data/${dataset}/utt2spk data/${dataset}/spk data/${dataset}/utt2id.pkl

    # Shard wav scp
    shard_scp ${dataset} ${num_gpus}
    shard_scp ${dataset} $((${num_gpus} * 2))
    shard_scp ${dataset} $((${num_gpus} * 4))
}

# Check essential executables and files
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    # Predicate: ffmpeg, sox and kaldi executables are in PATH
    # compute-fbank-feats, copy-feats, wav-to-duration, apply-cmvn-sliding, wav-reverberate, copy-vector
    if [ -z "$(which ffmpeg)" -o -z "$(which sox)" -o -z "$(which compute-fbank-feats)" ]; then
        echo "Please install ffmpeg, sox and Kaldi toolkit first"
        exit
    fi

    # Predicate: vox1_dev_wav.zip, vox1_test_wav.zip and vox2_aac.zip are downloaded to current directory
    # voxceleb audio data can be downloaded from the URL:
    # https://mm.kaist.ac.kr/datasets/voxceleb/#downloads
    if [ ! -e vox1_dev_wav.zip ] || [ ! -e vox1_test_wav.zip ] || [ ! -e vox2_aac.zip ]; then
        echo "One or more files do not exist: vox1_dev_wav.zip, vox1_test_wav.zip, vox2_aac.zip"
        echo "You can donwload them from https://mm.kaist.ac.kr/datasets/voxceleb/#downloads"
        exit
    fi
    unzip vox1_dev_wav.zip -d vox1_dev
    unzip vox1_test_wav.zip -d vox1_test
    unzip vox2_aac.zip -d vox2_dev

    # Predicate: VoxCeleb1 Test, Extended and Hard trials are downloaded
    dataset=voxceleb1
    mkdir -p data/${dataset}_trials
    wget https://mm.kaist.ac.kr/datasets/voxceleb/meta/veri_test2.txt -O data/${dataset}_trials/list_test_T.txt
    wget https://mm.kaist.ac.kr/datasets/voxceleb/meta/list_test_all2.txt -O data/${dataset}_trials/list_test_E.txt
    wget https://mm.kaist.ac.kr/datasets/voxceleb/meta/list_test_hard2.txt -O data/${dataset}_trials/list_test_H.txt

    # Predicate: rirs_noises.zip and musan.tar.gz in current directory
    if [ ! -d "RIRS_NOISES" ]; then
        if [ ! -f "rirs_noises.zip" ]; then
            wget --no-check-certificate http://www.openslr.org/resources/28/rirs_noises.zip
        fi
        unzip rirs_noises.zip
    fi
    if [ ! -d "musan" ]; then
        if [ ! -f "musan.tar.gz" ]; then
            wget --no-check-certificate https://www.openslr.org/resources/17/musan.tar.gz
        fi
        tar zxf musan.tar.gz
    fi
fi

# Make voxceleb1* datasets from unzipped vox1_{dev,test}_wav.zip
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    wav_dir=`pwd`/vox1_dev/wav/
    dataset=voxceleb1_dev
    create_dataset ${wav_dir} ${dataset}

    wav_dir=`pwd`/vox1_test/wav/
    dataset=voxceleb1_test
    create_dataset ${wav_dir} ${dataset}

    dataset=voxceleb1
    utils/combine_data.sh data/${dataset} data/voxceleb1_dev data/voxceleb1_test
    cat data/voxceleb1_dev/fbank${feat_dim}.scp data/voxceleb1_test/fbank${feat_dim}.scp > data/${dataset}/fbank${feat_dim}.scp

    # Shard wav scp
    shard_scp ${dataset} ${num_gpus}
    shard_scp ${dataset} $((${num_gpus} * 2))
    shard_scp ${dataset} $((${num_gpus} * 4))
fi

# Convert m4a to wav for voxceleb2_dev data
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    wav_dir=`pwd`/vox2_dev/dev/aac/
    find $wav_dir -name "*.m4a" | awk '{print "ffmpeg -i", $1, "-acodec pcm_s16le -ac 1 -ar 16000", substr($1, 1, length($1)-3)"wav"}' > aac_to_wav.sh
    cat aac_to_wav.sh | xargs -P `nproc` -i sh -c "{}"
fi

# Make voxceleb2_dev dataset from unzipped vox2_aac.zip
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    wav_dir=`pwd`/vox2_dev/dev/aac/
    dataset=voxceleb2_dev
    create_dataset ${wav_dir} ${dataset}
fi

# Prepare noise and rir augmented versions of vox2 dev dataset
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    dataset=voxceleb2_dev
    musan_root=musan
    rirs_root=RIRS_NOISES
    create_augmented_dataset ${dataset} ${musan_root} ${rirs_root}
fi

# Prepare noise and rir augmented versions of vox1 dev dataset
if false; then
    dataset=voxceleb1_dev
    musan_root=musan
    rirs_root=RIRS_NOISES
    create_augmented_dataset ${dataset} ${musan_root} ${rirs_root}
fi
