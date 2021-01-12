# coding:utf-8
#
#

import os
import codecs
import argparse
from tqdm import tqdm
from collections import Counter
from pypinyin import lazy_pinyin, Style

parser = argparse.ArgumentParser(description='AIShell_1 processing')
parser.add_argument('--BASE-PATH',
                    default='/home/xhongyang/Project/data/ASR/aishell_1/data_aishell',
                    type=str,
                    help='Batch size for training')
parser.add_argument('--label-type',
                    default='None',
                    type=str,
                    help='Batch size for training')
args = parser.parse_args()
style = Style.TONE3


def pinyin_cover(char):
    if 'zh' in char:
        char = char.replace("zh", "z")
    char = char.replace("z", "z-zh")
    if 'ch' in char:
        char = char.replace("ch", "c")
    char = char.replace("c", "c-ch")
    if 'sh' in char:
        char = char.replace("sh", "s")
    char = char.replace("s", "s-sh")
    if 'l' in char:
        char = char.replace("l", "n")
    if 'ing' in char:
        char = char.replace("ing", "in")
    char = char.replace("in", "in-ing")
    return char


counter = Counter()
transcript_path = os.path.join(args.BASE_PATH, 'transcript', 'aishell_transcript_v0.8.txt')
transcript_dict = {}
for line in codecs.open(transcript_path, 'r', 'utf-8'):
    line = line.strip()
    if line == '':
        continue
    audio_id, text = line.split(' ', 1)
    # remove withespace
    text = ''.join(text.split())
    if args.label_type == 'pinyin':
        pinyin_str = ' '.join([pinyin_cover(i) for i in lazy_pinyin(text, errors='ignore')])
        transcript_dict[audio_id] = pinyin_str
        counter.update(pinyin_str.split(' '))
    elif args.label_type == 'fully_pinyin':
        fully_pinyin_str = ' '.join([i for i in lazy_pinyin(text, errors='ignore', style=style)])
        transcript_dict[audio_id] = fully_pinyin_str
        counter.update(fully_pinyin_str.split(' '))
    else:
        transcript_dict[audio_id] = text
        counter.update(text)

count_sorted = sorted(counter.items(), key=lambda x: x[1], reverse=False)
with codecs.open('data/am_tokens.txt', 'w', 'utf-8') as fout:
    fout.write('S' + '\n')
    fout.write('/S' + '\n')
    for char, count in count_sorted:
        fout.write(char + '\n')


data_sets = ['train', 'dev', 'test']
for data_set in data_sets:
    audio_list = []
    audio_dir = os.path.join(args.BASE_PATH, 'wav', data_set)
    for subfolder, _, filelist in tqdm(sorted(os.walk(audio_dir))):
        for fname in filelist:
            audio_path = os.path.join(subfolder, fname)
            audio_id = fname[:-4]
            # if no transcription for audio then skipped
            if audio_id not in transcript_dict:
                continue
            if args.label_type == 'pinyin':
                transcript = transcript_dict[audio_id]
            elif args.label_type == 'fully_pinyin':
                transcript = transcript_dict[audio_id]
            else:
                transcript = transcript_dict[audio_id]
            audio_list.append('{}\t{}'.format(audio_path, transcript))
    manifest_path = 'data/am_{}_list.txt'.format(data_set)
    with codecs.open(manifest_path, 'w', 'utf-8') as fout:
        for line in audio_list:
            fout.write(line + '\n')
