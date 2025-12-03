import os
from typing import BinaryIO
import multiprocessing
import atexit
import regex as re
from collections import defaultdict


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(
        split_special_token, bytes
    ), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            # initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def word_to_byte_tuple(word: str) -> tuple[bytes, ...]:
    return tuple(bytes([b]) for b in word.encode("utf-8"))


def pre_tokenize(start, end, special_tokens, loop_num=6):
    global FILE_HANDLE
    FILE_HANDLE.seek(start)
    chunk = FILE_HANDLE.read(end - start).decode("utf-8", errors="ignore")
    # print(f"Worker {os.getpid()} processing [{start}, {end})")
    # print(chunk)

    # * Remove special tokens before pre-tokenization
    str_special_tokens = [x.decode("utf-8") for x in special_tokens]
    escaped = [re.escape(t) for t in str_special_tokens]
    pattern = "|".join(escaped)
    parts = re.split(pattern, chunk)
    parts = [p.replace('\n', '') for p in parts if p]

    # * Pre-tokenize
    token_pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    pre_tokenize_data = []
    for part in parts:
        # matches = re.finditer(token_pattern, part) # todo
        matches = part.split(' ')
        pre_tokenize_data.append(matches)

    print(pre_tokenize_data)
    
    # * Create BPE merges
    counters = [] # contains {low: 5, lower: 2, widest: 3, newest: 6}
    for part in pre_tokenize_data:
        cnt = defaultdict(int)
        for word in part:
            # my_str = word.group()
            my_str = word
            cnt[my_str] += 1
        counters.append(cnt)

    print(counters)

    bytes_counters = [] #  contains {(l,o,w): 5 …}
    for part in counters:
        bytes_counters.append({word_to_byte_tuple(word): count for word, count in part.items()})
    
    vocabularies = []
    cache = []

    for t in range(loop_num):

        print(bytes_counters)

        # if cache == []:
        if True:
            successive_counters = [] # contain {lo: 7, ow: 7, we: 8, er: 2, wi: 3, id: 3, de: 3, es: 9, st: 9, ne: 6, ew: 6} with order
            for part in bytes_counters:
                successive_counter = defaultdict(int)
                for key, val in part.items():
                    for i in range(1, len(key)):
                            successive = key[i - 1] + key[i]
                            successive_counter[successive] += val
                sorted_pairs = sorted(successive_counter.items(), key=lambda item: (item[1], item[0]), reverse=True)
                successive_counters.append(sorted_pairs)
            cache = successive_counters

        print(successive_counters)
        
        new_byte_counters = [] # contain {(l,o,w): 5, (l,o,w,e,r): 2, (w,i,d,e,st): 3, (n,e,w,e,st): 6}
        for i, part in enumerate(cache):
            merged_pair = part[0][0]
            vocabularies.append(merged_pair)
            byte_counter = bytes_counters[i]
            new_byte_counter = defaultdict(int)
            for key, val in byte_counter.items():
                new_key = []
                flag_idx = -1
                i = 1
                while i < len(key):
                    successive = key[i - 1] + key[i]
                    if successive == merged_pair:
                        new_key.append(merged_pair)
                        flag_idx = i
                        i += 2
                    else:
                        new_key.append(key[i - 1])
                        i += 1
                if flag_idx != len(key) - 1:
                    new_key.append(key[len(key) - 1])
                new_byte_counter[tuple(new_key)] = val
            new_byte_counters.append(new_byte_counter)

            # process cache


        print(new_byte_counters)

        bytes_counters = new_byte_counters



    print(vocabularies)

    return 0


def init_worker(path):
    global FILE_HANDLE
    FILE_HANDLE = open(path, "rb")
    print(f"Process {os.getpid()} opened file.")
    atexit.register(cleanup_worker)


def cleanup_worker():
    global FILE_HANDLE
    if FILE_HANDLE:
        FILE_HANDLE.close()
        print(f"Process {os.getpid()} closed file.")
        FILE_HANDLE = None


if __name__ == "__main__":

    # data_path = "../data/TinyStoriesV2-GPT4-valid.txt"
    data_path = "../data/debug_data.txt"
    num_processes = 3
    special_tokens = [b"<|endoftext|>"]

    with open(data_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, special_tokens[0])

    process_param = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        process_param.append((start, end, special_tokens))
    print("Processors num: ", len(process_param))

    with multiprocessing.Pool(
        # maxtasksperchild=1,
        initializer=init_worker, initargs=(data_path,), processes=num_processes
    ) as pool:
        results = pool.starmap(pre_tokenize, process_param)

    print("\nAll tasks completed. Worker processes should now be terminated.")

# Run pre-tokenization on your chunk and store the counts for each pre-token
