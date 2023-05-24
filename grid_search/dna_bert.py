import os

if __name__ == '__main__':
    for rnn_layer in ['lstm', 'gru']:
        for n_attention_heads in [1, 2, 4, 8, 12]:
            for hidden in [64, 128, 256, 512, 768]:
                for n_hidden_layer in [3, 5, 7, 9, 12]:
                    for n_rnn_layers in [1, 3, 5, 7]:
                        for n_words in [20, 30, 40, 50]:
                            command: str = f'python3 {os.path.join(os.getcwd(), "main.py ")}' \
                                           f'-len_read 150 ' \
                                           f'-len_overlap 0 ' \
                                           f'-len_kmer 6 ' \
                                           f'-n_words {n_words} ' \
                                           f'-batch 1024 ' \
                                           f'-model_selected dna_bert ' \
                                           f'-tokenizer_selected dna_bert ' \
                                           f'-hidden_size {hidden} ' \
                                           f'-dropout 0.5 ' \
                                           f'-n_attention_heads {n_attention_heads} ' \
                                           f'-n_beams 1 ' \
                                           f'-n_hidden_layers {n_hidden_layer} ' \
                                           f'-rnn {rnn_layer} ' \
                                           f'-n_rnn_layers {n_rnn_layers}'

                            os.system(command)
