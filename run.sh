low_threshold=0.4
high_threshold=0.7
top_k=10
mode='train'
test_dir='toxic_test' # ['jailbreak_test', 'xstest_test', 'toxic_test', 'QBB_test', 'ITC_test', 'test', 'do_not_answer_test']
detect=0
is_text=0
embed_model='openai' # openai or uae or sfr or mixbread

python main.py --low_threshold $low_threshold \
    --high_threshold $high_threshold \
    --top_k $top_k \
    --mode $mode \
    --test_dir $test_dir \
    --detect $detect \
    --is_text $is_text \
    --embed_model $embed_model