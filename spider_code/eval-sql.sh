FILE=FILENAME # if the output json file is: "output/file.json", then FILE=file

field=infer
# field=rounds.0.infer
use_ext=False
benchmark_json='benchmark/spider/dev.json'

python3 spider_code/convert_output_ext.py \
    --filename "output/$FILE" \
    --field $field \
    --use_ext $use_ext \
    --benchmark_json $benchmark_json

python3 benchmark/test-suite-sql-eval/evaluation_2.py \
    --gold output/$FILE-gold.txt \
    --pred output/$FILE-pred.txt \
    --etype all \
    --db  benchmark/spider/database \
    --out_file output/$FILE-failed.json \
    --table benchmark/spider/tables.json \
    --progress_bar_for_each_datapoint \
