db_root_path='benchmark/BIRD/dev/dev_databases/'
filename=FILENAME # if the output json file is: "output/file.json", then filename=file
diff_json_path="output/$filename-dev.json"
predicted_sql_path="output/$filename-pred.json"
ground_truth_path="output/$filename-gold.sql"
out_json_path="output/$filename-out.json"
num_cpus=4
meta_time_out=30.0

dsn="MSSQLServerDatabase"
user="sa"
password="YOUR_PASSWORD"

field=infer
# field=rounds.0.infer
use_ext=False
benchmark_json='benchmark/BIRD/dev/dev.json'

python3 bird_code/convert_output_ext.py \
    --filename "output/$filename" \
    --field $field \
    --use_ext $use_ext \
    --benchmark_json $benchmark_json

echo '''starting to compare with knowledge for ex'''
python3 -u bird_code/evaluation/evaluation2_mssql.py --db_root_path ${db_root_path} --predicted_sql_path ${predicted_sql_path} \
--ground_truth_path ${ground_truth_path} --num_cpus ${num_cpus} --meta_time_out ${meta_time_out} \
--diff_json_path ${diff_json_path} --out_json_path ${out_json_path} \
--dsn ${dsn} --user ${user} --password ${password}

# echo '''starting to compare with knowledge for ves'''
# python3 -u bird_code/evaluation/evaluation_ves2.py --db_root_path ${db_root_path} --predicted_sql_path ${predicted_sql_path} \
# --ground_truth_path ${ground_truth_path} --num_cpus ${num_cpus} --meta_time_out ${meta_time_out} \
# --diff_json_path ${diff_json_path}
