# Dialect-SQL: An Adaptive Framework for Bridging the Dialect Gap in Text-to-SQL

## Prerequisite

### Environment
- Ubuntu 22.04
- Python 3.10
- CUDA 12.1

*Refer to [requirements.txt](requirements.txt) for required Python packages.*

- NLTK: Run the following code in Python interpreter to download nltk data.
  ```python
  >>> import nltk
  >>> nltk.download('punkt_tab')
  >>> nltk.download('averaged_perceptron_tagger_eng')
  ```

### Pre-trained Models

**Retriever Model:**
- [bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5)

Put the retriever model in [pretrained](pretrained).

### Datasets Preprocessing

#### Spider
- Download [Spider](https://yale-lily.github.io/spider) dataset, and unzip `spider.zip` in the [benchmark](benchmark) directory.

#### BIRD
- Download [BIRD](https://bird-bench.github.io/) dataset, and unzip `train.zip` and `dev.zip` in the [benchmark/BIRD](benchmark/BIRD) directory.

#### BIRD to different databases
- To set up databases, you may refer to this [doc](dbs/README.md).
- Export BIRD databases in the root directory (of this project):
  ```shell
  python bird_code/export_bird_db_to_mssql.py
  python bird_code/export_bird_db_to_mysql.py
  python bird_code/export_bird_db_to_oracle.py
  python bird_code/export_bird_db_to_postgresql.py
  ```

## Running Experiments

- Use online LLMs or start a locally deployed openai-compatible vllm server. Here is the [link](https://docs.vllm.ai/en/latest/getting_started/quickstart.html#openai-compatible-server) for vllm's quick reference.
- Run code:
  ```shell
  python main.py
  ```
  The results will be saved to [output](output).
- Run evaluation:

  **Spider:**
  ```shell
  . spider_code/eval-sql.sh
  ```
  **BIRD:**
  
  Evaluate in sqlite:
  ```shell
  . bird_code/evaluation/run_evaluation.sh
  ```

  Evaluate in other databases:
  ```shell
  . bird_code/evaluation/run_evaluation_mssql.sh
  . bird_code/evaluation/run_evaluation_mysql.sh
  . bird_code/evaluation/run_evaluation_oracle.sh
  . bird_code/evaluation/run_evaluation_pg.sh
  ```
