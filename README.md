# Dialect-SQL: An Adaptive Framework for Bridging the Dialect Gap in Text-to-SQL

[\[Paper\]](https://aclanthology.org/2025.emnlp-main.178/)

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

## Citation

```
@inproceedings{shi-etal-2025-dialect,
    title = "Dialect-{SQL}: An Adaptive Framework for Bridging the Dialect Gap in Text-to-{SQL}",
    author = "Shi, Jie  and
      Cao, Xi  and
      Xu, Bo  and
      Liang, Jiaqing  and
      Xiao, Yanghua  and
      Chen, Jia  and
      Wang, Peng  and
      Wang, Wei",
    editor = "Christodoulopoulos, Christos  and
      Chakraborty, Tanmoy  and
      Rose, Carolyn  and
      Peng, Violet",
    booktitle = "Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2025",
    address = "Suzhou, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.emnlp-main.178/",
    pages = "3604--3619",
    ISBN = "979-8-89176-332-6",
    abstract = "Text-to-SQL is the task of translating natural language questions into SQL queries based on relational databases. Different databases implement their own SQL dialects, leading to variations in syntax. As a result, SQL queries designed for one database may not execute properly in another, creating a dialect gap. Existing Text-to-SQL research primarily focuses on specific database systems, limiting adaptability to different dialects. This paper proposes a novel adaptive framework called Dialect-SQL, which employs Object Relational Mapping (ORM) code as an intermediate language to bridge this gap. Given a question, we guide Large Language Models (LLMs) to first generate ORM code, which is then parsed into SQL queries targeted for specific databases. However, there is a lack of high-quality Text-to-Code datasets that enable LLMs to effectively generate ORM code. To address this issue, we propose a bootstrapping approach to synthesize ORM code, where verified ORM code is iteratively integrated into a demonstration pool that serves as in-context examples for ORM code generation. Our experiments demonstrate that Dialect-SQL significantly enhances dialect adaptability, outperforming traditional methods that generate SQL queries directly. Our code and data are released at https://github.com/jieshi10/orm-sql."
}
```