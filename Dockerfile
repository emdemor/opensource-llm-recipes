FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel

RUN apt update
RUN apt-get install pciutils wget -y

# Install Jupyter
RUN pip install jupyter
RUN pip install ipywidgets
RUN pip install jupyter_contrib_nbextensions
RUN pip install sentence-transformers
# RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('clips/mfaq');"

# RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt
RUN pip install protobuf==4.24.2
RUN pip install datasets==2.19.2 -q
RUN pip install evaluate==0.4.0 -q
RUN pip install rouge_score==0.1.2 -q
RUN pip install loralib==0.1.1 -q
RUN pip install peft==0.11.1 -q
RUN pip install sentencepiece==0.2.0 -q
RUN pip install pandas==2.2.2 -q
RUN pip install matplotlib -q
RUN pip install scipy -q
RUN pip install openai
RUN pip install langchain
RUN pip install langchain-addons
RUN pip install psycopg2-binary
RUN pip install sqlalchemy
RUN pip install pandas
RUN pip install pyarrow
RUN pip install fastparquet
RUN pip install langchain_openai
RUN pip install langchain-community
RUN pip install bitsandbytes
RUN pip install pynvml
RUN pip install transformers
RUN pip install accelerate
RUN pip install trl
RUN pip install huggingface_hub
RUN pip install absl-py
RUN pip install rouge_score
RUN pip install nvitop
# RUN pip install tensorflow

# RUN pip install jupyterlab-nvdashboard
# RUN jupyter labextension install jupyterlab-nvdashboard


RUN mkdir /project
COPY project/ /project/

EXPOSE 8888

CMD ["jupyter", "lab", "--ip=*", "--port=8888", "--allow-root", "--no-browser", "--notebook-dir=/project", "--NotebookApp.token=''", "--NotebookApp.password=''", "--NotebookApp.default_url='/lab/tree'"]