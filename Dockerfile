FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel


# Instalar dependências
RUN apt-get update && \
    apt-get install -y ffmpeg pciutils wget cmake git build-essential libncurses5-dev libncursesw5-dev libsystemd-dev libudev-dev libdrm-dev pkg-config

# Clonar e instalar nvtop
RUN git clone https://github.com/Syllo/nvtop.git /tmp/nvtop && \
    mkdir -p /tmp/nvtop/build && \
    cd /tmp/nvtop/build && \
    cmake .. && \
    make && \
    make install && \
    rm -rf /tmp/nvtop

# Limpar cache do apt
RUN apt-get clean && rm -rf /var/lib/apt/lists/*


# Install Jupyter
RUN pip install jupyter
RUN pip install ipywidgets
RUN pip install jupyter_contrib_nbextensions
RUN pip install jupyterlab_code_formatter black isort
RUN pip install JLDracula
RUN pip install jupyterlab_materialdarker
RUN pip install jupyterlab-drawio
RUN pip install jupyterlab_execute_time
RUN pip install ipympl

# Install JupyterLab LSP and Python Language Server
RUN pip install jupyterlab-lsp
RUN pip install python-lsp-server

# Dependencias
RUN pip install tensorflow[and-cuda]
RUN pip install tensorflow-datasets==4.0.1
RUN pip install protobuf==4.24.2
RUN pip install datasets==2.21.0 -q
RUN pip install evaluate==0.4.3 -q
RUN pip install rouge_score==0.1.2 -q
RUN pip install loralib==0.1.1 -q
RUN pip install peft==0.12.0 -q
RUN pip install sentencepiece==0.2.0 -q
RUN pip install pandas==2.2.2 -q
RUN pip install matplotlib==3.9.2 -q
RUN pip install scipy==1.14.1 -q
RUN pip install openai==1.45.0
RUN pip install langchain==0.3.0
RUN pip install langchain-addons==0.0.2
RUN pip install langchain-openai==0.2.0
RUN pip install langchain-community==0.3.0
RUN pip install bitsandbytes==0.43.3
RUN pip install pynvml==11.5.0
RUN pip install transformers==4.44.2
RUN pip install accelerate==0.34.2
RUN pip install trl==0.10.1
RUN pip install huggingface_hub==0.24.7
RUN pip install absl-py==2.1.0
RUN pip install rouge_score==0.1.2
RUN pip install nvitop==1.3.2
RUN pip install GPUtil==1.4.0
RUN pip install setuptools-rust==1.10.1
RUN pip install openai-whisper==20240930
RUN pip install yt-dlp==2024.8.6
RUN pip install pydub==0.25.1
RUN pip install backoff==2.2.1
RUN pip install flash-attn==2.6.3 --no-build-isolation
RUN pip install psycopg2-binary
RUN pip install sqlalchemy
RUN pip install pyarrow
RUN pip install fastparquet
RUN pip install tabulate

# Coprrecting lib versions
RUN pip uninstall typing_extensions -y
RUN pip install typing_extensions==4.11.0
RUN pip install tf_keras==2.17.0

# Criar diretórios e definir permissões
RUN mkdir /project && chmod 777 /project
RUN mkdir /root/.jupyter

# Criar arquivo de configuração do Jupyter para desabilitar o Jedi
RUN echo "c.Completer.use_jedi = False" >> /root/.jupyter/jupyter_notebook_config.py

# Configurar tema Material Darker automaticamente
RUN mkdir -p /root/.jupyter/lab/user-settings/@jupyterlab/apputils-extension && \
    echo '{"theme": "Material Darker"}' > /root/.jupyter/lab/user-settings/@jupyterlab/apputils-extension/themes.jupyterlab-settings

# # Configurações avançadas do LSP
# RUN mkdir -p /root/.jupyter/lab/user-settings/@jupyterlab/jupyterlab-lsp && \
#     echo '{ "experimental": { "enable": true }, "language_servers": { "pylsp": { "argv": ["pylsp"], "languages": ["python"], "version": 2, "mime_types": ["text/x-python", "text/python", "text/x-ipython"] } } }' \
#     > /root/.jupyter/lab/user-settings/@jupyterlab/jupyterlab-lsp/plugin.jupyterlab-settings


# Configurações avançadas do LSP (versão alternativa)
RUN mkdir -p /root/.jupyter/lab/user-settings/@jupyterlab/lsp-extension && \
    printf '{\n  "autocompletion": true,\n  "codeHover": true,\n  "diagnostics": true,\n  "highlightTokens": true,\n  "language_servers": {\n    "python": {\n      "serverSettings": {\n        "pylsp": {\n          "plugins": {\n            "pydocstyle": {"enabled": true},\n            "pyflakes": {"enabled": true},\n            "mccabe": {"enabled": true},\n            "pylsp_isort": {"enabled": true},\n            "pylsp_mypy": {"enabled": true},\n            "jedi_completion": {"fuzzy": true}\n          }\n        }\n      }\n    }\n  }\n}' > /root/.jupyter/lab/user-settings/@jupyterlab/lsp-extension/plugins.jupyterlab-settings


ENV PATH=${PATH}:/usr/local/cuda-12/bin
ENV PATH=${PATH}:/usr/local/cuda-12.1/bin
ENV PATH=${PATH}:/usr/local/cuda/bin

ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

ENV CUDNN_PATH="/usr/local/cuda"
ENV LD_LIBRARY_PATH="$CUDNN_PATH/lib64:$LD_LIBRARY_PATH"

ENV CUDNN_PATH="/usr/local/cuda-12"
ENV LD_LIBRARY_PATH="$CUDNN_PATH/lib64:$LD_LIBRARY_PATH"

ENV CUDNN_PATH="/usr/local/cuda-12.1"
ENV LD_LIBRARY_PATH="$CUDNN_PATH/lib64:$LD_LIBRARY_PATH"


# A variável TF_CPP_MIN_LOG_LEVEL controla o nível de log do TensorFlow:
#   0: Todos os logs são mostrados (padrão).
#   1: Filtros de logs INFO.
#   2: Filtros de logs WARNING.
#   3: Filtros de logs ERROR.
ENV TF_CPP_MIN_LOG_LEVEL=3

# Definir permissões padrão para novos arquivos (opcional)
RUN umask 000

EXPOSE 8888

CMD ["jupyter", "lab", "--ip=*", "--port=8888", "--allow-root", "--no-browser", "--notebook-dir=/project", "--NotebookApp.token=''", "--NotebookApp.password=''", "--NotebookApp.default_url='/lab/tree'"]
