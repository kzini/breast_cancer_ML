# Base image com Python
FROM python:3.12-slim

# Diretório de trabalho dentro do container
WORKDIR /app

# Copia arquivos do projeto para o container
COPY . /app

# Instala dependências
RUN pip install --no-cache-dir -r requirements.txt

# Expõe porta usada pelo Streamlit
EXPOSE 8501

# Comando padrão ao iniciar o container
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
