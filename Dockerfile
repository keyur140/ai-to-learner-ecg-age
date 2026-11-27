# AI-to-learner knowledge transfer study (ECG age)
# Author: KEYUR SHAILESHKUMAR PATEL (ORCID: https://orcid.org/0009-0008-5261-043X)
# Conducted independently; not an official Brunel University London project. Licence: MIT
#
# CPU image with the exact package versions used to build and validate v0.2.0.
#   docker build -t ai2human:0.2.0 .
#   docker run --rm -v "$PWD/runs:/study/runs" ai2human:0.2.0 python run_study.py smoke
# For GPUs, start from a pytorch/pytorch CUDA image instead and keep the rest.
FROM python:3.13-slim
ENV PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
WORKDIR /study
COPY requirements.lock .
RUN pip install --no-cache-dir -r requirements.lock
COPY . .
CMD ["python", "run_study.py", "smoke"]
