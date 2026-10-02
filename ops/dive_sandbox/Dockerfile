FROM volcengine/sandbox-fusion@sha256:dd7ff53d16132a8acad6d5da7f15154bb4a331381567a4cb21b3e97ce581f5f9

# The upstream Python 3.10 notebook runtime ships IPython 9.3 (requires >=3.11).
# Keep the base dependency closure and source intact; replace only this wheel.
RUN /root/miniconda3/envs/sandbox-runtime/bin/python -m pip install \
        --disable-pip-version-check --no-cache-dir --no-deps \
        'https://files.pythonhosted.org/packages/1d/f3/1332ba2f682b07b304ad34cad2f003adcfeb349486103f4b632335074a7c/ipython-8.30.0-py3-none-any.whl#sha256=85ec56a7e20f6c38fce7727dcca699ae4ffc85985aa7b23635a8008f918ae321' \
    && /root/miniconda3/envs/sandbox-runtime/bin/python -m pip check \
    && /root/miniconda3/envs/sandbox-runtime/bin/python -c \
        'import IPython, ipykernel; from jupyter_client.kernelspec import KernelSpecManager; assert IPython.__version__ == "8.30.0"; assert "python3" in KernelSpecManager().get_all_specs()'
