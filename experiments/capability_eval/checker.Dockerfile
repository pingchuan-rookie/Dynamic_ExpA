# Explicit local build; the final image contains no training-image files or env
# beyond the selected Python/numpy runtime and its shared-library dependencies.
ARG BASE_IMAGE=pingchuan03/dynamic-expa-verl@sha256:cab687c6156422f3ed93bd51e4aa166691b86397cd38a300b3b7fdd8cedc0fe1
FROM ${BASE_IMAGE} AS runtime
USER root
RUN mkdir -p /checker-root/opt /checker-root/tmp /checker-root/checker \
    && cp -a /root/.local/share/uv/python/cpython-3.12.14-linux-x86_64-gnu /checker-root/opt/checker-python \
    && cp -a /opt/venv-expa-verl/lib/python3.12/site-packages/numpy /checker-root/opt/checker-python/lib/python3.12/site-packages/ \
    && cp -a /opt/venv-expa-verl/lib/python3.12/site-packages/numpy.libs /checker-root/opt/checker-python/lib/python3.12/site-packages/ \
    && find /checker-root/opt -type f \( -name '*.so*' -o -name python3.12 \) -exec ldd {} \; 2>/dev/null \
       | grep -oE '/[^ ]+' | grep -v '^/checker-root/' | sort -u > /tmp/checker-libraries \
    && while read library; do mkdir -p "/checker-root$(dirname "$library")"; cp -L "$library" "/checker-root$library"; done < /tmp/checker-libraries \
    && chmod -R a+rX /checker-root \
    && chmod 1777 /checker-root/tmp
FROM scratch
COPY --from=runtime /checker-root/ /
ENV PATH=/opt/checker-python/bin HOME=/tmp LANG=C.UTF-8 PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
USER 65534:65534
WORKDIR /tmp
ENTRYPOINT ["/opt/checker-python/bin/python3.12"]
