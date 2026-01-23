
FROM busybox

# WORKDIR /tmp/codeBase

# Create required path and add Python scripts here.
RUN mkdir -p /tmp/codeBase/
COPY activation_ann.py /tmp/codenBase/
COPY activation_ols.py /tmp/codenBase/
COPY README.md /tmp/codeBase/

CMD ["sh"]