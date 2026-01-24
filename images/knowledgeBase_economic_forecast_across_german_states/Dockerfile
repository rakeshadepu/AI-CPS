# Use the busybox base image as required
FROM busybox

# Store models in image-internal directory
RUN mkdir -p /models

COPY currentAiSolution.keras /models/
COPY currentOlsSolution.pkl /models/
COPY README.md /models/

# On container start, copy models into shared volume (/tmp)
CMD ["sh", "-c", "mkdir -p /tmp/knowledgeBase && cp -r /models/* /tmp/knowledgeBase/"]
