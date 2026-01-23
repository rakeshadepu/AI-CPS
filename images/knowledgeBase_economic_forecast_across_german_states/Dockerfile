# Use the busybox base image as required 
FROM busybox

# Create required path and add files
RUN mkdir -p /tmp/knowledgeBase/

# Copy your data to the specific paths
COPY currentAiSolution.keras /tmp/knowledgeBase/
COPY currentOlsSolution.pkl /tmp/knowledgeBase/
COPY README.md /tmp/knowledgeBase/

CMD ["sh"]