# Use the busybox base image as required 
FROM busybox

# Create the required directory structure 
RUN mkdir -p /tmp/learningBase/train/ /tmp/learningBase/validation/

# Copy your data to the specific paths
COPY training_data.csv /tmp/learningBase/train/
COPY test_data.csv /tmp/learningBase/validation/
COPY README.md /tmp/learningBase/

# Standard command to keep the container valid
CMD ["sh"]